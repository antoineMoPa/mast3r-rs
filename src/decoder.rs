//! MASt3R asymmetric decoder: two stacks of blocks, each view cross-attends to the other.

use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use crate::encoder::ENCODER_DIM;
use crate::rope::RopeGrid;
use crate::transformer::{CrossAttention, Mlp, SelfAttention, layer_norm};

pub const DECODER_DIM: usize = 768;
const DECODER_DEPTH: usize = 12;
/// Decoder outputs the DPT head reads, besides the encoder output (hook 0).
const DPT_DECODER_HOOKS: [usize; 3] = [6, 9, 12];

struct DecoderBlock {
    norm1: LayerNorm,
    attn: SelfAttention,
    norm2: LayerNorm,
    norm_y: LayerNorm,
    cross_attn: CrossAttention,
    norm3: LayerNorm,
    mlp: Mlp,
}

impl DecoderBlock {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(DECODER_DIM, vb.pp("norm1"))?,
            attn: SelfAttention::new(DECODER_DIM, vb.pp("attn"))?,
            norm2: layer_norm(DECODER_DIM, vb.pp("norm2"))?,
            norm_y: layer_norm(DECODER_DIM, vb.pp("norm_y"))?,
            cross_attn: CrossAttention::new(DECODER_DIM, vb.pp("cross_attn"))?,
            norm3: layer_norm(DECODER_DIM, vb.pp("norm3"))?,
            mlp: Mlp::transformer_mlp(DECODER_DIM, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        context: &Tensor,
        rope_xs: &RopeGrid,
        rope_context: &RopeGrid,
    ) -> Result<Tensor> {
        let xs = (xs + self.attn.forward(&self.norm1.forward(xs)?, rope_xs)?)?;
        let context = self.norm_y.forward(context)?;
        let xs = (&xs
            + self.cross_attn.forward(
                &self.norm2.forward(&xs)?,
                &context,
                rope_xs,
                rope_context,
            )?)?;
        &xs + self.mlp.forward(&self.norm3.forward(&xs)?)?
    }
}

/// Token features of one view, as consumed by the prediction heads.
pub struct DecodedView {
    /// Encoder output followed by the decoder outputs at `DPT_DECODER_HOOKS`
    /// (the last one after `dec_norm`).
    pub dpt_features: [Tensor; 4],
    /// Encoder output concatenated with the final normed decoder output.
    pub local_features: Tensor,
}

pub struct Decoder {
    decoder_embed: Linear,
    blocks_view1: Vec<DecoderBlock>,
    blocks_view2: Vec<DecoderBlock>,
    dec_norm: LayerNorm,
}

impl Decoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let blocks = |prefix: &str| -> Result<Vec<DecoderBlock>> {
            (0..DECODER_DEPTH)
                .map(|i| DecoderBlock::new(vb.pp(format!("{prefix}.{i}"))))
                .collect()
        };
        Ok(Self {
            decoder_embed: candle_nn::linear(ENCODER_DIM, DECODER_DIM, vb.pp("decoder_embed"))?,
            blocks_view1: blocks("dec_blocks")?,
            blocks_view2: blocks("dec_blocks2")?,
            dec_norm: layer_norm(DECODER_DIM, vb.pp("dec_norm"))?,
        })
    }

    pub fn forward(
        &self,
        encoded1: &Tensor,
        encoded2: &Tensor,
        rope1: &RopeGrid,
        rope2: &RopeGrid,
    ) -> Result<(DecodedView, DecodedView)> {
        let mut xs1 = self.decoder_embed.forward(encoded1)?;
        let mut xs2 = self.decoder_embed.forward(encoded2)?;
        let mut hooked1 = Vec::with_capacity(DPT_DECODER_HOOKS.len());
        let mut hooked2 = Vec::with_capacity(DPT_DECODER_HOOKS.len());
        for (depth, (block1, block2)) in
            self.blocks_view1.iter().zip(&self.blocks_view2).enumerate()
        {
            // Both views attend to the other view's output from the previous depth.
            let next1 = block1.forward(&xs1, &xs2, rope1, rope2)?;
            let next2 = block2.forward(&xs2, &xs1, rope2, rope1)?;
            xs1 = next1;
            xs2 = next2;
            if DPT_DECODER_HOOKS.contains(&(depth + 1)) {
                hooked1.push(xs1.clone());
                hooked2.push(xs2.clone());
            }
        }
        let view = |encoded: &Tensor, hooked: Vec<Tensor>| -> Result<DecodedView> {
            let [hook6, hook9, hook12]: [Tensor; 3] =
                hooked.try_into().expect("one decoder output per hook");
            let hook12 = self.dec_norm.forward(&hook12)?;
            Ok(DecodedView {
                local_features: Tensor::cat(&[encoded, &hook12], 2)?,
                dpt_features: [encoded.clone(), hook6, hook9, hook12],
            })
        };
        Ok((view(encoded1, hooked1)?, view(encoded2, hooked2)?))
    }
}
