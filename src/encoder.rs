//! MASt3R ViT-Large image encoder (CroCo style, 2D RoPE, no class token).

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, VarBuilder};

use crate::rope::RopeGrid;
use crate::transformer::{Mlp, SelfAttention, layer_norm};

pub const PATCH_SIZE: usize = 16;
pub const ENCODER_DIM: usize = 1024;
const ENCODER_DEPTH: usize = 24;

struct EncoderBlock {
    norm1: LayerNorm,
    attn: SelfAttention,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl EncoderBlock {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(ENCODER_DIM, vb.pp("norm1"))?,
            attn: SelfAttention::new(ENCODER_DIM, vb.pp("attn"))?,
            norm2: layer_norm(ENCODER_DIM, vb.pp("norm2"))?,
            mlp: Mlp::transformer_mlp(ENCODER_DIM, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor, rope: &RopeGrid) -> Result<Tensor> {
        let xs = (xs + self.attn.forward(&self.norm1.forward(xs)?, rope)?)?;
        &xs + self.mlp.forward(&self.norm2.forward(&xs)?)?
    }
}

pub struct Encoder {
    patch_embed: Conv2d,
    blocks: Vec<EncoderBlock>,
    norm: LayerNorm,
}

impl Encoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let patch_config = Conv2dConfig {
            stride: PATCH_SIZE,
            ..Default::default()
        };
        Ok(Self {
            patch_embed: candle_nn::conv2d(
                3,
                ENCODER_DIM,
                PATCH_SIZE,
                patch_config,
                vb.pp("patch_embed.proj"),
            )?,
            blocks: (0..ENCODER_DEPTH)
                .map(|i| EncoderBlock::new(vb.pp(format!("enc_blocks.{i}"))))
                .collect::<Result<_>>()?,
            norm: layer_norm(ENCODER_DIM, vb.pp("enc_norm"))?,
        })
    }

    /// `image`: [batch, 3, H, W] in [-1, 1]. Returns [batch, tokens, ENCODER_DIM].
    pub fn forward(&self, image: &Tensor, rope: &RopeGrid) -> Result<Tensor> {
        let mut xs = self
            .patch_embed
            .forward(image)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        for block in &self.blocks {
            xs = block.forward(&xs, rope)?;
        }
        self.norm.forward(&xs)
    }
}
