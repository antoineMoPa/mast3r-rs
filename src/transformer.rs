//! Transformer building blocks shared by the encoder and the decoder.

use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use crate::rope::RopeGrid;

pub const LAYER_NORM_EPS: f64 = 1e-6;
pub const HEAD_DIM: usize = 64;
const MLP_RATIO: usize = 4;

pub fn layer_norm(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    candle_nn::layer_norm(dim, LAYER_NORM_EPS, vb)
}

/// fc1 -> exact (erf) GELU -> fc2.
pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(in_dim, hidden_dim, vb.pp("fc1"))?,
            fc2: candle_nn::linear(hidden_dim, out_dim, vb.pp("fc2"))?,
        })
    }

    pub fn transformer_mlp(dim: usize, vb: VarBuilder) -> Result<Self> {
        Self::new(dim, dim * MLP_RATIO, dim, vb)
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(xs)?.gelu_erf()?)
    }
}

/// [batch, tokens, heads * HEAD_DIM] -> [batch, heads, tokens, HEAD_DIM]
fn split_heads(xs: &Tensor) -> Result<Tensor> {
    let (batch, token_count, dim) = xs.dims3()?;
    xs.reshape((batch, token_count, dim / HEAD_DIM, HEAD_DIM))?
        .transpose(1, 2)?
        .contiguous()
}

/// [batch, heads, tokens, HEAD_DIM] -> [batch, tokens, heads * HEAD_DIM]
fn merge_heads(xs: &Tensor) -> Result<Tensor> {
    let (batch, heads, token_count, head_dim) = xs.dims4()?;
    xs.transpose(1, 2)?
        .reshape((batch, token_count, heads * head_dim))
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    // The fused kernel only exists on Metal.
    if q.device().is_metal() {
        return candle_nn::ops::sdpa(q, k, v, None, false, scale, 1.0);
    }
    let scores = (q.matmul(&k.t()?)? * scale as f64)?;
    candle_nn::ops::softmax_last_dim(&scores)?.matmul(v)
}

pub struct SelfAttention {
    qkv: Linear,
    proj: Linear,
}

impl SelfAttention {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: candle_nn::linear(dim, 3 * dim, vb.pp("qkv"))?,
            proj: candle_nn::linear(dim, dim, vb.pp("proj"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor, rope: &RopeGrid) -> Result<Tensor> {
        let dim = xs.dim(2)?;
        let qkv = self.qkv.forward(xs)?;
        let q = rope.apply(&split_heads(&qkv.narrow(2, 0, dim)?)?)?;
        let k = rope.apply(&split_heads(&qkv.narrow(2, dim, dim)?)?)?;
        let v = split_heads(&qkv.narrow(2, 2 * dim, dim)?)?;
        self.proj.forward(&merge_heads(&attention(&q, &k, &v)?)?)
    }
}

pub struct CrossAttention {
    projq: Linear,
    projk: Linear,
    projv: Linear,
    proj: Linear,
}

impl CrossAttention {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            projq: candle_nn::linear(dim, dim, vb.pp("projq"))?,
            projk: candle_nn::linear(dim, dim, vb.pp("projk"))?,
            projv: candle_nn::linear(dim, dim, vb.pp("projv"))?,
            proj: candle_nn::linear(dim, dim, vb.pp("proj"))?,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        context: &Tensor,
        rope_xs: &RopeGrid,
        rope_context: &RopeGrid,
    ) -> Result<Tensor> {
        let q = rope_xs.apply(&split_heads(&self.projq.forward(xs)?)?)?;
        let k = rope_context.apply(&split_heads(&self.projk.forward(context)?)?)?;
        let v = split_heads(&self.projv.forward(context)?)?;
        self.proj.forward(&merge_heads(&attention(&q, &k, &v)?)?)
    }
}
