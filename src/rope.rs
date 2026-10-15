//! 2D rotary position embedding, as in CroCo / MASt3R.
//!
//! The head dimension is split in two halves: the first half is rotated by the
//! token's row (y) in the patch grid, the second half by its column (x). Each
//! half is a regular 1D RoPE (non-interleaved, `rotate_half` style).

use candle_core::{D, DType, Device, Result, Tensor};

const ROPE_BASE_FREQUENCY: f32 = 100.0;

/// cos/sin tables indexed by token, for one patch grid.
pub struct RopeGrid {
    cos_y: Tensor,
    sin_y: Tensor,
    cos_x: Tensor,
    sin_x: Tensor,
    half_head_dim: usize,
}

impl RopeGrid {
    /// Tables for a `patch_h` x `patch_w` grid, tokens in row-major order.
    pub fn new(
        patch_h: usize,
        patch_w: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        assert_eq!(head_dim % 4, 0, "head_dim must split into two even halves");
        let half_head_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_head_dim)
            .step_by(2)
            .map(|i| 1.0 / ROPE_BASE_FREQUENCY.powf(i as f32 / half_head_dim as f32))
            .collect();

        let token_count = patch_h * patch_w;
        let mut angles_y = Vec::with_capacity(token_count * inv_freq.len());
        let mut angles_x = Vec::with_capacity(token_count * inv_freq.len());
        for row in 0..patch_h {
            for col in 0..patch_w {
                angles_y.extend(inv_freq.iter().map(|f| row as f32 * f));
                angles_x.extend(inv_freq.iter().map(|f| col as f32 * f));
            }
        }
        let angles_y = Tensor::from_vec(angles_y, (token_count, inv_freq.len()), device)?;
        let angles_x = Tensor::from_vec(angles_x, (token_count, inv_freq.len()), device)?;
        Ok(Self {
            cos_y: angles_y.cos()?.to_dtype(dtype)?,
            sin_y: angles_y.sin()?.to_dtype(dtype)?,
            cos_x: angles_x.cos()?.to_dtype(dtype)?,
            sin_x: angles_x.sin()?.to_dtype(dtype)?,
            half_head_dim,
        })
    }

    /// Rotates `tokens`: [batch, heads, token_count, head_dim].
    pub fn apply(&self, tokens: &Tensor) -> Result<Tensor> {
        let y_half = tokens
            .narrow(D::Minus1, 0, self.half_head_dim)?
            .contiguous()?;
        let x_half = tokens
            .narrow(D::Minus1, self.half_head_dim, self.half_head_dim)?
            .contiguous()?;
        let y_half = candle_nn::rotary_emb::rope(&y_half, &self.cos_y, &self.sin_y)?;
        let x_half = candle_nn::rotary_emb::rope(&x_half, &self.cos_x, &self.sin_x)?;
        Tensor::cat(&[y_half, x_half], D::Minus1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference formula: per half, t * cos(dup) + rotate_half(t) * sin(dup).
    #[test]
    fn matches_rotate_half_formula() -> Result<()> {
        let device = Device::Cpu;
        let (patch_h, patch_w, head_dim) = (3, 5, 8);
        let token_count = patch_h * patch_w;
        let tokens = Tensor::randn(0f32, 1.0, (1, 2, token_count, head_dim), &device)?;
        let rotated = RopeGrid::new(patch_h, patch_w, head_dim, DType::F32, &device)?
            .apply(&tokens)?
            .squeeze(0)?
            .to_vec3::<f32>()?;
        let tokens = tokens.squeeze(0)?.to_vec3::<f32>()?;

        let half = head_dim / 2;
        let quarter = half / 2;
        for head in 0..2 {
            for token in 0..token_count {
                let positions = [token / patch_w, token % patch_w];
                for (half_index, position) in positions.iter().enumerate() {
                    let t = &tokens[head][token][half_index * half..(half_index + 1) * half];
                    for i in 0..half {
                        let freq = 1.0
                            / ROPE_BASE_FREQUENCY.powf((2 * (i % quarter)) as f32 / half as f32);
                        let angle = *position as f32 * freq;
                        let rotate_half = if i < quarter {
                            -t[i + quarter]
                        } else {
                            t[i - quarter]
                        };
                        let expected = t[i] * angle.cos() + rotate_half * angle.sin();
                        let got = rotated[head][token][half_index * half + i];
                        assert!(
                            (expected - got).abs() < 1e-5,
                            "token {token} dim {i}: {expected} vs {got}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
