//! DPT (Dense Prediction Transformer) head: fuses 4 token maps into a full-resolution
//! map of 4 channels (xyz + confidence, before activation).

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, VarBuilder};

use crate::decoder::DECODER_DIM;
use crate::encoder::ENCODER_DIM;

const FEATURE_DIM: usize = 256;
const HEAD_HIDDEN_DIM: usize = FEATURE_DIM / 2;
const DPT_OUTPUT_CHANNELS: usize = 4;
/// Channel count of each hooked token map after `act_postprocess`.
const LAYER_DIMS: [usize; 4] = [96, 192, 384, 768];
const HOOK_INPUT_DIMS: [usize; 4] = [ENCODER_DIM, DECODER_DIM, DECODER_DIM, DECODER_DIM];

fn conv(
    in_dim: usize,
    out_dim: usize,
    kernel: usize,
    padding: usize,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let config = Conv2dConfig {
        padding,
        ..Default::default()
    };
    candle_nn::conv2d(in_dim, out_dim, kernel, config, vb)
}

fn upsample_2x(xs: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = xs.dims4()?;
    xs.upsample_bilinear2d(2 * h, 2 * w, true)
}

/// Brings one hooked token map to its pyramid scale: 4x up, 2x up, 1x, 2x down.
enum Resample {
    Up(ConvTranspose2d),
    Identity,
    Down(Conv2d),
}

struct HookAdapter {
    project: Conv2d,
    resample: Resample,
}

impl HookAdapter {
    fn new(hook: usize, vb: VarBuilder) -> Result<Self> {
        let dim = LAYER_DIMS[hook];
        let up = |factor: usize| -> Result<Resample> {
            let config = ConvTranspose2dConfig {
                stride: factor,
                ..Default::default()
            };
            Ok(Resample::Up(candle_nn::conv_transpose2d(
                dim,
                dim,
                factor,
                config,
                vb.pp("1"),
            )?))
        };
        let resample = match hook {
            0 => up(4)?,
            1 => up(2)?,
            2 => Resample::Identity,
            3 => {
                let config = Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                };
                Resample::Down(candle_nn::conv2d(dim, dim, 3, config, vb.pp("1"))?)
            }
            _ => panic!("DPT has 4 hooks, got hook {hook}"),
        };
        Ok(Self {
            project: conv(HOOK_INPUT_DIMS[hook], dim, 1, 0, vb.pp("0"))?,
            resample,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.project.forward(xs)?;
        match &self.resample {
            Resample::Up(layer) => layer.forward(&xs),
            Resample::Identity => Ok(xs),
            Resample::Down(layer) => layer.forward(&xs),
        }
    }
}

struct ResidualConvUnit {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResidualConvUnit {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: conv(FEATURE_DIM, FEATURE_DIM, 3, 1, vb.pp("conv1"))?,
            conv2: conv(FEATURE_DIM, FEATURE_DIM, 3, 1, vb.pp("conv2"))?,
        })
    }
}

impl Module for ResidualConvUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.conv1.forward(&xs.relu()?)?;
        let out = self.conv2.forward(&out.relu()?)?;
        out + xs
    }
}

struct FeatureFusionBlock {
    skip_unit: ResidualConvUnit,
    unit: ResidualConvUnit,
    out_conv: Conv2d,
}

impl FeatureFusionBlock {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            skip_unit: ResidualConvUnit::new(vb.pp("resConfUnit1"))?,
            unit: ResidualConvUnit::new(vb.pp("resConfUnit2"))?,
            out_conv: conv(FEATURE_DIM, FEATURE_DIM, 1, 0, vb.pp("out_conv"))?,
        })
    }

    fn forward(&self, xs: &Tensor, skip: Option<&Tensor>) -> Result<Tensor> {
        let xs = match skip {
            Some(skip) => (xs + self.skip_unit.forward(skip)?)?,
            None => xs.clone(),
        };
        self.out_conv
            .forward(&upsample_2x(&self.unit.forward(&xs)?)?)
    }
}

pub struct DptHead {
    hook_adapters: Vec<HookAdapter>,
    /// scratch.layer{1..4}_rn: project every pyramid level to FEATURE_DIM.
    layer_projections: Vec<Conv2d>,
    /// scratch.refinenet{1..4}, index 0 is the finest level.
    refinenets: Vec<FeatureFusionBlock>,
    head_conv1: Conv2d,
    head_conv2: Conv2d,
    head_conv3: Conv2d,
}

impl DptHead {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let scratch = vb.pp("scratch");
        let no_bias_3x3 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            hook_adapters: (0..4)
                .map(|hook| HookAdapter::new(hook, vb.pp(format!("act_postprocess.{hook}"))))
                .collect::<Result<_>>()?,
            layer_projections: LAYER_DIMS
                .iter()
                .enumerate()
                .map(|(i, &dim)| {
                    let name = format!("layer{}_rn", i + 1);
                    candle_nn::conv2d_no_bias(dim, FEATURE_DIM, 3, no_bias_3x3, scratch.pp(name))
                })
                .collect::<Result<_>>()?,
            refinenets: (1..=4)
                .map(|i| FeatureFusionBlock::new(scratch.pp(format!("refinenet{i}"))))
                .collect::<Result<_>>()?,
            head_conv1: conv(FEATURE_DIM, HEAD_HIDDEN_DIM, 3, 1, vb.pp("head.0"))?,
            head_conv2: conv(HEAD_HIDDEN_DIM, HEAD_HIDDEN_DIM, 3, 1, vb.pp("head.2"))?,
            head_conv3: conv(HEAD_HIDDEN_DIM, DPT_OUTPUT_CHANNELS, 1, 0, vb.pp("head.4"))?,
        })
    }

    /// `features`: 4 token maps [batch, patch_h * patch_w, dim].
    /// Returns [batch, DPT_OUTPUT_CHANNELS, patch_h * 16, patch_w * 16].
    pub fn forward(
        &self,
        features: &[Tensor; 4],
        patch_h: usize,
        patch_w: usize,
    ) -> Result<Tensor> {
        let pyramid: Vec<Tensor> = features
            .iter()
            .zip(&self.hook_adapters)
            .zip(&self.layer_projections)
            .map(|((tokens, adapter), projection)| {
                let (batch, _, dim) = tokens.dims3()?;
                let spatial = tokens
                    .transpose(1, 2)?
                    .reshape((batch, dim, patch_h, patch_w))?;
                projection.forward(&adapter.forward(&spatial)?)
            })
            .collect::<Result<_>>()?;

        // Coarsest level first; an odd patch grid makes the 2x-upsampled map one row/col too big.
        let mut path = self.refinenets[3].forward(&pyramid[3], None)?;
        for level in (0..3).rev() {
            let (_, _, h, w) = pyramid[level].dims4()?;
            path = path.narrow(2, 0, h)?.narrow(3, 0, w)?;
            path = self.refinenets[level].forward(&path, Some(&pyramid[level]))?;
        }

        let out = upsample_2x(&self.head_conv1.forward(&path)?)?;
        self.head_conv3
            .forward(&self.head_conv2.forward(&out)?.relu()?)
    }
}
