//! Full MASt3R stereo model: encoder + decoder + per-view prediction heads.

use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::decoder::{DECODER_DIM, DecodedView, Decoder};
use crate::dpt::DptHead;
use crate::encoder::{ENCODER_DIM, Encoder, PATCH_SIZE};
use crate::rope::RopeGrid;
use crate::transformer::{HEAD_DIM, Mlp};

pub const DESCRIPTOR_DIM: usize = 24;
/// Descriptor channels + 1 descriptor-confidence channel.
const LOCAL_FEATURE_CHANNELS: usize = DESCRIPTOR_DIM + 1;
const LOCAL_FEATURES_MLP_RATIO: usize = 4;
/// pts3d direction is divided by max(norm, this) before scaling.
const PTS3D_MIN_NORM: f64 = 1e-8;
/// conf = CONF_MIN + exp(raw)
const CONF_MIN: f64 = 1.0;

/// Dense outputs for one view, all [H, W, C] f32 at input image resolution.
/// Both views' points are expressed in the camera frame of view 1.
pub struct ViewPrediction {
    pub pts3d: Tensor,
    pub conf: Tensor,
    /// Unit-length matching descriptors.
    pub desc: Tensor,
    pub desc_conf: Tensor,
}

struct PredictionHead {
    dpt: DptHead,
    local_features: Mlp,
}

impl PredictionHead {
    fn new(vb: VarBuilder) -> Result<Self> {
        let in_dim = ENCODER_DIM + DECODER_DIM;
        let out_dim = LOCAL_FEATURE_CHANNELS * PATCH_SIZE * PATCH_SIZE;
        Ok(Self {
            dpt: DptHead::new(vb.pp("dpt"))?,
            local_features: Mlp::new(
                in_dim,
                in_dim * LOCAL_FEATURES_MLP_RATIO,
                out_dim,
                vb.pp("head_local_features"),
            )?,
        })
    }

    /// Single-image batch only.
    fn forward(
        &self,
        view: &DecodedView,
        patch_h: usize,
        patch_w: usize,
    ) -> Result<ViewPrediction> {
        let (height, width) = (patch_h * PATCH_SIZE, patch_w * PATCH_SIZE);

        // [1, 4, H, W] -> [H, W, 4]
        let dpt = self
            .dpt
            .forward(&view.dpt_features, patch_h, patch_w)?
            .squeeze(0)?
            .permute((1, 2, 0))?
            .to_dtype(DType::F32)?;
        let xyz = dpt.narrow(D::Minus1, 0, 3)?;
        let distance = xyz.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let pts3d = xyz
            .broadcast_div(&distance.clamp(PTS3D_MIN_NORM, f64::INFINITY)?)?
            .broadcast_mul(&(distance.exp()? - 1.0)?)?;
        let conf = (dpt.narrow(D::Minus1, 3, 1)?.exp()? + CONF_MIN)?;

        // Pixel shuffle: [1, tokens, C * 16 * 16] -> [H, W, C]
        let local = self
            .local_features
            .forward(&view.local_features)?
            .reshape((
                patch_h,
                patch_w,
                LOCAL_FEATURE_CHANNELS,
                PATCH_SIZE,
                PATCH_SIZE,
            ))?
            .permute((0, 3, 1, 4, 2))?
            .reshape((height, width, LOCAL_FEATURE_CHANNELS))?
            .to_dtype(DType::F32)?;
        let desc = local.narrow(D::Minus1, 0, DESCRIPTOR_DIM)?;
        let desc = desc.broadcast_div(&desc.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?)?;
        let desc_conf = local.narrow(D::Minus1, DESCRIPTOR_DIM, 1)?.exp()?;

        Ok(ViewPrediction {
            pts3d,
            conf,
            desc,
            desc_conf,
        })
    }
}

pub struct Mast3r {
    encoder: Encoder,
    decoder: Decoder,
    head1: PredictionHead,
    head2: PredictionHead,
}

impl Mast3r {
    /// `vb` over the original MASt3R checkpoint keys (enc_blocks.*, dec_blocks.*, downstream_head1.*, ...).
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            encoder: Encoder::new(vb.clone())?,
            decoder: Decoder::new(vb.clone())?,
            head1: PredictionHead::new(vb.pp("downstream_head1"))?,
            head2: PredictionHead::new(vb.pp("downstream_head2"))?,
        })
    }

    /// `image`: [1, 3, H, W] in [-1, 1], H and W multiples of PATCH_SIZE.
    pub fn encode(&self, image: &Tensor) -> Result<EncodedImage> {
        let (batch, _, height, width) = image.dims4()?;
        assert_eq!(batch, 1, "encode takes one image");
        assert!(
            height % PATCH_SIZE == 0 && width % PATCH_SIZE == 0,
            "image {height}x{width} is not a multiple of the {PATCH_SIZE}px patch"
        );
        let (patch_h, patch_w) = (height / PATCH_SIZE, width / PATCH_SIZE);
        let rope = RopeGrid::new(patch_h, patch_w, HEAD_DIM, image.dtype(), image.device())?;
        Ok(EncodedImage {
            tokens: self.encoder.forward(image, &rope)?,
            rope,
            patch_h,
            patch_w,
        })
    }

    /// Both predictions are in the camera frame of `view1`. The two views may have different sizes.
    pub fn decode_pair(
        &self,
        view1: &EncodedImage,
        view2: &EncodedImage,
    ) -> Result<(ViewPrediction, ViewPrediction)> {
        let (decoded1, decoded2) =
            self.decoder
                .forward(&view1.tokens, &view2.tokens, &view1.rope, &view2.rope)?;
        Ok((
            self.head1
                .forward(&decoded1, view1.patch_h, view1.patch_w)?,
            self.head2
                .forward(&decoded2, view2.patch_h, view2.patch_w)?,
        ))
    }
}

/// Encoder output for one image, reusable across every pair it is part of.
pub struct EncodedImage {
    tokens: Tensor,
    rope: RopeGrid,
    patch_h: usize,
    patch_w: usize,
}
