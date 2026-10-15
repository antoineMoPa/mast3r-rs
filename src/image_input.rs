//! Image loading with the DUSt3R / MASt3R preprocessing: resize the long edge to
//! 512, center-crop to a multiple of the patch size, square images cropped to 4:3.
//!
//! Transparent pixels (alpha < 50%) are background: the model sees them as a flat
//! `BACKDROP_RGB` backdrop, and they are left out of alignment and of the point cloud.
//! Images with transparency are first cropped to their foreground, so the object fills
//! the model's input. Cutting the background out is what makes turntable captures (fixed
//! camera, rotating object) work, since a static background otherwise tells the model the
//! camera never moved.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use image::imageops::FilterType;
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage, Rgba, RgbaImage};

use crate::encoder::PATCH_SIZE;

/// MASt3R_ViTLarge_BaseDecoder_512 is trained with a 512px long edge.
pub const LONG_EDGE: u32 = 512;
/// What transparent pixels are replaced with before the model sees them. Mid-gray is 0 in
/// the model's [-1, 1] input; on a turntable capture it gives foreground confidence of
/// 2.5 to 4 where a white backdrop gives 1.0, the minimum.
const BACKDROP_RGB: [u8; 3] = [128, 128, 128];
/// Pixels with at least this alpha are foreground.
const FOREGROUND_MIN_ALPHA: u8 = 128;
/// Margin kept around the foreground bounding box, as a fraction of its size, per side.
const FOREGROUND_MARGIN: f64 = 0.05;

pub struct InputImage {
    /// The cropped image the model sees, used to color the point cloud.
    pub rgb: RgbImage,
    /// Row-major, one entry per pixel of `rgb`. All true for images without alpha.
    pub foreground: Vec<bool>,
    /// [1, 3, H, W] in [-1, 1].
    pub tensor: Tensor,
}

fn read_upright(path: &Path) -> Result<DynamicImage> {
    let mut decoder = ImageReader::open(path)?
        .with_guessed_format()?
        .into_decoder()
        .with_context(|| format!("decoding {}", path.display()))?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image)
}

/// Crops to the foreground bounding box plus margin, widened to the image's own aspect
/// ratio. Parts of the crop outside the image are transparent. Fully opaque images are
/// returned whole.
fn crop_to_foreground(rgba: RgbaImage) -> Result<RgbaImage> {
    if rgba.pixels().all(|pixel| pixel[3] == u8::MAX) {
        return Ok(rgba);
    }
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (u32::MAX, u32::MAX, 0, 0);
    for (x, y, pixel) in rgba.enumerate_pixels() {
        if pixel[3] >= FOREGROUND_MIN_ALPHA {
            (min_x, min_y) = (min_x.min(x), min_y.min(y));
            (max_x, max_y) = (max_x.max(x), max_y.max(y));
        }
    }
    ensure!(min_x <= max_x, "image has no foreground pixel");
    let box_w = (max_x - min_x + 1) as f64 * (1.0 + 2.0 * FOREGROUND_MARGIN);
    let box_h = (max_y - min_y + 1) as f64 * (1.0 + 2.0 * FOREGROUND_MARGIN);
    let aspect = rgba.width() as f64 / rgba.height() as f64;
    let (crop_w, crop_h) = if box_w / box_h > aspect {
        (box_w, box_w / aspect)
    } else {
        (box_h * aspect, box_h)
    };
    let left = ((min_x + max_x + 1) as f64 - crop_w) / 2.0;
    let top = ((min_y + max_y + 1) as f64 - crop_h) / 2.0;
    let mut cropped = RgbaImage::from_pixel(
        crop_w.round() as u32,
        crop_h.round() as u32,
        Rgba([0, 0, 0, 0]),
    );
    image::imageops::replace(
        &mut cropped,
        &rgba,
        -left.round() as i64,
        -top.round() as i64,
    );
    Ok(cropped)
}

/// Blends color onto the backdrop by alpha, keeping alpha. Done at full resolution so
/// the resize filters never mix in the color hidden under transparent pixels.
fn composite_on_backdrop(mut rgba: RgbaImage) -> DynamicImage {
    for pixel in rgba.pixels_mut() {
        let alpha = pixel[3] as u32;
        for (channel, backdrop) in pixel.0.iter_mut().zip(BACKDROP_RGB) {
            *channel = ((*channel as u32 * alpha + backdrop as u32 * (255 - alpha)) / 255) as u8;
        }
    }
    DynamicImage::ImageRgba8(rgba)
}

fn resize_long_edge(image: &DynamicImage) -> DynamicImage {
    let (width, height) = (image.width(), image.height());
    let long_edge = width.max(height);
    if long_edge == LONG_EDGE {
        return image.clone();
    }
    // Same filters as PIL in the reference: Lanczos to shrink, bicubic to enlarge.
    let filter = if long_edge > LONG_EDGE {
        FilterType::Lanczos3
    } else {
        FilterType::CatmullRom
    };
    let scale = |side: u32| (side as f64 * LONG_EDGE as f64 / long_edge as f64).round() as u32;
    image.resize_exact(scale(width), scale(height), filter)
}

fn crop_to_patch_grid(image: &DynamicImage) -> Result<RgbaImage> {
    let (width, height) = (image.width(), image.height());
    let patch = PATCH_SIZE as u32;
    let (center_x, center_y) = (width / 2, height / 2);
    let crop_w = (2 * center_x) / patch * patch;
    let mut crop_h = (2 * center_y) / patch * patch;
    if width == height {
        crop_h = 3 * crop_w / 4;
    }
    ensure!(
        crop_h.is_multiple_of(patch) && crop_h > 0 && crop_w > 0,
        "cannot crop {width}x{height} to a {patch}px patch grid"
    );
    Ok(image
        .crop_imm(center_x - crop_w / 2, center_y - crop_h / 2, crop_w, crop_h)
        .to_rgba8())
}

pub fn load_image(path: &Path, dtype: DType, device: &Device) -> Result<InputImage> {
    let foreground_crop = crop_to_foreground(read_upright(path)?.to_rgba8())
        .with_context(|| format!("cropping {}", path.display()))?;
    let rgba = crop_to_patch_grid(&resize_long_edge(&composite_on_backdrop(foreground_crop)))?;
    let foreground: Vec<bool> = rgba
        .pixels()
        .map(|pixel| pixel[3] >= FOREGROUND_MIN_ALPHA)
        .collect();
    ensure!(
        foreground.contains(&true),
        "{} is fully transparent after cropping",
        path.display()
    );
    let rgb = DynamicImage::ImageRgba8(rgba).to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let tensor = Tensor::from_vec(rgb.as_raw().clone(), (height, width, 3), device)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2.0 / 255.0, -1.0)?
        .unsqueeze(0)?
        .to_dtype(dtype)?;
    Ok(InputImage {
        rgb,
        foreground,
        tensor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crop_size(width: u32, height: u32) -> (u32, u32) {
        crop_to_patch_grid(&DynamicImage::new_rgb8(width, height))
            .unwrap()
            .dimensions()
    }

    #[test]
    fn crops_to_patch_multiples() {
        assert_eq!(crop_size(512, 384), (512, 384));
        assert_eq!(crop_size(512, 341), (512, 336));
        assert_eq!(crop_size(383, 512), (368, 512));
    }

    #[test]
    fn crops_square_to_four_by_three() {
        assert_eq!(crop_size(512, 512), (512, 384));
    }

    #[test]
    fn transparent_pixels_become_backdrop() {
        let mut rgba = RgbaImage::from_pixel(2, 1, Rgba([10, 20, 30, 255]));
        rgba.put_pixel(1, 0, Rgba([10, 20, 30, 0]));
        let composited = composite_on_backdrop(rgba).to_rgba8();
        assert_eq!(composited.get_pixel(0, 0).0, [10, 20, 30, 255]);
        assert_eq!(composited.get_pixel(1, 0).0, [128, 128, 128, 0]);
    }

    #[test]
    fn crops_to_foreground_at_image_aspect() {
        // 200x100 image with a 20x20 opaque square at (90, 40).
        let mut rgba = RgbaImage::from_pixel(200, 100, Rgba([0, 0, 0, 0]));
        for y in 40..60 {
            for x in 90..110 {
                rgba.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
        let cropped = crop_to_foreground(rgba).unwrap();
        // 20px + 5% margin per side = 22px tall, widened to 2:1.
        assert_eq!(cropped.dimensions(), (44, 22));
        assert_eq!(cropped.get_pixel(22, 11).0, [255, 0, 0, 255]);
        assert_eq!(cropped.get_pixel(1, 11).0[3], 0);
    }

    #[test]
    fn keeps_opaque_images_whole() {
        let rgba = RgbaImage::from_pixel(30, 20, Rgba([1, 2, 3, 255]));
        assert_eq!(crop_to_foreground(rgba).unwrap().dimensions(), (30, 20));
    }
}
