//! Pair overlap from MASt3R's local descriptors, counted as mutual nearest neighbours.
//!
//! Confidence alone does not tell overlapping pairs from non-overlapping ones: the model
//! is confident about a mannequin's front and its back seen together. Descriptor matches
//! only exist where both images see the same surface, and a match only counts when the
//! pair puts both of its pixels at the same 3D point (geometric verification): descriptor
//! confusions, such as a mannequin's front matched to its back, fail that test.

use anyhow::{Result, ensure};
use candle_core::{D, Tensor};

use crate::model::ViewPrediction;

/// Foreground pixels are subsampled to at most this many per view, which bounds the
/// similarity matrix at MAX_SAMPLES^2 f32 (64 MB).
const MAX_SAMPLES: usize = 4096;
/// A match is kept when its two 3D points are closer than this fraction of the median
/// foreground distance from the first view's camera.
const MAX_MATCH_DISTANCE: f32 = 0.02;

/// [samples, DESCRIPTOR_DIM] descriptors at evenly spaced foreground pixels, and those pixels.
fn sample_descriptors(view: &ViewPrediction, foreground: &[bool]) -> Result<(Tensor, Vec<u32>)> {
    let (height, width, channels) = view.desc.dims3()?;
    ensure!(
        foreground.len() == height * width,
        "foreground has {} pixels, descriptors are {width}x{height}",
        foreground.len()
    );
    let pixels: Vec<u32> = foreground
        .iter()
        .enumerate()
        .filter(|(_, is_foreground)| **is_foreground)
        .map(|(pixel, _)| pixel as u32)
        .collect();
    let step = pixels.len().div_ceil(MAX_SAMPLES);
    let sampled: Vec<u32> = pixels.into_iter().step_by(step).collect();
    let index = Tensor::new(sampled.as_slice(), view.desc.device())?;
    let descriptors = view
        .desc
        .reshape((height * width, channels))?
        .index_select(&index, 0)?;
    Ok((descriptors, sampled))
}

fn points(view: &ViewPrediction) -> Result<Vec<[f32; 3]>> {
    Ok(view
        .pts3d
        .flatten_all()?
        .to_vec1::<f32>()?
        .chunks_exact(3)
        .map(|xyz| [xyz[0], xyz[1], xyz[2]])
        .collect())
}

/// Number of sampled foreground pixels of `view1` and `view2` whose descriptors are each
/// other's nearest neighbour and whose 3D points agree. Descriptors are unit length, so
/// similarity is a dot product.
pub fn count_mutual_matches(
    view1: &ViewPrediction,
    foreground1: &[bool],
    view2: &ViewPrediction,
    foreground2: &[bool],
) -> Result<usize> {
    let (descriptors1, pixels1) = sample_descriptors(view1, foreground1)?;
    let (descriptors2, pixels2) = sample_descriptors(view2, foreground2)?;
    let similarity = descriptors1.matmul(&descriptors2.t()?)?;
    let nearest_in_2: Vec<u32> = similarity.argmax(D::Minus1)?.to_vec1()?;
    let nearest_in_1: Vec<u32> = similarity.argmax(0)?.to_vec1()?;
    let (points1, points2) = (points(view1)?, points(view2)?);
    let mut camera_distances: Vec<f32> = pixels1
        .iter()
        .map(|&pixel| norm(points1[pixel as usize]))
        .collect();
    camera_distances.sort_by(f32::total_cmp);
    let max_distance = MAX_MATCH_DISTANCE * camera_distances[camera_distances.len() / 2];
    Ok(nearest_in_2
        .iter()
        .enumerate()
        .filter(|(sample1, sample2)| nearest_in_1[**sample2 as usize] as usize == *sample1)
        .filter(|(sample1, sample2)| {
            let point1 = points1[pixels1[*sample1] as usize];
            let point2 = points2[pixels2[**sample2 as usize] as usize];
            norm([0, 1, 2].map(|axis| point1[axis] - point2[axis])) < max_distance
        })
        .count())
}

fn norm(vector: [f32; 3]) -> f32 {
    vector.iter().map(|c| c * c).sum::<f32>().sqrt()
}
