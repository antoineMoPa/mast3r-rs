//! Multi-view alignment. A spanning tree, as in the `init_mst` step of DUSt3R's global
//! alignment, gives the initial placement, which `refinement` then adjusts jointly.
//!
//! Every pair prediction holds both views' points in the camera frame of its first view, at
//! a scale of its own. A maximum spanning tree over the pairs, weighted by descriptor
//! matches, is grown from the pair with the most matches, whose first camera becomes the
//! world frame. Each
//! newly attached pair shares one already-placed image with the scene. The pair's points
//! for that image are matched pixel-for-pixel to the placed points, which gives a
//! similarity transform (scale, rotation, translation) for the pair's other image.
//! Background pixels do not count in the fits.

use anyhow::{Result, ensure};
use candle_core::D;
use nalgebra::{Matrix3, Vector3};

use crate::model::ViewPrediction;
use crate::refinement;

/// One view's dense points, row-major, on the CPU.
#[derive(Clone)]
pub struct PointMap {
    pub height: usize,
    pub width: usize,
    pub points: Vec<[f32; 3]>,
    pub conf: Vec<f32>,
    /// Pixels of the object being reconstructed, from the input image's alpha.
    pub foreground: Vec<bool>,
}

impl PointMap {
    pub fn from_prediction(prediction: &ViewPrediction, foreground: &[bool]) -> Result<Self> {
        let (height, width, _) = prediction.pts3d.dims3()?;
        ensure!(
            foreground.len() == height * width,
            "foreground has {} pixels, prediction is {width}x{height}",
            foreground.len()
        );
        let points = prediction
            .pts3d
            .flatten_all()?
            .to_vec1::<f32>()?
            .chunks_exact(3)
            .map(|xyz| [xyz[0], xyz[1], xyz[2]])
            .collect();
        let conf = prediction
            .conf
            .squeeze(D::Minus1)?
            .flatten_all()?
            .to_vec1()?;
        Ok(Self {
            height,
            width,
            points,
            conf,
            foreground: foreground.to_vec(),
        })
    }

    /// Confidence on the foreground, zero on the background.
    pub(crate) fn foreground_conf(&self) -> Vec<f32> {
        self.conf
            .iter()
            .zip(&self.foreground)
            .map(|(&conf, &foreground)| if foreground { conf } else { 0.0 })
            .collect()
    }
}

/// Decoder output for images `first` and `second`, both in the camera frame of `first`.
pub struct PairPrediction {
    pub first: usize,
    pub second: usize,
    pub first_view: PointMap,
    pub second_view: PointMap,
    /// Mutual descriptor matches between the two views, a measure of their overlap.
    pub matches: usize,
}

pub(crate) struct Similarity {
    scale: f64,
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
}

impl Similarity {
    pub(crate) fn apply(&self, point: &[f32; 3]) -> [f32; 3] {
        let p = Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64);
        let q = self.scale * self.rotation * p + self.translation;
        [q.x as f32, q.y as f32, q.z as f32]
    }
}

/// Weighted Umeyama: the similarity minimizing sum(w * |s R source + t - target|^2).
pub(crate) fn fit_similarity(
    source: &[[f32; 3]],
    target: &[[f32; 3]],
    weights: &[f32],
) -> Similarity {
    assert_eq!(
        source.len(),
        target.len(),
        "source and target must be the same pixels"
    );
    let to_vector = |p: &[f32; 3]| Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64);
    let weight_sum: f64 = weights.iter().map(|&w| w as f64).sum();
    let weighted_mean = |points: &[[f32; 3]]| -> Vector3<f64> {
        points
            .iter()
            .zip(weights)
            .map(|(p, &w)| to_vector(p) * w as f64)
            .sum::<Vector3<f64>>()
            / weight_sum
    };
    let source_mean = weighted_mean(source);
    let target_mean = weighted_mean(target);

    let mut covariance = Matrix3::zeros();
    let mut source_variance = 0.0;
    for ((s, t), &w) in source.iter().zip(target).zip(weights) {
        let s = to_vector(s) - source_mean;
        let t = to_vector(t) - target_mean;
        covariance += (t * s.transpose()) * w as f64;
        source_variance += s.norm_squared() * w as f64;
    }
    covariance /= weight_sum;
    source_variance /= weight_sum;

    let svd = covariance.svd(true, true);
    let (u, v_t) = (svd.u.unwrap(), svd.v_t.unwrap());
    // Flip the smallest axis if needed so the result is a rotation, not a reflection.
    let reflection_sign = (u * v_t).determinant().signum();
    let correction = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, reflection_sign));
    let rotation = u * correction * v_t;
    let scale = (svd.singular_values.component_mul(&correction.diagonal())).sum() / source_variance;
    Similarity {
        scale,
        rotation,
        translation: target_mean - scale * rotation * source_mean,
    }
}

pub struct SceneAlignment {
    /// One per image, in the camera frame of the first image of the pair with the most
    /// matches.
    pub point_maps: Vec<PointMap>,
    /// Confidence-weighted RMS distance between where pairs put a point and where the
    /// scene has it, relative to the scene's RMS radius, after the spanning tree.
    pub tree_residual: f64,
    /// The same after refinement.
    pub refined_residual: f64,
}

/// Each image's this many pairs with the most matches are its neighbor pairs, the ones
/// trusted for alignment. Pairs that share no surface (a mannequin's front and back)
/// still get confident, made-up relative poses: one of them in the spanning tree folds a
/// 360 degree capture in half.
const MATCH_NEIGHBORS: usize = 2;

/// One flag per pair: whether it is among the `MATCH_NEIGHBORS` pairs with the most
/// matches of either of its images.
fn neighbor_pairs(image_count: usize, pairs: &[PairPrediction]) -> Vec<bool> {
    let mut is_neighbor = vec![false; pairs.len()];
    for image in 0..image_count {
        let mut own: Vec<usize> = (0..pairs.len())
            .filter(|&index| pairs[index].first == image || pairs[index].second == image)
            .collect();
        own.sort_by_key(|&index| std::cmp::Reverse(pairs[index].matches));
        for index in own.into_iter().take(MATCH_NEIGHBORS) {
            is_neighbor[index] = true;
        }
    }
    is_neighbor
}

/// Places every image in one frame. `pairs` must connect all images.
pub fn align_scene(image_count: usize, pairs: &[PairPrediction]) -> SceneAlignment {
    let is_neighbor = neighbor_pairs(image_count, pairs);
    let (tree_point_maps, tree_pairs) = spanning_tree(image_count, pairs, &is_neighbor);
    let selected: Vec<usize> = (0..pairs.len())
        .filter(|&index| is_neighbor[index] || tree_pairs.contains(&index))
        .collect();
    let root = pairs[tree_pairs[0]].first;
    let refinement = refinement::refine(tree_point_maps, pairs, &selected, root);
    SceneAlignment {
        point_maps: refinement.point_maps,
        tree_residual: refinement.initial_residual,
        refined_residual: refinement.final_residual,
    }
}

/// Places images one pair at a time: neighbor pairs first, then by most matches. Returns
/// the point maps and the indices of the pairs used, root pair first.
fn spanning_tree(
    image_count: usize,
    pairs: &[PairPrediction],
    is_neighbor: &[bool],
) -> (Vec<PointMap>, Vec<usize>) {
    let preference = |(index, pair): &(usize, &PairPrediction)| (is_neighbor[*index], pair.matches);
    let (root_index, root) = pairs
        .iter()
        .enumerate()
        .max_by_key(preference)
        .expect("at least one pair");
    let mut tree_pairs = vec![root_index];
    let mut world: Vec<Option<PointMap>> = vec![None; image_count];
    world[root.first] = Some(root.first_view.clone());
    world[root.second] = Some(root.second_view.clone());

    while world.iter().any(Option::is_none) {
        let (edge_index, edge) = pairs
            .iter()
            .enumerate()
            .filter(|(_, pair)| world[pair.first].is_some() != world[pair.second].is_some())
            .max_by_key(preference)
            .expect("pairs do not connect every image");
        tree_pairs.push(edge_index);
        let (placed, placed_view, new, new_view) = if world[edge.first].is_some() {
            (edge.first, &edge.first_view, edge.second, &edge.second_view)
        } else {
            (edge.second, &edge.second_view, edge.first, &edge.first_view)
        };
        let placed_world = world[placed].as_ref().unwrap();
        let transform = fit_similarity(
            &placed_view.points,
            &placed_world.points,
            &placed_view.foreground_conf(),
        );
        world[new] = Some(PointMap {
            points: new_view.points.iter().map(|p| transform.apply(p)).collect(),
            ..new_view.clone()
        });
    }
    (world.into_iter().map(Option::unwrap).collect(), tree_pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_similarity() {
        let source: Vec<[f32; 3]> = (0..50)
            .map(|i| {
                let i = i as f32;
                [i.sin(), (i * 0.7).cos(), i * 0.1]
            })
            .collect();
        let expected = Similarity {
            scale: 2.5,
            rotation: *nalgebra::Rotation3::from_euler_angles(0.3, -0.5, 1.1).matrix(),
            translation: Vector3::new(1.0, -2.0, 0.5),
        };
        let target: Vec<[f32; 3]> = source.iter().map(|p| expected.apply(p)).collect();
        let fitted = fit_similarity(&source, &target, &vec![1.0; source.len()]);
        assert!((fitted.scale - expected.scale).abs() < 1e-4);
        assert!((fitted.rotation - expected.rotation).norm() < 1e-4);
        assert!((fitted.translation - expected.translation).norm() < 1e-4);
    }
}
