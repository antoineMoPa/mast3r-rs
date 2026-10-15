//! Joint refinement of the spanning-tree placement by alternating least squares
//! (generalized Procrustes analysis), a closed-form stand-in for DUSt3R's global alignment.
//!
//! Each refinement pair gets a similarity transform from its own frame into the world
//! frame, fit on both of its views at once. Each image's world points then become the
//! confidence-weighted average of where its pairs put them. Both steps lower the same
//! weighted squared error. After each round the scene is moved back onto the root image's
//! spanning-tree points: this keeps the world frame where the tree put it, and stops the
//! scene from shrinking to a point, which would have zero error.

use crate::alignment::{PairPrediction, PointMap, Similarity, fit_similarity};

const ITERATIONS: usize = 50;

pub struct Refinement {
    pub point_maps: Vec<PointMap>,
    /// See `SceneAlignment::tree_residual`.
    pub initial_residual: f64,
    pub final_residual: f64,
}

/// (image index, view) for both views of `pair`.
fn views(pair: &PairPrediction) -> [(usize, &PointMap); 2] {
    [
        (pair.first, &pair.first_view),
        (pair.second, &pair.second_view),
    ]
}

fn foreground_pixels(view: &PointMap) -> impl Iterator<Item = usize> + '_ {
    (0..view.points.len()).filter(|&pixel| view.foreground[pixel])
}

/// The similarity taking `pair`'s frame onto the current world points of both its images.
fn fit_pair(pair: &PairPrediction, world: &[PointMap]) -> Similarity {
    let (mut source, mut target, mut weights) = (Vec::new(), Vec::new(), Vec::new());
    for (image, view) in views(pair) {
        for pixel in foreground_pixels(view) {
            source.push(view.points[pixel]);
            target.push(world[image].points[pixel]);
            weights.push(view.conf[pixel]);
        }
    }
    fit_similarity(&source, &target, &weights)
}

/// World points as the confidence-weighted average of where each pair puts them, and
/// confidence as the mean over those pairs. Background pixels are left as they are.
fn average_pair_predictions(
    world: &[PointMap],
    pairs: &[PairPrediction],
    selected: &[usize],
    transforms: &[Similarity],
) -> Vec<PointMap> {
    let mut sums: Vec<Vec<[f64; 4]>> = world
        .iter()
        .map(|point_map| vec![[0.0; 4]; point_map.points.len()])
        .collect();
    let mut pair_counts = vec![0usize; world.len()];
    for (&index, transform) in selected.iter().zip(transforms) {
        for (image, view) in views(&pairs[index]) {
            pair_counts[image] += 1;
            for pixel in foreground_pixels(view) {
                let point = transform.apply(&view.points[pixel]);
                let weight = view.conf[pixel] as f64;
                let sum = &mut sums[image][pixel];
                for axis in 0..3 {
                    sum[axis] += weight * point[axis] as f64;
                }
                sum[3] += weight;
            }
        }
    }
    world
        .iter()
        .zip(sums)
        .zip(pair_counts)
        .map(|((point_map, image_sums), pair_count)| {
            let mut averaged = point_map.clone();
            for (pixel, [x, y, z, weight]) in image_sums.into_iter().enumerate() {
                if point_map.foreground[pixel] {
                    averaged.points[pixel] = [
                        (x / weight) as f32,
                        (y / weight) as f32,
                        (z / weight) as f32,
                    ];
                    averaged.conf[pixel] = (weight / pair_count as f64) as f32;
                }
            }
            averaged
        })
        .collect()
}

/// Weighted RMS of |transform(pair point) - world point| over the selected pairs'
/// foreground, divided by the RMS distance of the world's foreground points to their mean.
fn relative_residual(
    world: &[PointMap],
    pairs: &[PairPrediction],
    selected: &[usize],
    transforms: &[Similarity],
) -> f64 {
    let distance_squared = |a: &[f32; 3], b: &[f32; 3]| -> f64 {
        (0..3)
            .map(|axis| (a[axis] as f64 - b[axis] as f64).powi(2))
            .sum()
    };
    let (mut error, mut weight_sum) = (0.0, 0.0);
    for (&index, transform) in selected.iter().zip(transforms) {
        for (image, view) in views(&pairs[index]) {
            for pixel in foreground_pixels(view) {
                let weight = view.conf[pixel] as f64;
                let predicted = transform.apply(&view.points[pixel]);
                error += weight * distance_squared(&predicted, &world[image].points[pixel]);
                weight_sum += weight;
            }
        }
    }
    let foreground: Vec<&[f32; 3]> = world
        .iter()
        .flat_map(|point_map| foreground_pixels(point_map).map(|pixel| &point_map.points[pixel]))
        .collect();
    let mean = (0..3)
        .map(|axis| foreground.iter().map(|p| p[axis]).sum::<f32>() / foreground.len() as f32)
        .collect::<Vec<_>>();
    let mean = [mean[0], mean[1], mean[2]];
    let spread = foreground
        .iter()
        .map(|p| distance_squared(p, &mean))
        .sum::<f64>()
        / foreground.len() as f64;
    (error / weight_sum).sqrt() / spread.sqrt()
}

/// Refines on the pairs at indices `selected`, which must connect all images. The world
/// frame stays on image `root`'s points from `tree_point_maps`.
pub fn refine(
    tree_point_maps: Vec<PointMap>,
    pairs: &[PairPrediction],
    selected: &[usize],
    root: usize,
) -> Refinement {
    let anchor = tree_point_maps[root].clone();
    let anchor_weights = anchor.foreground_conf();
    let fit_all = |world: &[PointMap]| -> Vec<Similarity> {
        selected
            .iter()
            .map(|&index| fit_pair(&pairs[index], world))
            .collect()
    };

    let mut world = tree_point_maps;
    let initial_residual = relative_residual(&world, pairs, selected, &fit_all(&world));
    for _ in 0..ITERATIONS {
        world = average_pair_predictions(&world, pairs, selected, &fit_all(&world));
        let back_to_anchor = fit_similarity(&world[root].points, &anchor.points, &anchor_weights);
        for point_map in &mut world {
            for point in &mut point_map.points {
                *point = back_to_anchor.apply(point);
            }
        }
    }
    let final_residual = relative_residual(&world, pairs, selected, &fit_all(&world));
    Refinement {
        point_maps: world,
        initial_residual,
        final_residual,
    }
}
