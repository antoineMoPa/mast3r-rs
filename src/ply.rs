//! Colored point cloud export as binary little-endian PLY.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Result, ensure};
use image::RgbImage;

use crate::alignment::PointMap;

pub struct ColoredPoint {
    pub xyz: [f32; 3],
    pub rgb: [u8; 3],
}

/// Foreground points of `point_map` whose confidence is at least `min_conf`, colored by `image`.
///
/// MASt3R predicts in the OpenCV camera convention (x right, y down, z forward); viewers
/// expect y up, so points are turned 180 degrees around x (y and z negated).
pub fn confident_points(
    point_map: &PointMap,
    image: &RgbImage,
    min_conf: f32,
) -> Result<Vec<ColoredPoint>> {
    ensure!(
        (point_map.width as u32, point_map.height as u32) == image.dimensions(),
        "point map {}x{} does not match image {:?}",
        point_map.width,
        point_map.height,
        image.dimensions()
    );
    Ok(point_map
        .conf
        .iter()
        .zip(&point_map.points)
        .zip(image.pixels())
        .zip(&point_map.foreground)
        .filter(|(((conf, _), _), foreground)| **foreground && **conf >= min_conf)
        .map(|(((_, [x, y, z]), pixel), _)| ColoredPoint {
            xyz: [*x, -y, -z],
            rgb: pixel.0,
        })
        .collect())
}

pub fn write_ply(path: &Path, points: &[ColoredPoint]) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    write!(
        out,
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n\
         property float x\nproperty float y\nproperty float z\n\
         property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n",
        points.len()
    )?;
    for point in points {
        for coordinate in point.xyz {
            out.write_all(&coordinate.to_le_bytes())?;
        }
        out.write_all(&point.rgb)?;
    }
    out.flush()?;
    Ok(())
}
