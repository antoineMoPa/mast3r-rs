use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};

use mast3r_rs::alignment::{PairPrediction, PointMap, align_scene};
use mast3r_rs::image_input::load_image;
use mast3r_rs::matching::count_mutual_matches;
use mast3r_rs::model::Mast3r;
use mast3r_rs::ply::{confident_points, write_ply};

/// MASt3R ViT-Large weights, fp16 cast of naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.
const WEIGHTS_REPO: &str = "Aedelon/mast3r-vit-large-fp16";
const WEIGHTS_FILE: &str = "unified.safetensors";

#[derive(Clone, Copy, ValueEnum)]
enum Precision {
    F32,
    F16,
    Bf16,
}

impl From<Precision> for DType {
    fn from(precision: Precision) -> Self {
        match precision {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::Bf16 => DType::BF16,
        }
    }
}

/// Reconstruct a colored 3D point cloud from two or more images of the same scene.
///
/// Every pair of images goes through the decoder (n * (n - 1) / 2 pairs), then the
/// pairwise point maps are chained into one frame along a spanning tree that keeps the
/// pairs with the most descriptor matches.
#[derive(Parser)]
struct Args {
    /// Two or more JPEG or PNG images. Transparent pixels of PNGs are background: cut the
    /// background out for turntable captures (fixed camera, rotating object).
    #[arg(required = true, num_args = 2..)]
    images: Vec<PathBuf>,
    /// Output point cloud (y up), in the frame of a camera of the pair with the most matches.
    #[arg(long, default_value = "reconstruction.ply")]
    out: PathBuf,
    /// Keep points with confidence >= this (confidence is 1 + exp(raw), so always > 1).
    #[arg(long, default_value_t = 1.5)]
    min_conf: f32,
    #[arg(long, value_enum, default_value = "f16")]
    precision: Precision,
    /// MASt3R safetensors checkpoint; downloaded from Hugging Face when omitted.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Also save the aligned dense point maps (image{i}.pts3d [H, W, 3], image{i}.conf [H, W],
    /// image{i}.foreground [H, W] u8, OpenCV camera convention) to this safetensors file.
    #[arg(long)]
    raw_out: Option<PathBuf>,
    /// Run on the CPU instead of Metal.
    #[arg(long)]
    cpu: bool,
}

fn point_map_tensors(index: usize, point_map: &PointMap) -> Result<[(String, Tensor); 3]> {
    let (height, width) = (point_map.height, point_map.width);
    let points: Vec<f32> = point_map.points.iter().flatten().copied().collect();
    let foreground: Vec<u8> = point_map.foreground.iter().map(|&f| f as u8).collect();
    Ok([
        (
            format!("image{index}.pts3d"),
            Tensor::from_vec(points, (height, width, 3), &Device::Cpu)?,
        ),
        (
            format!("image{index}.conf"),
            Tensor::from_slice(&point_map.conf, (height, width), &Device::Cpu)?,
        ),
        (
            format!("image{index}.foreground"),
            Tensor::from_vec(foreground, (height, width), &Device::Cpu)?,
        ),
    ])
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };
    let dtype = DType::from(args.precision);

    let weights = match args.weights {
        Some(path) => path,
        None => hf_hub::api::sync::Api::new()?
            .model(WEIGHTS_REPO.to_string())
            .get(WEIGHTS_FILE)?,
    };
    let started = Instant::now();
    // SAFETY: the checkpoint file is not modified while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], dtype, &device)? };
    let model = Mast3r::new(vb)?;
    println!("loaded {} in {:.2?}", weights.display(), started.elapsed());

    let started = Instant::now();
    let images = args
        .images
        .iter()
        .map(|path| load_image(path, dtype, &device))
        .collect::<Result<Vec<_>>>()?;
    let encoded = images
        .iter()
        .map(|image| model.encode(&image.tensor))
        .collect::<candle_core::Result<Vec<_>>>()?;
    device.synchronize()?;
    println!(
        "encoded {} images in {:.2?}",
        images.len(),
        started.elapsed()
    );

    let started = Instant::now();
    let mut pairs = Vec::new();
    for first in 0..images.len() {
        for second in first + 1..images.len() {
            let (first_view, second_view) = model.decode_pair(&encoded[first], &encoded[second])?;
            let (first_foreground, second_foreground) =
                (&images[first].foreground, &images[second].foreground);
            pairs.push(PairPrediction {
                first,
                second,
                matches: count_mutual_matches(
                    &first_view,
                    first_foreground,
                    &second_view,
                    second_foreground,
                )?,
                first_view: PointMap::from_prediction(&first_view, first_foreground)?,
                second_view: PointMap::from_prediction(&second_view, second_foreground)?,
            });
        }
    }
    println!("decoded {} pairs in {:.2?}", pairs.len(), started.elapsed());

    let started = Instant::now();
    let alignment = align_scene(images.len(), &pairs);
    println!(
        "aligned in {:.2?}, residual {:.4} after spanning tree, {:.4} after refinement",
        started.elapsed(),
        alignment.tree_residual,
        alignment.refined_residual
    );
    let aligned = alignment.point_maps;

    if let Some(raw_out) = &args.raw_out {
        let tensors: HashMap<String, Tensor> = aligned
            .iter()
            .enumerate()
            .map(|(index, point_map)| point_map_tensors(index, point_map))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        candle_core::safetensors::save(&tensors, raw_out)?;
        println!("wrote {}", raw_out.display());
    }

    let mut points = Vec::new();
    for (point_map, image) in aligned.iter().zip(&images) {
        points.extend(confident_points(point_map, &image.rgb, args.min_conf)?);
    }
    write_ply(&args.out, &points)?;
    println!("wrote {} points to {}", points.len(), args.out.display());
    Ok(())
}
