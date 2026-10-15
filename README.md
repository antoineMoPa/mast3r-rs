# mast3r-rs

3D reconstruction from photos with [MASt3R](https://github.com/naver/mast3r), in Rust.

This is a tiny port of [Aedelon/mlx-mast3r](https://github.com/Aedelon/mlx-mast3r) to
[candle](https://github.com/huggingface/candle). It takes two or more photos of a scene and
writes a colored point cloud (`.ply`). It runs on Apple Silicon GPUs (Metal) or on the CPU.

## Usage

Requires a recent stable Rust toolchain. Building on macOS needs Metal, which comes with Xcode
or the Command Line Tools.

```sh
cargo run --release -- photo1.jpg photo2.jpg photo3.jpg --out scene.ply
```

On first run, the model weights (1.4 GB, fp16) download from Hugging Face into
`~/.cache/huggingface`.

| option | default | |
|---|---|---|
| `--out` | `reconstruction.ply` | binary PLY, xyz + rgb, y up |
| `--min-conf` | `1.5` | drop points below this confidence (confidence is always > 1) |
| `--precision` | `f16` | `f32`, `f16` or `bf16` |
| `--weights` | Hugging Face download | local MASt3R `.safetensors` (original PyTorch key names) |
| `--raw-out` | none | also save dense per-image `pts3d` and `conf` as `.safetensors` |
| `--cpu` | off | run on the CPU instead of Metal |

### Objects on a turntable

For a capture where the camera stays put and the object turns, cut the background out
first. Otherwise the static background tells the model the camera never moved. Transparent
pixels of a PNG are background: the image is cropped to the object, the model sees a gray
backdrop there, and those pixels are left out of alignment and of the point cloud. On macOS
14+, `tools/cutout.swift` does this with Apple's Vision foreground segmentation:

```sh
swiftc -O tools/cutout.swift -o cutout
for f in photos/*.jpg; do ./cutout "$f" "cutouts/$(basename "${f%.jpg}").png"; done
cargo run --release -- cutouts/*.png --out object.ply
```

To view the result, open the `.ply` in [MeshLab](https://www.meshlab.net),
[CloudCompare](https://www.cloudcompare.org) or Blender.

## How it works

1. Each image is resized to a 512px long edge, center-cropped to multiples of 16, and encoded
   once by the ViT-Large encoder.
2. Each image pair goes through the decoder, where each view cross-attends to the other, and
   through the DPT head. The pair's two dense point maps share the camera frame of its first
   image.
3. Each pair is scored by its geometrically verified descriptor matches: mutual nearest
   neighbours whose two 3D points agree. Each image's two best pairs are its neighbors.
4. A spanning tree, built from neighbor pairs first, places the images one at a time with a
   weighted similarity transform (Umeyama) fit on the image the new pair shares with the
   scene. This is the `init_mst` step of DUSt3R's global alignment.
5. All neighbor pairs are then refined together by alternating least squares (generalized
   Procrustes): each pair's transform is refit, and each image's points become the
   confidence-weighted average of where its pairs put them.

The number of pairs grows as n(n-1)/2: 10 photos make 45 pairs. On an M5 a pair takes about
2 s at f16.

| file | |
|---|---|
| `src/image_input.rs` | loading, EXIF orientation, resize and crop |
| `src/encoder.rs`, `src/rope.rs`, `src/transformer.rs` | ViT-L/16 encoder with 2D RoPE |
| `src/decoder.rs` | 2 × 12 decoder blocks |
| `src/dpt.rs`, `src/model.rs` | DPT head, local descriptors, output activations |
| `src/matching.rs` | pair overlap from verified descriptor matches |
| `src/alignment.rs` | pair selection, spanning-tree placement |
| `src/refinement.rs` | joint refinement of all neighbor pairs |
| `src/ply.rs` | point cloud export |

## Accuracy

On the upstream `NLE_tower` sample pair, the dense outputs (`pts3d`, `conf`, `desc`,
`desc_conf`) correlate with mlx-mast3r (fp32) at ≥ 0.99995, at both f32 and f16.

## Limitations

- Alignment uses only the pairs' 3D point maps, never reprojection or focal length.
  Upstream's sparse global alignment (`sparse_ga`) is not ported. Two known consequences:
  - Pairwise poses are not loop-consistent. On an 8-photo turntable capture, the 8 adjacent
    pair rotations add up to 471° instead of 360°, because the model predicts the object
    too thin from front to back. A 360° capture cannot close cleanly.
  - One wrong but confident pair, such as an object's front matched to its back, can fold
    the scene: views from opposite sides end up stacked.
- Not ported: the DUNE / DuneMASt3R models, retrieval-based pair selection, and the demo UI.
- The DPT head's full-resolution convolutions take most of the per-pair time.

## License

The code in this repository is licensed under [Apache-2.0](LICENSE), the license of
mlx-mast3r, from which it is ported.

**The model weights are not covered by this license.** MASt3R and its weights are © Naver
Corporation and licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), which means
**non-commercial use only**. The default download (`Aedelon/mast3r-vit-large-fp16`) is an fp16
conversion of
[naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric](https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric).

## Acknowledgements

- [MASt3R](https://github.com/naver/mast3r) and [DUSt3R](https://github.com/naver/dust3r) by
  Naver Labs Europe
- [mlx-mast3r](https://github.com/Aedelon/mlx-mast3r) by Delanoe Pirard (Aedelon)
- [candle](https://github.com/huggingface/candle) by Hugging Face
