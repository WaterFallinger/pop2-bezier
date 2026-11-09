# Bezier edge control for primitive operations

This fork adds **per-rim quadratic Bezier control** to the rectangle primitives of
[Primitive Operation Painter 2](https://github.com/wonderfulearth/primitive-operation-painter)
(MIT). Ellipses are unchanged.

## Why

The stock primitive set is a rectangle or an ellipse, each with only position, angle,
size and colour. That limits every stroke to two shapes. Replacing a rim with a quadratic
Bezier lets a single primitive bulge, taper or curve, so a long thin bent region can act
as a brush stroke.

The key point is that this is **cheap**. A symmetrically parameterised quadratic Bezier rim
collapses to a parabola:

```
x = hw * (1 + e/2 * (1 - (y/hh)^2))
```

so the signed distance of the four rims has a closed form and costs the same as the
straight rectangle case. No polygon subdivision, no root finding, no extra passes.

## Protocol change

Each primitive went from **9 to 13 tokens**; the token vocabulary grew from 7069 to 8097.

| field | bins | offset | notes |
|---|---|---|---|
| x, y | 1024 each | 0, 1024 | half-pixel |
| angle | 270 | 2048 | 1/3 degree |
| width, height | 2048 each | 2318, 4366 | quarter-pixel |
| shape | 256 | 6414 | 0 background / 1 rect / 2 ellipse |
| r, g, b | 128 each | 6670 / 6798 / 6926 | ×2 → 0..254 |
| **e0..e3** | **256 each** | **7054 / 7310 / 7566 / 7822** | rim midpoint offset, normalised [-1, 1], 0 = straight |
| special | 4 | 8078 | |
| empty | 12 | 8082 | 8 original + 4 for e0..e3 |
| true_end / interrupt_end / BOS | 3 | 8094 / 8095 / 8096 | |

`token_layout.py` asserts the new sizes. The old 64/192/256 layouts are untouched.

## What was changed

| file | change |
|---|---|
| `model/token_layout.py` | `edge_bins/edge_count`, `edge_offset(s)`, shifted `special_offset`, `field_slice` support, new version string |
| `model/block_state.py` | 13 attribute names; offsets/bins derived through `field_slice` |
| `model/config.py` | `TOKENS_PER_STEP = ATTRIBUTES_PER_PRIMITIVE` (13) |
| `model/dataset_gpt_full_sequence.py` | optional `e0..e3` CSV columns (old CSVs still load); cache format bumped to v7 |
| `model/reference_visualize.py` | encode/decode 13 fields; `_composite_shape` reads edge values |
| `model/infer.py` | vocab 8097, prefix `[T,13]` |
| `model/checkpoint_compat.py` | **`load_bezier_initialization`**: maps a 7069-vocab checkpoint onto 8097 (attributes copied, edge rows zero-initialised) |
| `model/train_gpt_pretrain.py` | opt-in `POP_BEZIER_INIT=1` to use the above |
| `model/fused_multilane.py` | falls back to the PyTorch reference attention when Triton is unavailable |
| `data_tools/.../shader.wgsl` | `GpuShape.e0..e3`; closed-form bent-rim coverage; edge mutation and initialisation; bbox inflation |
| `data_tools/.../main.rs` | `EDGE_MUTATION_WEIGHT` (0 disables bending for A/B baselines), `EDGE_MUTATION_RANGE` |
| `data_tools/.../types.rs` | four extra `f32` in `GpuShape` |
| `data_tools/.../v3.rs` | four extra CSV columns; format string bumped |
| `data_tools/.../tools/render_csv.py` | replay supports bent rims |

## Results (same image, same seed, only the edge switch differs)

Two 512×512 crops of an anime illustration; 896-record budget; `--mode normal`.

| crop | shapes off → on | bent share | RMSE off → on | change |
|---|---|---|---|---|
| upper body | 716 → 698 (−2.5%) | 84.0% | 15.381 → 14.168 | **−7.9%** |
| lower body | 770 → 762 (−1.0%) | 82.5% | 18.003 → 16.336 | **−9.3%** |

Serialisation time per batch: 17.3 s → 27.3 s (+58 %) on an RTX 4060 Laptop.

The baseline build is verified to produce **0 %** bent primitives, so the comparison is
`bent` vs `straight` rather than two random draws.

```
work/ab3/A-B_anime_upper.png     labelled side-by-side
work/ab3/A-B_anime_mid.png
```

## Build

The generator is Rust + WGSL and needs a working Vulkan/DX12 device. On Windows with
Rust's MSVC toolchain, a full MSVC installation is required. This fork was built with:

- `rustup toolchain install stable-x86_64-pc-windows-gnu`
- MSYS2 `binutils 2.47` + `gcc 16.2` (mingw-w64)
- all crates vendored locally (`scripts/vendor_crates.py`) and
  `data_tools/image_to_sequence/.cargo/config.toml` copied from `config.toml.example`

```sh
cd data_tools/image_to_sequence
cargo +stable-x86_64-pc-windows-gnu build --locked --offline --release
```

`scripts/build_variants.py` builds both the bent and the straight binaries for A/B and
records their hashes. **Paths inside `scripts/` assume this repository layout; adjust
them if you relocate files.**

## Train / infer

The Python side is unchanged apart from the wider token:

```sh
# strict protocol check + forward pass
python scripts/verify_pop2.py

# initialise a new run from a 7069-vocab checkpoint
POP_BEZIER_INIT=1 python model/train.py --data-dir ... --caption-root ... \
    --checkpoint-dir ... --init-weights weights/pop2-1b-epoch60/model.pt

# inference with the new vocabulary
python scripts/run_infer.py args.json
```

## Limitations

- Only the rectangle rims bend; ellipses keep the original analytic SDF.
- No model has been trained on the 13-field protocol yet. The numbers above describe
  what the **data generator** gains, not what a trained model can produce.
- The A/B uses two crops and one seed; larger samples and edge-distribution statistics
  are outstanding.
- The seam between a bent rectangle and its neighbours is not smoothed; corner handling
  is `max()` of four rim distances.

## Licence

Upstream is MIT; this fork keeps the same licence and the original `LICENSE`,
`THIRD_PARTY_NOTICES.md` and attribution.
