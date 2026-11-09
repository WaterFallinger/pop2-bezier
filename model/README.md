# Model, training and Python reference inference

This is the POP 2 / v3.3 shared-history Multi-Lane model code. The published 1B checkpoint has 35 layers, hidden width 1536, 24 heads, FFN 4096 and 1,004,182,914 parameters. One full primitive combines nine discrete attribute embeddings into a history position. Training samples two attribute queries per supervised primitive by default (`RAND_K=2`) while sharing complete history. Learned empty attributes, text-prefix conditioning, the termination head, numerical guards, and resumable checkpoint protocol are retained.

The training core comes from the actual later A800 production snapshot, not a newly invented replacement. Machine-specific deployment, remote uploads, BMC tools, experiment outputs and optimizer-reset trials are excluded. Portable changes are recorded in `SOURCE_PROVENANCE.json`.

## Environment

Use Python 3.10+ and install a hardware-appropriate recent **PyTorch CUDA or ROCm build** following PyTorch's instructions, then from the release root:

```sh
python -m pip install -r model/requirements.txt
```

Production fused/fixed-grid training and multi-GPU DDP are intended for **Linux with supported CUDA or ROCm PyTorch/Inductor/Triton**. Required APIs include `torch.compile` and `torch.nn.attention.flex_attention.{create_block_mask, flex_attention}`; optional compilation settings can vary by PyTorch version. This release does not assert that every version exposing those APIs is validated.

Recorded environment evidence: the historical local 0.1B fullgraph/FP32-master GPU check used **2.11.0+cu130**; archived AutoDL 1B acceptance reports used **2.12.1+cu130**. The source-packaging CPU tests here used **2.11.0+cu130**. These are separate recorded checks, not a newly rerun multi-GPU production benchmark or a claim that 2.11 is the unique minimum. The shipped Windows App supporting NVIDIA/AMD does not mean PyTorch fused training is verified on Windows. CPU protocol/reference tests run without a GPU; they do not certify 1B CPU training speed. The default public training CLI uses one device and no cloud archive service. `--compile-level 0` disables outer-model compilation; the production fixed-grid FlexAttention kernel still uses Inductor/Triton. CPU tests exercise the explicit reference path, not a promise of kernel-free CPU production training. `--compile-level 2` and GPU FP32-master sharding are opt-in.

## Prepare your own data

Run `data_tools` to produce 512-pixel native sequences. The accepted root contains `manifest.json`, `v1/`, optionally `v2/` etc. Each version needs packed `data_part_N.csv` and a committed `.completed_images.txt`. Do not supply unfinished generator outputs or old 144-step POP 1 CSVs.

You can start with the included purely synthetic example. `examples/synthetic/captions.json` maps each extension-free `image_name` to a prompt string or equal-length list of prompt variants. No private image selection list or original 100k-image corpus is required.

```sh
python model/prepare_data.py --data-dir examples/synthetic/sequences --captions examples/synthetic/captions.json --output work/synthetic-captions
```

The preparer validates every caption before writing, creates the selection index and compact token sidecar, and records the variant count. Any positive sample count and any uniform positive number of prompt variants are supported. Text over **224 tokens including BOS/EOS** raises an error; it is not truncated or silently skipped. Existing nonempty preparation directories are not overwritten.

For explicit image-only training, use a separate output directory:

```sh
python model/prepare_data.py --data-dir examples/synthetic/sequences --unconditional --output work/synthetic-no-text
```

This writes the same canonical `[BOS, EOS] = [2, 3]` condition used by training's text dropout. It does not encode a fake descriptive prompt. The released 1500-piece SentencePiece tokenizer is in `model/assets/`; training a different tokenizer changes the text embedding contract and is not a drop-in replacement for the published weights.

## Train from scratch

Start with a small architecture to validate your own data pipeline on a supported GPU:

```sh
python model/train.py --data-dir examples/synthetic/sequences --caption-root work/synthetic-captions --checkpoint-dir work/tiny-checkpoints --versions 1 --epochs 2 --micro-batch 1 --accumulation 1 --tiny --from-scratch
```

`--tiny` uses the same protocol/body with two layers, width 64, four heads and FFN 128. It is not compatible with released 1B weights. Omit `--tiny` to train the released 1B architecture. The example is only a pipeline example, not a useful training corpus.

A Linux three-GPU run with BF16 working weights and FP32 GPU masters/Adam state can use:

```sh
torchrun --standalone --nproc_per_node=3 model/train.py --data-dir /data/sequences --caption-root /data/captions --checkpoint-dir /data/checkpoints --versions 50 --epochs 60 --micro-batch 10 --accumulation 20 --gpu-fp32-master --compile-level 2 --from-scratch
```

Cold FP32 rank initialization is serial by default to avoid assuming the original 360-GiB host; increase `POP_INITIALIZATION_PARALLELISM` only with sufficient host RAM.

Effective global batch is `world_size × micro_batch × accumulation`. The CLI derives device count from `torchrun` rather than assuming two or three A800s. For ordinary single-device/consolidated training, leave `--gpu-fp32-master` off. Training equations and the original LR schedule live in `train_gpt_pretrain.py`; architecture and `RAND_K` controls live in `config.py`.

## Initialize or resume correctly

From the release root, download inference weights with:

```sh
python tools/download_model.py
```

Then initialize a **new** optimizer/run from them:

```sh
python model/train.py --data-dir /data/sequences --caption-root /data/captions --checkpoint-dir work/finetune --versions 50 --epochs 10 --micro-batch 1 --init-weights weights/pop2-1b-epoch60/model.pt
```

`--init-weights` deliberately starts Epoch 1/global step 0 and new Adam state. It is different from exact resume. The Hugging Face inference package has no optimizer shards. A nonempty destination with existing checkpoints is rejected for a new initialization, preserving old runs.

For exact resume, provide a complete checkpoint directory from the same training mode:

```sh
python model/train.py --data-dir /data/sequences --caption-root /data/captions --checkpoint-dir /data/checkpoints --resume-dir /data/checkpoints --versions 50 --epochs 70
```

The sharded GPU-master mode requires the complete `READY.json`, `model.pt` and all matching `optimizer_rank_N.pt` shards, with the expected world size; launch matching `torchrun` and `--gpu-fp32-master`. Other world sizes require explicit resharding and are not silently accepted. Only load trusted PyTorch checkpoints.

Default local runs **retain completed epoch checkpoints**, do not contact your old server, do not delete them to satisfy a two-checkpoint relay limit, and do not wait for cloud archive ACKs. Reserve enough disk for weights and optimizer state. Original atomic checkpoint commit, size/format checks and low-space waiting remain. If you intentionally integrate your own archiver, `--archive-ack-control DIR` enables the old explicit ACK gate; the release does not supply or start that private WAN deployment.

## Python reference inference

```sh
python model/infer.py --checkpoint weights/pop2-1b-epoch60/model.pt --prompt "金发蓝眼的白皙圆脸女性，明亮的动画风格。" --batch 1 --alpha 4 --cfg 1 --output work/inference
```

The CLI loads real weights strictly, never substitutes a random model when weights are missing. It reads positive architecture dimensions from POP 2 checkpoint metadata, so matching self-trained/tiny models can be evaluated too; protocol vocabulary 7069/1500, total context1024, architecture version and state-dict shapes remain strict. POP 1 and metadata-free checkpoints are rejected. It supports no-text/Chinese text and an optional `.npy [T,9]` **global-stage prefix** (unique BOS, background and up to 85 global shapes). It writes PNGs, token `.npy` files and a JSON result. The sampler selects the top rounded `exp(H_alpha)` legal candidates and draws **uniformly** within that support. Background uses shape bin 0, ordinary shapes bins 1/2; the stage controller uses 85 global shapes plus background, coarse cap 20, fine cap 10.

`--cfg 2` uses the verified paired conditional/empty-text **logit** guidance before Rényi sampling. CFG 1 directly uses the conditional model. The eager CFG reference shares model weights and separates KV batch rows. `--compile` and `--graph` opt into the formal static-KV GPU adapter for CFG 1; unsupported combinations raise an error instead of pretending to accelerate.

This Python entry is deliberately **context-bounded**: raw geometry is at most 896 records plus the extra BOS, and text+geometry stays within 1024 positions. `--max-records` can stop earlier. Its result records whether all 80 regions completed; reaching context is not called full completion. No experimental rolling-region scripts are bundled as if they were equivalent to the current App.

For the latest interactive canvas, arbitrary edits, state-tree branching, region reassignment, partial revisits and context-overclocking policy, use the native App and the shipped `app/` source. The Python CLI does not promise those application features.

## Validation

```sh
python -m unittest discover -s model/tests -v
```

Release checks cover a CPU small-model shared-history backward pass, causal prefix invariance, the actual Python decoder's background/ordinary shape rules, exact CFG=1 and residual arithmetic, variable caption cardinality and strict long-text rejection, local checkpoint retention without an archiver, explicit ACK waiting without unsafe deletion, and preservation of existing runs. A synthetic sequence produced by the real data generator was converted into the trainer's mmap input successfully. Full 1B training was not rerun for this source packaging task.

An additional real public-CLI lifecycle check ran the included synthetic sample with **630,850 parameters**, ordinary AdamW, micro-batch1/accumulation1 and outer compile-level0 on the local RTX5070Ti Laptop. It trained two epochs, saved complete checkpoints, restored epoch2 including Adam state to train epoch3, and evaluated that trained checkpoint through `infer.py` for eight geometry records. Saved global_step/Adam step counts were exactly1→2→3; model-only initialization into a separate new run was also checked to restart at1 instead of reusing step3.

That particular Windows test used the machine's pre-existing **PyTorch2.11.0+cu130, triton-windows3.6.0.post26 and MSVC/Windows SDK**, with `TORCHINDUCTOR_COMPILE_THREADS=1` for the Windows process-pool limitation. These packages are not bundled here. It validates this configured local stack, not arbitrary fresh Windows Python installations or full-scale multi-GPU production training.
