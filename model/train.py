"""Portable entry for the released shared-history Multi-Lane trainer."""
import argparse
import sys
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True, help='Generated v1..vN sequence directories')
    p.add_argument('--caption-root', type=Path, required=True, help='Output from prepare_data.py')
    p.add_argument('--checkpoint-dir', type=Path, default=HERE/'outputs/checkpoints')
    p.add_argument('--resume-dir', type=Path, help='Full compatible checkpoint directory, including optimizer shards')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--init-weights', type=Path, help='Model-only initialization; deliberately starts a new optimizer and epoch 1')
    group.add_argument('--from-scratch', action='store_true', help='Explicitly allow random initialization when no checkpoint exists')
    p.add_argument('--epochs', type=int, default=40, help='Absolute final epoch, not additional epochs')
    p.add_argument('--versions', type=int, default=1)
    p.add_argument('--caption-variants', type=int, help='Defaults to prepare_data.py metadata')
    p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--compile-level', type=int, choices=(0,1,2), default=0)
    p.add_argument('--gpu-fp32-master', action='store_true', help='Linux DDP only: BF16 working model + ZeRO-1 FP32 GPU masters and sharded optimizer')
    p.add_argument('--tiny', action='store_true', help='Small architecture for pipeline exploration; incompatible with released 1B weights')
    p.add_argument('--archive-ack-control', type=Path, help='Opt-in external archive ACK control root; local default keeps complete epochs')
    return p

def configure(args):
    if args.caption_variants is None:
        prepared=args.caption_root/'prepared.json'
        args.caption_variants=int(json.loads(prepared.read_text(encoding='utf-8'))['variants']) if prepared.is_file() else 1
    if args.resume_dir and (args.init_weights or args.from_scratch):
        raise ValueError('Choose resume or a new initialization, not both')
    if (args.init_weights or args.from_scratch) and args.checkpoint_dir.exists() and any(args.checkpoint_dir.glob('geometrize_gpt_pretrain_epoch_*')):
        raise FileExistsError('New initialization requires an empty checkpoint destination; existing runs are preserved')
    for name in ('epochs','versions','caption_variants','micro_batch','accumulation'):
        if getattr(args,name)<1: raise ValueError(f'{name} must be positive')
    if not (args.init_weights or args.from_scratch or args.resume_dir) and not any(args.checkpoint_dir.glob('geometrize_gpt_pretrain_epoch_*')):
        raise ValueError('Select --from-scratch, --init-weights, or a full --resume-dir for the first run')
    from config import TrainConfig
    world=int(os.environ.get('WORLD_SIZE','1'))
    if world>1 and os.name=='nt': raise RuntimeError('This DDP trainer uses NCCL/RCCL: run multi-GPU training on Linux')
    if args.gpu_fp32_master and world<2: raise ValueError('--gpu-fp32-master requires torchrun with two or more ranks')
    TrainConfig.DATA_DIR=str(args.data_dir.resolve())
    TrainConfig.DATA_CACHE_DIR=str((args.checkpoint_dir.resolve().parent/'geometry_cache'))
    TrainConfig.TORCH_COMPILE_CACHE_DIR=str(args.checkpoint_dir.resolve().parent/'compile_cache')
    os.environ.setdefault('A800_RUNTIME_ROOT',str(args.checkpoint_dir.resolve().parent))
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',TrainConfig.TORCH_COMPILE_CACHE_DIR)
    os.environ.setdefault('TRITON_CACHE_DIR',str(args.checkpoint_dir.resolve().parent/'triton_cache'))
    TrainConfig.CAPTION_SELECTION_PATH=str(args.caption_root.resolve()/'selected_images.tsv')
    TrainConfig.CAPTION_PROMPTS_DIR=str(args.caption_root.resolve()/'prompts')
    TrainConfig.CAPTION_CACHE_DIR=str(args.caption_root.resolve()/'cache')
    TrainConfig.CAPTION_VARIANTS=args.caption_variants
    TrainConfig.CKPT_DIR=str(args.checkpoint_dir.resolve())
    TrainConfig.RESUME_CKPT_DIR=str((args.resume_dir or args.checkpoint_dir).resolve())
    TrainConfig.PARALLEL_MODE='ddp' if world>1 else 'single'
    TrainConfig.PARALLEL_DEVICE_IDS=list(range(world))
    TrainConfig.MICRO_BATCH_SIZE=args.micro_batch
    TrainConfig.GRADIENT_ACCUMULATION_STEPS=args.accumulation
    TrainConfig.VERSION_CYCLE_EPOCHS=args.versions
    TrainConfig.GPU_FP32_MASTER_WEIGHTS=args.gpu_fp32_master
    TrainConfig.SHARDED_GPU_MASTER_CHECKPOINTS=args.gpu_fp32_master
    TrainConfig.TORCH_COMPILE_LEVEL=args.compile_level
    TrainConfig.RANDOM_INITIALIZATION_IF_NO_CHECKPOINT=args.from_scratch
    if args.tiny:
        TrainConfig.MODEL_D_MODEL=64;TrainConfig.MODEL_N_LAYER=2
        TrainConfig.MODEL_N_HEAD=4;TrainConfig.MODEL_FFN_HIDDEN=128
    os.environ['POP_EPOCHS']=str(args.epochs)
    if args.init_weights:
        if not args.init_weights.is_file(): raise FileNotFoundError(args.init_weights)
        os.environ['POP_INIT_WEIGHTS']=str(args.init_weights.resolve())
    else: os.environ.pop('POP_INIT_WEIGHTS',None)
    control=args.checkpoint_dir.resolve().parent/'control'
    os.environ['V33_TRAINING_CONTROL_DIR']=str((args.archive_ack_control or control).resolve())
    os.environ['POP_REQUIRE_ARCHIVE_ACK']='1' if args.archive_ack_control else '0'
    if args.resume_dir and not args.gpu_fp32_master:
        candidates=sorted(args.resume_dir.glob('geometrize_gpt_pretrain_epoch_*.pt'))
        if not candidates: raise ValueError('Single/consolidated resume requires .pt checkpoints, not model-only Hugging Face weights')
        from train_gpt_pretrain import get_ckpt_sort_key
        os.environ['V32_RESUME_CHECKPOINT_PATH']=str(max(candidates,key=lambda path:get_ckpt_sort_key(str(path))).resolve())
    return TrainConfig

def main():
    args=parser().parse_args()
    configure(args)
    import train_gpt_pretrain
    print('Local training: native primitive protocol, no WAN/archive daemon; complete epoch checkpoints retained.',flush=True)
    train_gpt_pretrain.run_training()

if __name__=='__main__': main()
