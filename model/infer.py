"""Text/global-prefix inference using the validated three-stage Python decoder.

This reference CLI is context-bounded. The native App supplies interactive edits,
branch histories and context overclocking; those features are not claimed here.
"""
import argparse
import sys
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
import json
import math
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=HERE.parent/'weights/pop2-1b-epoch60/model.pt')
    p.add_argument('--tokenizer',type=Path,default=HERE/'assets/caption_spm_1500.model')
    p.add_argument('--prompt',default='',help='Chinese text; empty string uses the trained [BOS, EOS] no-text condition')
    p.add_argument('--prefix',type=Path,help='Optional .npy [T,13], BOS + background/global primitives only; no coarse/fine controls')
    p.add_argument('--output',type=Path,default=HERE/'outputs/inference')
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--device',default='auto',help='auto, cpu, cuda:0 (also used by Linux ROCm PyTorch)')
    p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--alpha',type=float,default=4.)
    p.add_argument('--cfg',type=float,default=1.)
    p.add_argument('--max-records',type=int,default=897,help='Geometry positions including BOS; also bounded by 1024 minus text width')
    p.add_argument('--compile',action='store_true',help='Opt in to validated static-KV torch.compile decoder on CUDA/ROCm')
    p.add_argument('--graph',action='store_true',help='Opt in to graph capture (requires --compile)')
    return p

def main():
    args=parser().parse_args()
    if args.batch<1 or not 2<=args.max_records<=897:raise ValueError('batch must be positive and max-records in [2,897]')
    if not math.isfinite(args.alpha) or args.alpha<1:raise ValueError('Rényi alpha must be finite and >= 1')
    if not math.isfinite(args.cfg) or args.cfg<0:raise ValueError('CFG must be finite and nonnegative')
    if args.graph and not args.compile:raise ValueError('--graph requires --compile')
    if args.cfg!=1 and (args.compile or args.graph):raise ValueError('The portable CFG reference uses eager paired KV; disable --compile/--graph, or use the native App')
    if not args.checkpoint.is_file():raise FileNotFoundError(f'{args.checkpoint}: download first with python tools/download_model.py from the release root')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Output must be empty; use a new directory to preserve previous generations')
    import torch
    import numpy as np
    import sentencepiece as spm
    from PIL import Image
    from block_state import canonical_bos_tokens, ATTRIBUTE_OFFSETS, ATTRIBUTE_BINS, ATTRIBUTES_PER_PRIMITIVE
    from token_layout import TOKEN_LAYOUT as L
    from model_gpt import GeometrizeGPT
    os.environ.setdefault('MPLBACKEND','Agg')
    import reference_visualize as v
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else torch.device(args.device)
    if args.compile and device.type!='cuda':raise RuntimeError('--compile requires a CUDA/ROCm PyTorch device; use eager on CPU')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA/ROCm device unavailable; no silent CPU fallback')
    sp=spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    ids=[2,*sp.encode(args.prompt),3] if args.prompt.strip() else [2,3]
    if (sp.vocab_size(),sp.pad_id(),sp.bos_id(),sp.eos_id())!=(1500,0,2,3):raise ValueError('Tokenizer contract mismatch')
    if len(ids)>224:raise ValueError(f'Prompt has {len(ids)} text tokens, limit 224; not truncated')
    if sp.unk_id() in ids:raise ValueError('Prompt contains text outside the released tokenizer vocabulary')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    protocol={'total_context_tokens':1024,'vocab_size':8097,'text_vocab_size':1500}
    for key,value in protocol.items():
        if checkpoint.get(key)!=value:raise ValueError(f'Checkpoint protocol mismatch: {key}={checkpoint.get(key)!r}, expected {value}')
    if checkpoint.get('architecture_version')!=GeometrizeGPT.ARCHITECTURE_VERSION:
        raise ValueError('Expected POP 2 shared-history text-prefix architecture metadata; old POP 1/unknown checkpoints are incompatible')
    dimensions={key:checkpoint.get(key) for key in ('model_d_model','model_n_layer','model_n_head','model_ffn_hidden')}
    if any(not isinstance(value,int) or isinstance(value,bool) or value<1 for value in dimensions.values()):
        raise ValueError('Checkpoint requires explicit positive integer model architecture dimensions')
    width,layers,heads,ffn=(dimensions[key] for key in ('model_d_model','model_n_layer','model_n_head','model_ffn_hidden'))
    if width%heads or (width//heads)%2:raise ValueError('Attention head width must divide hidden width and be even for RoPE')
    norm_eps=float(checkpoint.get('model_norm_eps',1e-6));rope_base=float(checkpoint.get('model_rope_base',10000.))
    if not math.isfinite(norm_eps) or norm_eps<=0 or not math.isfinite(rope_base) or rope_base<=0:
        raise ValueError('Invalid normalization/RoPE metadata')
    model=GeometrizeGPT(vocab_size=8097,d_model=width,n_layer=layers,n_head=heads,ffn_hidden=ffn,max_context_len=1024,max_block_positions=1024,text_vocab_size=1500,max_text_tokens=224,norm_eps=norm_eps,rope_base=rope_base)
    state=checkpoint.get('model_state_dict')
    if state is None:raise ValueError('model_state_dict missing; old POP 1 weights are incompatible')
    model.load_state_dict(state,strict=True)
    dtype=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16 if device.type=='cuda' else torch.float32
    model=model.to(device=device,dtype=dtype).eval().requires_grad_(False)
    meta={key:checkpoint.get(key) for key in ('epoch','global_step','architecture_version')};meta.update(dimensions);meta['parameter_count']=sum(p.numel() for p in model.parameters());del checkpoint,state
    prefix=canonical_bos_tokens(1,'cpu').reshape(1,ATTRIBUTES_PER_PRIMITIVE)
    if args.prefix:
        a=np.load(args.prefix,allow_pickle=False)
        if a.ndim!=2 or a.shape[1]!=ATTRIBUTES_PER_PRIMITIVE or len(a)<1 or len(a)>87 or not np.issubdtype(a.dtype,np.integer):raise ValueError('Prefix must be integer [T,13], 1..87 records')
        prefix=torch.tensor(a,dtype=torch.long)
        if not torch.equal(prefix[0],canonical_bos_tokens(1,'cpu')[0]):raise ValueError('Prefix requires a unique leading geometry BOS')
        for i,row in enumerate(prefix[1:],1):
            for f,(offset,count) in enumerate(zip(ATTRIBUTE_OFFSETS,ATTRIBUTE_BINS)):
                if not offset<=int(row[f])<offset+count:raise ValueError('Prefix attributes out of vocabulary')
            if i==1 and int(row[5])!=L.shape_offset:raise ValueError('First visible record must be background shape bin 0')
            if i>1 and int(row[5])-L.shape_offset not in (1,2):raise ValueError('Only complete global ordinary primitives supported by this prefix CLI')
    target=min(args.max_records,1024-len(ids))
    if len(prefix)>=target:raise ValueError('Prefix leaves no requested inference context')
    v.RENYI_ALPHA=args.alpha;v.SAMPLING_MODE='renyi';v.TARGET_SHAPE_STEPS=target;v.START_STEPS=len(prefix)
    v.ENABLE_OPTIMIZED_INFERENCE=bool(args.compile);v.OPTIMIZED_INFERENCE_COMPILE=bool(args.compile);v.OPTIMIZED_INFERENCE_GRAPH=bool(args.graph)
    v.config.CAPTION_TOKENIZER_PATH=str(args.tokenizer)
    if args.cfg!=1:
        from cfg import GuidedModel
        model=GuidedModel(model,args.cfg)
    inputs=prefix.unsqueeze(0).expand(args.batch,-1,-1).to(device)
    text=torch.tensor(ids,device=device,dtype=torch.long).unsqueeze(0).expand(args.batch,-1)
    mask=torch.ones_like(text,dtype=torch.bool)
    torch.manual_seed(args.seed)
    class RandomLanes:
        def __init__(self):self.rngs=[np.random.default_rng(args.seed+i) for i in range(args.batch)]
        def for_lane(self,lane):return self.rngs[lane]
        def random(self):return self.rngs[0].random()
    with torch.inference_mode():
        outputs=v.generate_candidate_batch(model,inputs,device,RandomLanes(),intervention_policy='normal',text_ids=text,text_mask=mask)
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(checkpoint=meta,device=str(device),dtype=str(dtype),alpha=args.alpha,cfg=args.cfg,seed=args.seed,text_only=not bool(args.prefix),context_overclock=False,within_support_sampling='uniform',results=[])
    report['acceleration']=v.get_optimized_inference_info(model) if args.cfg==1 else {'optimized':False,'reason':'eager paired logit CFG reference'}
    for i,(tokens,closed,order,active) in enumerate(outputs,1):
        records=tokens.numpy().reshape(-1,ATTRIBUTES_PER_PRIMITIVE)
        np.save(args.output/f'{i:03d}.npy',records)
        Image.fromarray(v.rasterize_render_data(v.decode_tokens_to_render_data(tokens))).save(args.output/f'{i:03d}.png')
        report['results'].append(dict(candidate=i,geometry_records=len(records),completed_regions=closed,active_region=active,all_regions_complete=closed==80,stop_reason='all_regions_complete' if closed==80 else 'context_or_requested_limit'))
    (args.output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
