"""Build the portable caption sidecar for a generated sequence dataset."""
import argparse
import sys
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
import csv
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--captions',type=Path,help='JSON mapping image_name -> string or equal-length list of variants')
    p.add_argument('--unconditional',action='store_true',help='Explicit no-text training: use [BOS, EOS] for every sample')
    p.add_argument('--tokenizer',type=Path,default=Path(__file__).resolve().parent/'assets/caption_spm_1500.model')
    args=p.parse_args()
    if bool(args.captions)==bool(args.unconditional):p.error('Choose exactly one of --captions or --unconditional')
    files=list(args.data_dir.glob('v*/.completed_images.txt'))
    if not files: raise FileNotFoundError('No generated vN/.completed_images.txt manifests found')
    names=None
    for file in sorted(files):
        values=file.read_text(encoding='utf-8').splitlines()
        if not values or any(not n.strip() for n in values) or len(values)!=len(set(values)):raise ValueError(f'Invalid completion manifest: {file}')
        current=set(values)
        if names is None:names=current
        elif current!=names:raise ValueError('All selected geometry versions must have the same image names')
    names=sorted(names)
    captions=json.loads(args.captions.read_text(encoding='utf-8')) if args.captions else {n:[''] for n in names}
    if not isinstance(captions,dict):raise ValueError('Captions must be a JSON object keyed by image_name')
    if set(captions)!=set(names):raise ValueError('Caption keys must exactly match sequence image names; no silent missing/extra entries')
    rows=[];variants=None
    import sentencepiece as spm
    sp=spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if (sp.vocab_size(),sp.pad_id(),sp.bos_id(),sp.eos_id())!=(1500,0,2,3):raise ValueError('Released tokenizer contract mismatch')
    # Validate all input before creating output files.
    for image_id,name in enumerate(names,1):
        values=captions[name];values=[values] if isinstance(values,str) else values
        if not isinstance(values,list) or not values or any(not isinstance(x,str) for x in values):raise ValueError(f'Invalid captions for {name}')
        if not args.unconditional and any(not x.strip() for x in values):raise ValueError(f'Empty caption for {name}; use --unconditional explicitly')
        if variants is None:variants=len(values)
        if len(values)!=variants:raise ValueError('Every image must have the same positive variant count')
        for variant,text in enumerate(values,1):
            ids=[2,*sp.encode(text),3]
            if len(ids)>224:raise ValueError(f'{name} variant {variant}: {len(ids)} text tokens exceeds 224; input preserved, not truncated')
        filename=name+'.png'
        if Path(filename).stem!=name:raise ValueError('Use extension-free image_name as written by the generator')
        rows.append((image_id,name,filename,values))
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Caption output must be empty; preserve old caches by using a new directory')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'selected_images.tsv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['priority','image_id','filename'],delimiter='\t');writer.writeheader()
        for priority,(image_id,name,filename,values) in enumerate(rows):
            writer.writerow({'priority':priority,'image_id':image_id,'filename':filename})
            target=args.output/'prompts'/f'{(image_id-1)//1000:03d}'/f'{image_id:06d}.json';target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps({'filename':filename,'prompts':[{'variant':i,'text':text or '[UNCONDITIONAL]'} for i,text in enumerate(values,1)]},ensure_ascii=False),encoding='utf-8')
    from config import TrainConfig
    from caption_data import build_caption_cache
    TrainConfig.CAPTION_TOKENIZER_PATH=str(args.tokenizer.resolve());TrainConfig.CAPTION_SELECTION_PATH=str((args.output/'selected_images.tsv').resolve())
    TrainConfig.CAPTION_PROMPTS_DIR=str((args.output/'prompts').resolve());TrainConfig.CAPTION_CACHE_DIR=str((args.output/'cache').resolve());TrainConfig.CAPTION_VARIANTS=variants
    TrainConfig.CAPTION_EXPECTED_IMAGES=len(rows)
    if args.unconditional:
        # Canonical explicit no-text sidecar: never encode a placeholder as a prompt.
        import numpy as np
        cache=args.output/'cache';cache.mkdir();offsets=np.empty((len(rows),1,2),dtype=np.int64)
        for i in range(len(rows)):offsets[i,0]=[i*2,2]
        np.tile(np.array([2,3],dtype='<u2'),len(rows)).tofile(cache/'captions.tokens.bin');np.save(cache/'captions.offsets.npy',offsets)
        (cache/'manifest.json').write_text(json.dumps({'schema_version':1,'image_count':len(rows),'variants_per_image':1,'token_count':2*len(rows),'vocab_size':1500,'max_text_tokens':224}),encoding='utf-8')
    else: build_caption_cache()
    (args.output/'prepared.json').write_text(json.dumps({'images':len(rows),'variants':variants,'unconditional':args.unconditional,'text_max_tokens':224},indent=2),encoding='utf-8')
    print(json.dumps({'images':len(rows),'variants':variants,'caption_root':str(args.output.resolve())}))
if __name__=='__main__': main()
