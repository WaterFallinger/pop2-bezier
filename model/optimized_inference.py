"""Production visualization adapter: lazy serialized compilation, reusable graph."""
import os
from pathlib import Path
import threading
import time
import torch
from filelock import FileLock
from optimized_decode import StaticKVModel,RenyiGPUSampler

def serious_gpu_error(error):
    message=str(error).lower()
    return any(s in message for s in ('illegal memory access','device-side assert','device lost','hiperrorillegaladdress'))

class VisualizationInferenceEngine(StaticKVModel):
    def __init__(self,model,compile_enabled=True,graph_enabled=True,compile_threads=4,alpha=14.):
        super().__init__(model,fixed_capacity=compile_enabled,packed_text_positions=True)
        self.generation_lock=threading.RLock()
        self.graph_enabled=graph_enabled
        self.sampler=RenyiGPUSampler(next(model.parameters()).device,alpha)
        self.attempted=set()
        self.info=dict(optimized=True,packed_text_positions=True,compile_enabled=compile_enabled,
                       graph_enabled=False,compile_threads=compile_threads,warmup_seconds=0.,
                       preserve_eager_precision=True)
        if compile_enabled:
            try:self.enable_compile(compile_threads,preserve_eager_precision=True)
            except Exception as e:
                self.compiled=None;self.fixed_capacity=False
                self.info.update(compile_enabled=False,compile_error=repr(e))

    def _prepare(self,tokens,position_ids):
        cache=self.cache
        key=(tokens.shape[0],str(tokens.device),str(cache.keys[0].dtype))
        if key in self.attempted:return
        self.attempted.add(key)
        if self.compiled is None:return
        root=Path(os.environ.get('PAINTER_INFERENCE_CACHE',str(Path.home()/'.cache/primitive_painter_inference')))
        root.mkdir(parents=True,exist_ok=True)
        began=time.perf_counter()
        with FileLock(str(root/'warmup.lock')):
            acquired=time.perf_counter()
            print(f'[InferenceOptimize] cuda:{tokens.device.index} batch={tokens.shape[0]} serialized compile/graph warmup',flush=True)
            with torch.random.fork_rng(devices=[tokens.device.index]):
                cache.write_index.fill_(cache.length)
                pos=position_ids
                if pos is None:pos=torch.full((tokens.shape[0],1),cache.length-cache.text_width,device=tokens.device,dtype=torch.long)
                absolute=pos+(cache.position_offsets if cache.position_offsets is not None else cache.text_width)
                args=(tokens,absolute,cache.keys,cache.values,cache.valid,cache.positions,cache.write_index,None)
                try:
                    self.compiled(*args);torch.cuda.synchronize(tokens.device)
                except Exception as e:
                    if serious_gpu_error(e):raise
                    self.compiled=None;self.fixed_capacity=False
                    self.info.update(compile_enabled=False,compile_error=repr(e))
                    print('[InferenceOptimize] compile unavailable; using eager static KV: '+repr(e),flush=True)
                if self.compiled is not None and self.graph_enabled:
                    try:self.capture_graph(tokens,pos);self.info['graph_enabled']=True
                    except Exception as e:
                        if serious_gpu_error(e):raise
                        self.graph=None;self.graph_inputs=None;self.graph_outputs=None
                        self.info.update(graph_enabled=False,graph_error=repr(e))
                        print('[InferenceOptimize] graph unavailable; using compiled decode: '+repr(e),flush=True)
        self.info.update(warmup_seconds=time.perf_counter()-began,lock_wait_seconds=acquired-began,
                         compile_graph_seconds=time.perf_counter()-acquired,active_batch=tokens.shape[0],
                         kv_capacity=cache.keys[0].shape[2],weight_dtype=str(next(self.model.parameters()).dtype))
        print('[InferenceOptimize] ready '+str(self.info),flush=True)

    def __call__(self,tokens,position_ids=None,past_key_values=None,**kwargs):
        if past_key_values is not None:self._prepare(tokens,position_ids)
        return super().__call__(tokens,position_ids=position_ids,past_key_values=past_key_values,**kwargs)

    def stats(self):
        return dict(self.info,alpha=self.sampler.alpha,forward_calls=self.forward_calls,
                    region_entropy_audit=self.sampler.audit_stats())
