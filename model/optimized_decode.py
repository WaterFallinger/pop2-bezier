"""Inference-only static KV and GPU-resident Renyi sampling; no weight changes."""
from dataclasses import dataclass
import math
import os
import torch
from torch.nn import functional as F


@dataclass
class KVStore:
    keys: list
    values: list
    valid: torch.Tensor
    positions: torch.Tensor
    write_index: torch.Tensor
    length: int = 0
    text_width: int = 0
    text_lengths: torch.Tensor = None
    position_offsets: torch.Tensor = None


class StaticKVModel:
    def __init__(self, model, fixed_capacity=False, packed_text_positions=False):
        self.model=model
        self.fixed_capacity=fixed_capacity
        self.packed_text_positions=packed_text_positions
        self.cache=None
        self.compiled=None
        self.graph=None
        self.graph_inputs=None
        self.graph_outputs=None
        self.forward_calls=0

    def reset_count(self): self.forward_calls=0

    def _decode(self,tokens,absolute_positions,keys,values,valid,positions,index,end):
        x=self.model.encode_primitive_tokens(tokens)
        rope=self.model.h[0].attn.rope.cos_sin(absolute_positions,x.dtype)
        # end=None is the fixed-shape compiled/graph path. No GPU scalar is read on CPU.
        if end is None:
            mask=(valid & (positions<=index[0]).unsqueeze(0))[:,None,None,:]
        else:
            mask=valid[:,:end,None].transpose(1,2).unsqueeze(1)
        for i,block in enumerate(self.model.h):
            normalized=block.norm_1(x)
            q,k,v=block.attn.project_qkv(normalized,rotary_cos_sin=rope)
            q=q.transpose(1,2);k=k.transpose(1,2);v=v.transpose(1,2)
            # Partial-attribute predictions overwrite a scratch slot; only a
            # complete primitive advances the committed length in __call__.
            keys[i].index_copy_(2,index,k);values[i].index_copy_(2,index,v)
            kk=keys[i] if end is None else keys[i][:,:,:end,:]
            vv=values[i] if end is None else values[i][:,:,:end,:]
            y=F.scaled_dot_product_attention(q,kk,vv,attn_mask=mask,is_causal=False)
            y=y.transpose(1,2).contiguous().view(x.shape)
            x=x+block.attn.c_proj(y)
            x=x+block.mlp(block.norm_2(x))
        hidden=self.model.final_norm(x)
        return self.model.project_attribute_logits(hidden),self.model.termination_head(hidden)

    def enable_compile(self,threads=4,preserve_eager_precision=False):
        self.fixed_capacity=True
        options={'triton.cudagraphs':False,'compile_threads':threads}
        if preserve_eager_precision:options['emulate_precision_casts']=True
        self.compiled=torch.compile(self._decode,backend='inductor',fullgraph=True,dynamic=False,
                                    options=options)

    def __call__(self,tokens,position_ids=None,past_key_values=None,use_cache=False,
                 return_termination=False,text_ids=None,text_mask=None,**kwargs):
        if kwargs:raise ValueError('This validation adapter only supports the visualization decode contract')
        self.forward_calls+=1
        if past_key_values is None:
            if not use_cache:raise ValueError('The optimized prefill must initialize a KV store')
            relative_positions=position_ids
            lengths=None
            offsets=None
            if self.packed_text_positions and text_mask is not None:
                lengths=text_mask.sum(dim=1).long().unsqueeze(1)
                # Right PAD has start0; left PAD has a common RoPE translation.
                # Place each primitive immediately after the last valid word,
                # preserving both padding conventions used by the latest driver.
                offsets=lengths+text_mask.long().argmax(dim=1,keepdim=True)
                if relative_positions is None:
                    relative_positions=torch.arange(tokens.shape[1],device=tokens.device)[None].expand(tokens.shape[0],-1)
                # The original forward adds the padded text width. Cancel that
                # gap so each lane's RoPE matches its unpadded standalone run.
                relative_positions=relative_positions+offsets-text_mask.shape[1]
            logits,termination,past=self.model(tokens,position_ids=relative_positions,use_cache=True,
                return_termination=True,text_ids=text_ids,text_mask=text_mask)
            b=tokens.shape[0];n=past[0][0].shape[2];capacity=self.model.max_context_len
            if self.packed_text_positions:capacity+=self.model.max_text_tokens
            if (self.cache is None or self.cache.keys[0].shape[0]!=b
                    or self.cache.keys[0].dtype!=past[0][0].dtype
                    or self.cache.keys[0].device!=tokens.device):
                # A captured graph owns specific buffer addresses; never replay
                # it against a replacement cache after a batch/device/dtype change.
                self.graph=None;self.graph_inputs=None;self.graph_outputs=None
                shape=(b,self.model.n_head,capacity,self.model.d_model//self.model.n_head)
                self.cache=KVStore([torch.zeros(shape,device=tokens.device,dtype=past[0][0].dtype) for _ in past],
                    [torch.zeros(shape,device=tokens.device,dtype=past[0][1].dtype) for _ in past],
                    torch.ones((b,capacity),dtype=torch.bool,device=tokens.device),
                    torch.arange(capacity,device=tokens.device),
                    torch.zeros(1,dtype=torch.long,device=tokens.device))
            cache=self.cache;cache.length=n;cache.text_width=0 if text_mask is None else text_mask.shape[1]
            cache.text_lengths=lengths
            cache.position_offsets=offsets
            cache.valid.fill_(True)
            if text_mask is not None:cache.valid[:,:cache.text_width].copy_(text_mask)
            for i,(k,v) in enumerate(past):
                cache.keys[i].zero_();cache.values[i].zero_()
                cache.keys[i][:,:,:n].copy_(k);cache.values[i][:,:,:n].copy_(v)
            if return_termination:return logits,termination,cache
            return logits,cache
        cache=past_key_values
        if cache is not self.cache or tokens.shape[1]!=1:raise ValueError('Expected one query and this model KV store')
        if cache.length>=cache.keys[0].shape[2]:raise ValueError('KV capacity exceeded')
        cache.write_index.fill_(cache.length)
        if position_ids is None:
            absolute=torch.full((tokens.shape[0],1),cache.length,device=tokens.device,dtype=torch.long)
            if cache.position_offsets is not None:absolute=absolute-cache.text_width+cache.position_offsets
        else:absolute=position_ids+(cache.position_offsets if cache.position_offsets is not None else cache.text_width)
        args=(tokens,absolute,cache.keys,cache.values,cache.valid,cache.positions,cache.write_index,
              None if self.fixed_capacity else cache.length+1)
        if self.graph is not None:
            self.graph_inputs[0].copy_(tokens);self.graph_inputs[1].copy_(absolute)
            self.graph.replay();logits,termination=self.graph_outputs
        else:
            fn=self.compiled if self.compiled is not None else self._decode
            logits,termination=fn(*args)
        if use_cache:
            cache.length+=1
            if return_termination:return logits,termination,cache
            return logits,cache
        if return_termination:return logits,termination
        return logits

    def capture_graph(self,tokens,position_ids):
        if self.compiled is None:raise ValueError('Compile and warm up first')
        c=self.cache;c.write_index.fill_(c.length)
        absolute=position_ids+(c.position_offsets if c.position_offsets is not None else c.text_width)
        self.graph_inputs=(tokens.clone(),absolute.clone())
        args=(*self.graph_inputs,c.keys,c.values,c.valid,c.positions,c.write_index,None)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.compiled(*args)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):self.graph_outputs=self.compiled(*args)
        self.graph=graph


def renyi_entropy_and_support(logits,alpha,legal_mask=None):
    """Alpha1 is the Shannon limit; keep the existing alpha>1 arithmetic."""
    if not math.isfinite(alpha) or alpha<1:
        raise ValueError('Renyi alpha must be finite and >=1')
    logits=logits.float()
    if legal_mask is not None:
        logits=logits.masked_fill(~legal_mask,float('-inf'))
    if alpha==1:
        # entr(0)=0 handles underflowed probabilities without0*log(0) NaNs.
        entropy=torch.special.entr(torch.softmax(logits,dim=-1)).sum(dim=-1)
    else:
        log_z=torch.logsumexp(logits,dim=-1)
        log_za=torch.logsumexp(alpha*logits,dim=-1)
        entropy=(log_za-alpha*log_z)/(1.0-alpha)
    if legal_mask is None:
        entropy=entropy.clamp(0.0,math.log(logits.shape[-1]))
        support=torch.floor(torch.exp(entropy)+.5).long().clamp(1,logits.shape[-1])
    else:
        legal_count=legal_mask.sum(-1).clamp_min(1)
        entropy=torch.minimum(entropy.clamp_min(0),legal_count.float().log())
        support=torch.minimum(torch.floor(torch.exp(entropy)+.5).long().clamp_min(1),legal_count)
    return entropy,support


class RenyiGPUSampler:
    def __init__(self,device,alpha=9.0):
        if not math.isfinite(alpha) or alpha<1:raise ValueError('Renyi alpha must be finite and >=1')
        self.alpha=alpha
        self.nonfinite=torch.zeros((),device=device,dtype=torch.bool)
        self.audit_enabled=os.environ.get('PAINTER_REGION_ENTROPY_AUDIT','0')=='1'
        self.audit=torch.zeros((3,6),device=device,dtype=torch.float64)
    def reset(self):
        self.nonfinite.zero_();self.audit.zero_()
    def __call__(self,field_logits,candidate_ids,legal_mask=None):
        logits=field_logits.float()
        if legal_mask is None:
            finite=torch.isfinite(logits).all(dim=-1)
        else:
            has_legal=legal_mask.any(-1)
            finite=torch.isfinite(logits).all(-1)&has_legal
            # An invalid empty mask is recorded and rejected by check(); keep
            # interim calculations defined without synchronizing every field.
            first=torch.arange(logits.shape[-1],device=logits.device)==0
            legal_mask=legal_mask|((~has_legal)[:,None]&first[None])
        self.nonfinite.logical_or_(~finite.all())
        # Never silently deliver an invalid batch: check() raises before saving.
        logits=torch.where(finite[:,None],logits,torch.zeros_like(logits))
        entropy,support=renyi_entropy_and_support(logits,self.alpha,legal_mask)
        if legal_mask is not None and self.audit_enabled:
            full_entropy,full_support=renyi_entropy_and_support(logits,self.alpha)
            outside=(torch.softmax(logits,-1)*(~legal_mask)).sum(-1)
            counts=legal_mask.sum(-1)
            stage=torch.where(counts==1023,0,torch.where(counts==255,1,2))
            values=torch.stack((torch.ones_like(outside),outside,full_entropy,entropy,
                                full_support.float(),support.float()),-1).double()
            self.audit.scatter_add_(0,stage[:,None].expand(-1,6),values)
        # Fixed output shape: no .item() to choose topk size. Same uniform top-N law.
        selection_logits=logits if legal_mask is None else logits.masked_fill(~legal_mask,float('-inf'))
        _,order=torch.topk(selection_logits,k=field_logits.shape[-1],dim=-1)
        uniform=(torch.rand(support.shape,device=logits.device)*support).floor().long()
        chosen=order.gather(1,uniform[:,None]).squeeze(1)
        return candidate_ids[chosen]
    def check(self):
        if bool(self.nonfinite.item()):raise ValueError('Nonfinite logits detected; refuse to save batch')
    def audit_stats(self):
        if not self.audit_enabled:return None
        result={}
        for name,values in zip(('global','coarse','fine'),self.audit.cpu().tolist()):
            n=values[0]
            result[name]=dict(coordinate_draws=int(n),mean_outside_probability=values[1]/n if n else None,
                mean_full_entropy=values[2]/n if n else None,mean_legal_entropy=values[3]/n if n else None,
                mean_full_support=values[4]/n if n else None,mean_legal_support=values[5]/n if n else None)
        return result


