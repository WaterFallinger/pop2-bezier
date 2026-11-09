"""Shared weights, separate batch-row KV histories, raw-logit CFG before sampling."""
import torch

class GuidedModel:
    def __init__(self,model,scale=2.0):
        self.model=model;self.scale=scale;self.paired_mask=None
        self.calls=0;self.cache_updates=0

    def combine(self,logits,batch):
        conditional,unconditional=logits[:batch].float(),logits[batch:].float()
        scale=torch.as_tensor(self.scale,device=conditional.device,dtype=torch.float32)
        if scale.ndim==1:
            if scale.numel()!=batch:raise ValueError('OneCFGscale perlogical candidate required')
            scale=scale.reshape((batch,)+(1,)*(conditional.ndim-1))
        elif scale.ndim!=0:raise ValueError('CFGscale must be scalar or batchvector')
        result=unconditional+scale*(conditional-unconditional)
        # Preserve exact baseline logits at s1, not just algebraic equivalence.
        result=torch.where(scale==1,conditional,result)
        return torch.where(scale==0,unconditional,result)

    def __call__(self,tokens,**kwargs):
        batch=tokens.size(0)
        cache=kwargs.get('past_key_values')
        text_ids=kwargs.get('text_ids')
        mask=kwargs.get('text_mask')
        if cache is None:
            if text_ids is None or mask is None or text_ids.size(1)<2:
                raise ValueError('CFG needs the conditional text prefix on first forward')
            # Match training's no-text representation. LeftPAD is fully masked;
            # relative RoPE positions match a standalone [BOS,EOS] prefix.
            unconditional_ids=torch.zeros_like(text_ids)
            unconditional_mask=torch.zeros_like(mask)
            unconditional_ids[:,-2]=2;unconditional_ids[:,-1]=3
            unconditional_mask[:,-2:]=True
            self.paired_mask=torch.cat((mask,unconditional_mask),0)
            kwargs['text_ids']=torch.cat((text_ids,unconditional_ids),0)
        elif text_ids is not None:
            raise ValueError('Text must not be reinserted into cached CFG histories')
        if self.paired_mask is None or self.paired_mask.size(0)!=2*batch:
            raise ValueError('Mismatched CFG cache batch')
        kwargs['text_mask']=self.paired_mask
        if kwargs.get('position_ids') is not None:
            kwargs['position_ids']=torch.cat((kwargs['position_ids'],kwargs['position_ids']),0)
        # Only one sampled graphic stream: both branch rows get identical
        # committed records or identical partial attribute queries.
        raw=self.model(torch.cat((tokens,tokens),0),**kwargs)
        self.calls+=1
        use_cache=kwargs.get('use_cache',False)
        termination=kwargs.get('return_termination',False)
        if use_cache:self.cache_updates+=1
        if use_cache and termination:
            attrs,term,new_cache=raw
            return self.combine(attrs,batch),self.combine(term,batch),new_cache
        if use_cache:
            attrs,new_cache=raw;return self.combine(attrs,batch),new_cache
        if termination:
            attrs,term=raw;return self.combine(attrs,batch),self.combine(term,batch)
        return self.combine(raw,batch)
