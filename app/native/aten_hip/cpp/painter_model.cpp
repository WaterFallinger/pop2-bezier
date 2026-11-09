#include "painter_model.h"
#include <ATen/ops/_scaled_dot_product_efficient_attention.h>
#include <ATen/ops/rms_norm.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace pp {
namespace {
constexpr int64_t kEnd = 7066, kInterrupt = 7067, kBos = 7068;
bool same_prefix(const Prefix& a, const Prefix& b) {
  return a.text_ids==b.text_ids && a.records==b.records &&
      a.text_valid==b.text_valid && a.position_offset==b.position_offset;
}
at::Tensor additive_bias(const at::Tensor& allowed, int64_t heads,
    at::ScalarType dtype) {
  const auto b=allowed.size(0), q=allowed.size(2), n=allowed.size(3);
  // Memory-efficient SDPA requires its bias row stride to be aligned even
  // when the current causal prefix has a non-multiple-of-eight length.
  const auto aligned=((n+7)/8)*8;
  auto backing=at::zeros({b,heads,q,aligned},allowed.options().dtype(dtype));
  auto bias=backing.narrow(3,0,n);
  bias.masked_fill_(allowed.logical_not(),-std::numeric_limits<float>::infinity());
  return bias;
}
}

Scope Scope::field(int64_t field) {
  TORCH_CHECK(field>=0 && field<9,"attribute field must be 0..8");
  return Scope{kOffsets[field],kBins[field],false};
}
Scope Scope::next() { return Scope{0,kBins[0],true}; }
Scope Scope::all() { return Scope{0,kAttributeVocab,true}; }

Model::Model(const StateDict& state,at::Device device,at::ScalarType dtype)
    :device_(device),dtype_(dtype) {
  TORCH_CHECK(dtype==at::kBFloat16 || dtype==at::kFloat,"unsupported model activation dtype");
  const auto read=[&](const std::string& name) {
    const auto it=state.find(name);
    TORCH_CHECK(it!=state.end(),"missing model tensor: ",name);
    TORCH_CHECK(it->second.defined() && it->second.is_floating_point(),"invalid model tensor: ",name);
    return it->second.detach().to(device_,dtype_).contiguous();
  };
  embedding_=read("wte.weight"); text_embedding_=read("text_wte.weight");
  TORCH_CHECK(embedding_.dim()==2 && embedding_.size(0)==kPrimitiveVocab,
      "primitive vocabulary must be 7069");
  architecture_.width=embedding_.size(1);
  TORCH_CHECK(text_embedding_.sizes()==at::IntArrayRef({kTextVocab,architecture_.width}),
      "text vocabulary must be 1500 and use the model width");
  const auto qnorm=read("h.0.attn.q_norm.weight");
  architecture_.head_width=qnorm.numel();
  TORCH_CHECK(architecture_.head_width>0 && architecture_.head_width%2==0 &&
      architecture_.width%architecture_.head_width==0,"invalid model head geometry");
  architecture_.heads=architecture_.width/architecture_.head_width;
  const auto gate=read("h.0.mlp.gate_up.weight");
  TORCH_CHECK(gate.dim()==2 && gate.size(0)%2==0,"invalid SwiGLU matrix");
  architecture_.ffn_hidden=gate.size(0)/2;
  for(int64_t layer=0;state.count("h."+std::to_string(layer)+".attn.c_attn.weight");++layer) {
    const auto key=[&](const char* suffix){return "h."+std::to_string(layer)+"."+suffix;};
    Block block{read(key("norm_1.weight")),read(key("attn.c_attn.weight")),
      read(key("attn.q_norm.weight")),read(key("attn.k_norm.weight")),read(key("attn.c_proj.weight")),
      read(key("norm_2.weight")),read(key("mlp.gate_up.weight")),read(key("mlp.down.weight"))};
    const auto w=architecture_.width,h=architecture_.ffn_hidden,d=architecture_.head_width;
    TORCH_CHECK(block.norm1.numel()==w && block.norm2.numel()==w &&
        block.q_norm.numel()==d && block.k_norm.numel()==d &&
        block.qkv.sizes()==at::IntArrayRef({3*w,w}) &&
        block.projection.sizes()==at::IntArrayRef({w,w}) &&
        block.gate_up.sizes()==at::IntArrayRef({2*h,w}) &&
        block.down.sizes()==at::IntArrayRef({w,h}),"layer geometry mismatch: ",layer);
    blocks_.push_back(std::move(block));
  }
  architecture_.layers=static_cast<int64_t>(blocks_.size());
  TORCH_CHECK(architecture_.layers>0,"checkpoint contains no transformer layers");
  final_norm_=read("final_norm.weight");
  termination_weight_=read("termination_head.weight");termination_bias_=read("termination_head.bias");
  TORCH_CHECK(final_norm_.numel()==architecture_.width &&
      termination_weight_.sizes()==at::IntArrayRef({2,architecture_.width}) &&
      termination_bias_.sizes()==at::IntArrayRef({2}),"output head geometry mismatch");
  const auto f32=embedding_.options().dtype(at::kFloat);
  auto powers=at::arange(0,architecture_.head_width,2,f32)/static_cast<double>(architecture_.head_width);
  inverse_frequency_=at::pow(10000.,powers).reciprocal();
}

Cache Model::make_cache(int64_t batch,int64_t capacity) const {
  TORCH_CHECK(batch>0 && capacity>0 && capacity<=kContext,"invalid cache dimensions");
  // Tensor/storage representability is the only row-count constraint here.
  // Real memory exhaustion is returned by ATen instead of reducing the batch.
  int64_t elements=batch;
  for(const auto dimension:{architecture_.heads,capacity,architecture_.head_width}){
    TORCH_CHECK(dimension>0&&elements<=std::numeric_limits<int64_t>::max()/dimension,"cache tensor element count overflow");
    elements*=dimension;
  }
  TORCH_CHECK(uint64_t(elements)<=uint64_t(std::numeric_limits<ptrdiff_t>::max())/embedding_.element_size(),"cache tensor storage size overflow");
  Cache result;result.batch=batch;result.capacity=capacity;result.int8=int8_kv_enabled_;
  const auto options=embedding_.options();
  if(result.int8){
    result.int8_kernels=int8_kernels_;
    result.packed_int8=at::zeros({architecture_.layers,2,batch,architecture_.heads,capacity,architecture_.head_width},options.dtype(at::kChar));
    result.packed_scales=at::ones({architecture_.layers,2,batch,architecture_.heads,capacity},options.dtype(at::kFloat));
    result.packed_scratch=at::empty({architecture_.layers,2,batch,architecture_.heads,1,architecture_.head_width},options);
  }
  for(int64_t i=0;i<architecture_.layers;++i) {
    if(result.int8){
      result.keys.push_back(result.packed_int8[i][0]);result.values.push_back(result.packed_int8[i][1]);
      result.key_scales.push_back(result.packed_scales[i][0]);result.value_scales.push_back(result.packed_scales[i][1]);
      result.scratch_keys.push_back(result.packed_scratch[i][0]);result.scratch_values.push_back(result.packed_scratch[i][1]);
    }else{
      result.keys.push_back(at::zeros({batch,architecture_.heads,capacity,architecture_.head_width},options));
      result.values.push_back(at::zeros_like(result.keys.back()));
    }
  }
  result.valid=at::zeros({batch,capacity},options.dtype(at::kBool));
  result.positions=at::arange(capacity,options.dtype(at::kLong));
  result.lengths=at::zeros({batch},options.dtype(at::kLong));
  result.absolute_positions=at::zeros_like(result.lengths);
  return result;
}
void Model::prepare_int8_kv(){
  int8_kernels_->prepare(device_,architecture_.head_width);
  int8_kernels_->validate(device_);
}
void Model::configure_int8_kv(bool enabled){TORCH_CHECK(!enabled||int8_kernels_->ready(),"INT8 KV kernel was not prepared");int8_kv_enabled_=enabled;}

void Model::validate_cache(const Cache& cache) const {
  TORCH_CHECK(cache.batch>0 && cache.capacity>0 && cache.capacity<=kContext &&
      cache.keys.size()==blocks_.size() && cache.values.size()==blocks_.size(),"cache/model layout mismatch");
  TORCH_CHECK(cache.lengths.device()==device_ && cache.lengths.scalar_type()==at::kLong &&
      cache.lengths.sizes()==at::IntArrayRef({cache.batch}) &&
      cache.absolute_positions.device()==device_ && cache.absolute_positions.scalar_type()==at::kLong &&
      cache.absolute_positions.sizes()==at::IntArrayRef({cache.batch}) &&
      cache.positions.device()==device_ && cache.positions.scalar_type()==at::kLong &&
      cache.positions.sizes()==at::IntArrayRef({cache.capacity}) &&
      cache.valid.device()==device_ && cache.valid.scalar_type()==at::kBool &&
      cache.valid.sizes()==at::IntArrayRef({cache.batch,cache.capacity}),"invalid cache metadata tensors");
  for(size_t i=0;i<blocks_.size();++i) {
    const std::array<int64_t,4> expected{cache.batch,architecture_.heads,cache.capacity,architecture_.head_width};
    TORCH_CHECK(cache.keys[i].device()==device_ && cache.values[i].device()==device_ &&
        cache.keys[i].scalar_type()==(cache.int8?at::kChar:dtype_) && cache.values[i].scalar_type()==(cache.int8?at::kChar:dtype_) &&
        cache.keys[i].sizes()==at::IntArrayRef(expected) && cache.values[i].sizes()==at::IntArrayRef(expected),"invalid KV layer ",i);
    if(cache.int8){
      TORCH_CHECK(cache.key_scales.size()==blocks_.size()&&cache.value_scales.size()==blocks_.size()
        &&cache.scratch_keys.size()==blocks_.size()&&cache.scratch_values.size()==blocks_.size(),"INT8 KV scale/scratch count differs");
      const std::array<int64_t,3> scale_shape{cache.batch,architecture_.heads,cache.capacity};
      const std::array<int64_t,4> scratch_shape{cache.batch,architecture_.heads,1,architecture_.head_width};
      for(const auto* scale:{&cache.key_scales[i],&cache.value_scales[i]})TORCH_CHECK(scale->device()==device_&&scale->scalar_type()==at::kFloat&&scale->sizes()==at::IntArrayRef(scale_shape),"INT8 KV scale shape differs");
      for(const auto* scratch:{&cache.scratch_keys[i],&cache.scratch_values[i]})TORCH_CHECK(scratch->device()==device_&&scratch->scalar_type()==dtype_&&scratch->sizes()==at::IntArrayRef(scratch_shape),"INT8 KV scratch shape differs");
    }
  }
}

at::Tensor Model::primitive_embedding(const at::Tensor& records) const {
  TORCH_CHECK(records.dim()>=2 && records.size(-1)==9 && records.scalar_type()==at::kLong &&
      records.device()==device_,"primitive tokens must be same-device Long[...,9]");
  auto shape=records.sizes().vec();shape.back()=9;shape.push_back(architecture_.width);
  auto safe=records.clamp(0,kPrimitiveVocab-1);
  auto encoded=embedding_.index_select(0,safe.reshape({-1})).reshape(shape).sum(-2);
  auto first=safe.select(-1,0);
  auto controls=first.eq(kBos).logical_or(first.eq(kEnd)).logical_or(first.eq(kInterrupt));
  auto first_shape=first.sizes().vec();first_shape.push_back(architecture_.width);
  auto control_embedding=embedding_.index_select(0,first.reshape({-1})).reshape(first_shape);
  auto result=at::where(controls.unsqueeze(-1),control_embedding,encoded);
  auto legal=records.ge(0).logical_and(records.lt(kPrimitiveVocab)).all(-1,true);
  return at::where(legal,result,at::full_like(result,std::numeric_limits<float>::quiet_NaN()));
}

at::Tensor Model::rms(const at::Tensor& x,const at::Tensor& weight) const {
  // ATen's native RMSNorm retains BF16 activations and accumulates statistics
  // in FP32, matching the original model's GPU forward rather than autocast.
  return at::rms_norm(x,{x.size(-1)},weight,1e-6);
}

std::pair<at::Tensor,at::Tensor> Model::rope_factors(const at::Tensor& absolute) const {
  auto angles=absolute.to(at::kFloat).unsqueeze(-1)*inverse_frequency_;
  return {angles.cos().unsqueeze(-2).to(dtype_),angles.sin().unsqueeze(-2).to(dtype_)};
}

std::pair<at::Tensor,at::Tensor> Model::rope(const at::Tensor& q,const at::Tensor& k,
    const at::Tensor& cosine,const at::Tensor& sine) const {
  const auto rotate=[&](const at::Tensor& x) {
    auto even=x.slice(-1,0,architecture_.head_width,2);
    auto odd=x.slice(-1,1,architecture_.head_width,2);
    return at::stack({even*cosine-odd*sine,even*sine+odd*cosine},-1).flatten(-2);
  };
  return {rotate(q),rotate(k)};
}

at::Tensor Model::attention(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const at::Tensor& bias,AttentionMode mode) const {
  if(mode==AttentionMode::Efficient) {
    TORCH_CHECK(device_.is_cuda(),"efficient attention requires the selected native HIP device");
    return std::get<0>(at::_scaled_dot_product_efficient_attention(q,k,v,bias,false,0.,false,std::nullopt));
  }
  auto score=q.to(at::kFloat).matmul(k.to(at::kFloat).transpose(-2,-1)) /
      std::sqrt(static_cast<double>(architecture_.head_width));
  score=score+bias.to(at::kFloat);
  auto has_key=at::isfinite(bias).any(-1,true);
  auto safe=at::where(has_key,score,at::zeros_like(score));
  auto output=safe.softmax(-1).matmul(v.to(at::kFloat));
  return at::where(has_key,output,at::zeros_like(output)).to(dtype_);
}

at::Tensor Model::feed_forward(const at::Tensor& x,const Block& block) const {
  auto gate_up=at::linear(rms(x,block.norm2),block.gate_up);
  auto gate=gate_up.narrow(-1,0,architecture_.ffn_hidden);
  auto up=gate_up.narrow(-1,architecture_.ffn_hidden,architecture_.ffn_hidden);
  return at::linear(at::silu(gate)*up,block.down);
}

Prediction Model::project(const at::Tensor& hidden,Scope scope) const {
  TORCH_CHECK(scope.offset>=0 && scope.count>0 && scope.offset+scope.count<=kAttributeVocab,
      "projected attribute head range is invalid");
  auto flattened=hidden.reshape({-1,architecture_.width});
  Prediction result;result.attribute_offset=scope.offset;
  result.attributes=at::linear(flattened,embedding_.narrow(0,scope.offset,scope.count)).to(at::kFloat);
  if(scope.include_termination)
    result.termination=at::linear(flattened,termination_weight_,termination_bias_).to(at::kFloat);
  return result;
}

at::Tensor Model::prefill_row(Cache& cache,int64_t row,const Prefix& prefix,
    int64_t chunk_size,AttentionMode mode,int64_t retained) const {
  const auto text_count=static_cast<int64_t>(prefix.text_ids.size());
  const auto record_count=static_cast<int64_t>(prefix.records.size());
  const auto length=retained+text_count+record_count;
  TORCH_CHECK(retained>=0&&text_count+record_count>0 && length<=cache.capacity && prefix.position_offset>=0 &&
      length+prefix.position_offset<=kContext,"prefill exceeds physical or absolute context");
  TORCH_CHECK(prefix.text_valid.empty() || prefix.text_valid.size()==prefix.text_ids.size(),"text mask length mismatch");
  for(auto id:prefix.text_ids)TORCH_CHECK(id>=0 && id<kTextVocab,"text token outside vocabulary");
  for(const auto& record:prefix.records)for(auto id:record)TORCH_CHECK(id>=0 && id<kPrimitiveVocab,"primitive token outside vocabulary");
  auto long_options=embedding_.options().dtype(at::kLong);
  auto cpu_long=at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  at::Tensor text,images;
  if(text_count) {
    auto ids=at::from_blob(const_cast<int64_t*>(prefix.text_ids.data()),{text_count},cpu_long).to(device_);
    text=text_embedding_.index_select(0,ids).unsqueeze(0);
  }
  if(record_count) {
    auto ids=at::from_blob(const_cast<int64_t*>(prefix.records.front().data()),{1,record_count,9},cpu_long).to(device_);
    images=primitive_embedding(ids);
  }
  cache.valid.select(0,row).narrow(0,retained,cache.capacity-retained).zero_();
  cache.valid.select(0,row).narrow(0,retained,length-retained).fill_(true);
  if(!prefix.text_valid.empty() && text_count) {
    auto mask=at::from_blob(const_cast<uint8_t*>(prefix.text_valid.data()),{text_count},at::TensorOptions().dtype(at::kByte)).to(device_,at::kBool);
    cache.valid.select(0,row).narrow(0,retained,text_count).copy_(mask);
  }
  at::Tensor x;
  for(int64_t begin=retained;begin<length;begin+=chunk_size) {
    const auto end=std::min(begin+chunk_size,length),query=end-begin;
    const auto relative_begin=begin-retained,relative_end=end-retained;
    std::vector<at::Tensor> pieces;
    if(relative_begin<text_count)pieces.push_back(text.narrow(1,relative_begin,std::min(relative_end,text_count)-relative_begin));
    if(relative_end>text_count)pieces.push_back(images.narrow(1,std::max(relative_begin,text_count)-text_count,relative_end-std::max(relative_begin,text_count)));
    x=pieces.size()==1?pieces.front():at::cat(pieces,1);
    auto absolute=(at::arange(begin,end,long_options)+prefix.position_offset).unsqueeze(0);
    const auto rotary=rope_factors(absolute);
    auto qpositions=at::arange(begin,end,long_options).unsqueeze(1);
    auto keys=cache.positions.narrow(0,0,end).unsqueeze(0);
    auto allowed=keys.le(qpositions).unsqueeze(0).unsqueeze(0).logical_and(
        cache.valid.narrow(0,row,1).narrow(1,0,end).unsqueeze(1).unsqueeze(1));
    auto bias=additive_bias(allowed,architecture_.heads,mode==AttentionMode::Efficient?dtype_:at::kFloat);
    auto query_has_key=allowed.any(-1,true);
    for(size_t layer=0;layer<blocks_.size();++layer) {
      const auto& block=blocks_[layer];
      auto packed=at::linear(rms(x,block.norm1),block.qkv);
      auto q=rms(packed.narrow(-1,0,architecture_.width).reshape({1,query,architecture_.heads,architecture_.head_width}),block.q_norm);
      auto k=rms(packed.narrow(-1,architecture_.width,architecture_.width).reshape({1,query,architecture_.heads,architecture_.head_width}),block.k_norm);
      auto v=packed.narrow(-1,2*architecture_.width,architecture_.width).reshape({1,query,architecture_.heads,architecture_.head_width}).transpose(1,2);
      auto rotated=rope(q,k,rotary.first,rotary.second);q=rotated.first.transpose(1,2);k=rotated.second.transpose(1,2);
      auto key_row=cache.keys[layer].narrow(0,row,1),value_row=cache.values[layer].narrow(0,row,1);
      at::Tensor y;
      if(cache.int8){
        auto ks=cache.key_scales[layer].narrow(0,row,1),vs=cache.value_scales[layer].narrow(0,row,1);
        // Prefill matches the historical adapter: BF16 current chunk, INT8
        // old chunks. Dequantized old rows are temporary for this one layer.
        auto all_k=begin?at::cat({dequantize_kv_chunk(key_row.narrow(2,0,begin),ks.narrow(2,0,begin),dtype_),k},2):k;
        auto all_v=begin?at::cat({dequantize_kv_chunk(value_row.narrow(2,0,begin),vs.narrow(2,0,begin),dtype_),v},2):v;
        y=attention(q,all_k,all_v,bias,mode);
        quantize_kv_chunk(k,key_row.narrow(2,begin,query),ks.narrow(2,begin,query));
        quantize_kv_chunk(v,value_row.narrow(2,begin,query),vs.narrow(2,begin,query));
      }else{
        key_row.narrow(2,begin,query).copy_(k);value_row.narrow(2,begin,query).copy_(v);
        y=attention(q,key_row.narrow(2,0,end),value_row.narrow(2,0,end),bias,mode);
      }
      // Explicit all-masked PAD queries cannot seed NaNs into later valid KV.
      y=at::where(query_has_key,y,at::zeros_like(y));
      x=x+at::linear(y.transpose(1,2).contiguous().reshape({1,query,architecture_.width}),block.projection);
      x=x+feed_forward(x,block);
    }
  }
  return rms(x.narrow(1,x.size(1)-1,1),final_norm_);
}
Prediction Model::append_prefix(Cache& cache,int64_t retained,const Prefix& suffix,int64_t chunk_size,AttentionMode mode)const{
  validate_cache(cache);TORCH_CHECK(cache.batch==1&&suffix.text_ids.empty()&&!suffix.records.empty(),"incremental prefill expects one visual suffix row");
  const auto hidden=prefill_row(cache,0,suffix,chunk_size,mode,retained);
  const auto length=retained+int64_t(suffix.records.size());
  cache.lengths.fill_(length);cache.absolute_positions.fill_(length+suffix.position_offset);
  return project(hidden,Scope::next());
}

Prediction Model::prefill_rows(Cache& cache,const std::vector<Prefix>& prefixes,
    Scope scope,int64_t chunk_size,AttentionMode mode) const {
  validate_cache(cache);
  TORCH_CHECK(prefixes.size()==static_cast<size_t>(cache.batch) && chunk_size>0 && chunk_size<=kContext,
      "prefill row or chunk count invalid");
  std::vector<at::Tensor> hidden;
  std::vector<int64_t> lengths,absolute;
  for(int64_t row=0;row<cache.batch;++row) {
    const auto& prefix=prefixes[row];
    const auto length=static_cast<int64_t>(prefix.text_ids.size()+prefix.records.size());
    int64_t shared=-1;
    for(int64_t previous=0;previous<row;++previous)if(same_prefix(prefix,prefixes[previous])){shared=previous;break;}
    if(shared>=0) {
      for(size_t layer=0;layer<blocks_.size();++layer) {
        cache.keys[layer].narrow(0,row,1).narrow(2,0,length).copy_(cache.keys[layer].narrow(0,shared,1).narrow(2,0,length));
        cache.values[layer].narrow(0,row,1).narrow(2,0,length).copy_(cache.values[layer].narrow(0,shared,1).narrow(2,0,length));
        if(cache.int8){
          cache.key_scales[layer].narrow(0,row,1).narrow(2,0,length).copy_(cache.key_scales[layer].narrow(0,shared,1).narrow(2,0,length));
          cache.value_scales[layer].narrow(0,row,1).narrow(2,0,length).copy_(cache.value_scales[layer].narrow(0,shared,1).narrow(2,0,length));
        }
      }
      cache.valid.select(0,row).copy_(cache.valid.select(0,shared));
      hidden.push_back(hidden[shared]);
    }else hidden.push_back(prefill_row(cache,row,prefix,chunk_size,mode));
    lengths.push_back(length);absolute.push_back(length+prefix.position_offset);
  }
  const auto cpu=at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  cache.lengths.copy_(at::from_blob(lengths.data(),{cache.batch},cpu));
  cache.absolute_positions.copy_(at::from_blob(absolute.data(),{cache.batch},cpu));
  return project(at::cat(hidden,0),scope);
}

Prediction Model::decode(const at::Tensor& records,Cache& cache,Scope scope,
    int64_t bucket_end,AttentionMode mode) const {
  validate_cache(cache);
  TORCH_CHECK(records.device()==device_ && records.scalar_type()==at::kLong &&
      records.sizes()==at::IntArrayRef({cache.batch,9}),"decode requires same-device Long[B,9]");
  TORCH_CHECK(bucket_end>0 && bucket_end<=cache.capacity,"attention bucket outside cache");
  auto valid_indices=cache.lengths.ge(0).logical_and(cache.lengths.lt(bucket_end))
      .logical_and(cache.absolute_positions.ge(0)).logical_and(cache.absolute_positions.lt(kContext));
  auto indices=cache.lengths.clamp(0,bucket_end-1).reshape({cache.batch,1});
  cache.valid.scatter_(1,indices,at::ones_like(indices,indices.options().dtype(at::kBool)));
  auto write=indices.reshape({cache.batch,1,1,1}).expand({cache.batch,architecture_.heads,1,architecture_.head_width});
  auto absolute=cache.absolute_positions.clamp(0,kContext-1).reshape({cache.batch,1});
  const auto rotary=rope_factors(absolute);
  auto allowed=cache.positions.narrow(0,0,bucket_end).reshape({1,1,1,bucket_end})
      .le(cache.lengths.reshape({cache.batch,1,1,1})).logical_and(
          cache.valid.narrow(1,0,bucket_end).reshape({cache.batch,1,1,bucket_end}));
  auto bias=additive_bias(allowed,architecture_.heads,mode==AttentionMode::Efficient?dtype_:at::kFloat);
  auto x=primitive_embedding(records.unsqueeze(1));
  for(size_t layer=0;layer<blocks_.size();++layer) {
    const auto& block=blocks_[layer];
    auto packed=at::linear(rms(x,block.norm1),block.qkv);
    auto q=rms(packed.narrow(-1,0,architecture_.width).reshape({cache.batch,1,architecture_.heads,architecture_.head_width}),block.q_norm);
    auto k=rms(packed.narrow(-1,architecture_.width,architecture_.width).reshape({cache.batch,1,architecture_.heads,architecture_.head_width}),block.k_norm);
    auto v=packed.narrow(-1,2*architecture_.width,architecture_.width).reshape({cache.batch,1,architecture_.heads,architecture_.head_width}).transpose(1,2);
    auto rotated=rope(q,k,rotary.first,rotary.second);q=rotated.first.transpose(1,2);k=rotated.second.transpose(1,2);
    at::Tensor y;
    if(cache.int8){
      cache.scratch_keys[layer].copy_(k);cache.scratch_values[layer].copy_(v);
      y=int8_kernels_->attention(q,cache.keys[layer],cache.values[layer],cache.key_scales[layer],cache.value_scales[layer],
        cache.scratch_keys[layer],cache.scratch_values[layer],cache.valid,cache.lengths,bucket_end);
    }else{
      cache.keys[layer].scatter_(2,write,k);cache.values[layer].scatter_(2,write,v);
      y=attention(q,cache.keys[layer].narrow(2,0,bucket_end),cache.values[layer].narrow(2,0,bucket_end),bias,mode);
    }
    x=x+at::linear(y.transpose(1,2).contiguous().reshape({cache.batch,1,architecture_.width}),block.projection);
    x=x+feed_forward(x,block);
  }
  auto result=project(rms(x,final_norm_),scope);
  auto valid=valid_indices.unsqueeze(1);
  result.attributes=at::where(valid,result.attributes,at::full_like(result.attributes,std::numeric_limits<float>::quiet_NaN()));
  if(result.termination.defined())result.termination=at::where(valid,result.termination,at::full_like(result.termination,std::numeric_limits<float>::quiet_NaN()));
  return result;
}

void Model::advance_cache(Cache& cache) {
  if(cache.int8){
    TORCH_CHECK(cache.int8_kernels,"INT8 KV fused-commit kernel missing");
    cache.int8_kernels->commit(cache.packed_int8,cache.packed_scales,cache.packed_scratch,cache.lengths);
  }
  cache.lengths.add_(1);cache.absolute_positions.add_(1);
}
} // namespace pp
