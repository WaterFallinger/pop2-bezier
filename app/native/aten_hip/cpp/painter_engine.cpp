#include "painter_engine.h"
#include "checkpoint_loader.h"
#include <ATen/ops/_scaled_dot_product_efficient_attention.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/StreamGuard.h>
#include <torch/version.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#ifdef _WIN32
#include <windows.h>
#endif
#if __has_include("build_id.h")
#include "build_id.h"
#else
#define PP_ATEN_BUILD_ID "unversioned"
#endif

namespace pp {
namespace {
std::atomic<uint64_t> next_id{1};
void trace(const char* phase){if(std::getenv("PAINTER_ATEN_TRACE")){std::fprintf(stderr,"ATEN_PHASE %s\n",phase);std::fflush(stderr);}}
std::string module_path(const wchar_t* name){
#ifdef _WIN32
  const auto module=GetModuleHandleW(name);if(!module)return {};
  std::wstring path(32768,L'\0');const auto n=GetModuleFileNameW(module,path.data(),DWORD(path.size()));
  if(!n||n>=path.size())return {};path.resize(n);
  const auto bytes=WideCharToMultiByte(CP_UTF8,0,path.data(),int(path.size()),nullptr,0,nullptr,nullptr);
  std::string utf8(bytes,'\0');WideCharToMultiByte(CP_UTF8,0,path.data(),int(path.size()),utf8.data(),bytes,nullptr,nullptr);return utf8;
#else
  return {};
#endif
}
DeviceSelection choose_device(uint32_t requested) {
  trace("choose_device_count");
  int count=0;TORCH_CHECK(hipGetDeviceCount(&count)==hipSuccess&&count>0,"no HIP model device; no CPU/CUDA substitution");
  trace("choose_device_count_done");
  DeviceSelection result;result.count=count;result.automatic=requested==std::numeric_limits<uint32_t>::max();
  struct Candidate{int ordinal;hipDeviceProp_t p;};std::vector<Candidate> eligible;std::ostringstream list;list<<'[';
  for(int ordinal=0;ordinal<count;++ordinal){hipDeviceProp_t p{};const auto status=hipGetDeviceProperties(&p,ordinal);
    if(ordinal)list<<',';list<<"{\"ordinal\":"<<ordinal<<",\"properties_status\":"<<int(status);
    if(status==hipSuccess){const bool amd=std::string(p.gcnArchName).rfind("gfx",0)==0;
      list<<",\"name\":"<<json_quote(p.name)<<",\"architecture\":"<<json_quote(p.gcnArchName)
        <<",\"integrated\":"<<(p.integrated?"true":"false")<<",\"vram_bytes\":"<<uint64_t(p.totalGlobalMem)
        <<",\"compute_units\":"<<p.multiProcessorCount<<",\"clock_khz\":"<<p.clockRate<<",\"amd_gfx\":"<<(amd?"true":"false");
      if(amd)eligible.push_back({ordinal,p});}
    list<<'}';
  }
  list<<']';result.candidates_json=list.str();
  if(!result.automatic){TORCH_CHECK(requested<uint32_t(count),"HIP model device ordinal out of range");
    TORCH_CHECK(std::any_of(eligible.begin(),eligible.end(),[&](const auto& c){return c.ordinal==int(requested);}),"explicit HIP device did not report usable AMD gfx properties");
    result.ordinal=int(requested);result.reason="explicit HIP ordinal; no substitution";return result;}
  TORCH_CHECK(!eligible.empty(),"no HIP device reported AMD gfx properties");
  std::sort(eligible.begin(),eligible.end(),[](const auto&a,const auto&b){
    if(bool(a.p.integrated)!=bool(b.p.integrated))return !a.p.integrated;
    if(a.p.totalGlobalMem!=b.p.totalGlobalMem)return a.p.totalGlobalMem>b.p.totalGlobalMem;
    if(a.p.multiProcessorCount!=b.p.multiProcessorCount)return a.p.multiProcessorCount>b.p.multiProcessorCount;
    if(a.p.clockRate!=b.p.clockRate)return a.p.clockRate>b.p.clockRate;
    return a.ordinal<b.ordinal;
  });
  result.ordinal=eligible.front().ordinal;
  result.reason="actual HIP properties: discrete first, then VRAM, compute units, clock, lowest ordinal tie-break";
  return result;
}
at::Tensor cpu_long(const std::vector<int64_t>& values) {
  return at::from_blob(const_cast<int64_t*>(values.data()),{int64_t(values.size())},at::TensorOptions().dtype(at::kLong)).clone();
}
at::Tensor long_on(const std::vector<int64_t>& values,at::Device device) {
  return cpu_long(values).to(device);
}
CacheSnapshot leased(CacheSnapshot value){
  auto& leases=value.owner->row_leases;
  if(leases.size()!=size_t(value.owner->cache.batch))leases.resize(size_t(value.owner->cache.batch));
  auto& row=leases.at(size_t(value.row));row.erase(std::remove_if(row.begin(),row.end(),[](const auto& lease){return lease.expired();}),row.end());
  value.lease=std::make_shared<RowLease>();row.push_back(value.lease);return value;
}
bool row_unreferenced(const CacheOwner& owner,size_t row){
  if(row>=owner.row_leases.size())return true;
  return std::all_of(owner.row_leases[row].begin(),owner.row_leases[row].end(),[](const auto& lease){return lease.expired();});
}
void copy_prefix_row(Cache& destination,int64_t row,const Cache& source,int64_t source_row,int64_t length){
  TORCH_CHECK(destination.int8==source.int8&&length>=0&&length<=source.capacity&&length<=destination.capacity,"KV prefix copy precision/length mismatch");
  for(size_t layer=0;layer<destination.keys.size();++layer){
    destination.keys[layer].select(0,row).narrow(1,0,length).copy_(source.keys[layer].select(0,source_row).narrow(1,0,length));
    destination.values[layer].select(0,row).narrow(1,0,length).copy_(source.values[layer].select(0,source_row).narrow(1,0,length));
    if(destination.int8){
      destination.key_scales[layer].select(0,row).narrow(1,0,length).copy_(source.key_scales[layer].select(0,source_row).narrow(1,0,length));
      destination.value_scales[layer].select(0,row).narrow(1,0,length).copy_(source.value_scales[layer].select(0,source_row).narrow(1,0,length));
    }
  }
  destination.valid.select(0,row).zero_();destination.valid.select(0,row).narrow(0,0,length).copy_(source.valid.select(0,source_row).narrow(0,0,length));
}
void validate_prefix(const Prefix& p) {
  TORCH_CHECK(!p.text_ids.empty()&&p.text_ids.size()<=224&&p.text_ids.front()==2&&p.text_ids.back()==3,"text prefix requires original BOS/EOS within224 tokens");
  TORCH_CHECK(!p.records.empty()&&p.position_offset>=0,"empty/invalid primitive prefix");
  TORCH_CHECK(p.position_offset+int64_t(p.text_ids.size()+p.records.size())<kContext,"prefix leaves no complete record slot");
  TORCH_CHECK(p.text_valid.empty()||p.text_valid.size()==p.text_ids.size(),"text mask shape differs");
  for(auto id:p.text_ids)TORCH_CHECK(id>=0&&id<kTextVocab,"text ID outside original vocabulary");
  for(const auto& row:p.records)for(auto id:row)TORCH_CHECK(id>=0&&id<kPrimitiveVocab,"primitive ID outside original vocabulary");
}
void validate_frame(const std::vector<CacheSnapshot>& inputs,const PpAtenFrameOptions& o,
    const uint32_t* bounds,const float* uniforms,uint64_t model_id) {
  TORCH_CHECK(o.struct_size>=sizeof(PpAtenFrameOptions),"invalid native frame options");
  const auto geometry=row_geometry(o.logical_rows,o.guidance);
  TORCH_CHECK(inputs.size()==geometry.physical,"physical cache count differs");
  TORCH_CHECK(bounds&&uniforms&&std::isfinite(o.cfg)&&o.cfg>=0&&std::isfinite(o.alpha)&&o.alpha>1,"invalid frame scalars");
  TORCH_CHECK(o.guidance==2||o.cfg==1.,"one-row guidance requires CFG1");
  if(o.sampling_gpu)TORCH_CHECK(std::isfinite(float(o.cfg))&&std::isfinite(float(o.alpha))&&float(o.alpha)>1,"GPU sampler scalar precision unsupported");
  for(const auto& c:inputs){
    TORCH_CHECK(c.owner&&c.pending&&c.owner->model_id==model_id&&!c.owner->poisoned,"cache is missing, belongs to another model or has uncertain state");
    TORCH_CHECK(c.row>=0&&c.row<c.owner->cache.batch&&c.pending_row>=0&&c.pending_row<c.pending->attributes.size(0),"cache row is invalid");
    TORCH_CHECK(c.owner->cache.int8==inputs[0].owner->cache.int8,"mixed INT8/BF16 physical cache rows");
    TORCH_CHECK(c.length>0&&c.length<kContext&&c.absolute>=c.length&&c.absolute<kContext,"cache context capacity exhausted");
    TORCH_CHECK(c.length<=c.owner->current_lengths[c.row],"cache snapshot is newer than its backing owner");
  }
  for(size_t f=0;f<9;++f)for(size_t row=0;row<o.logical_rows;++row){
    const auto i=f*o.logical_rows+row;const auto start=bounds[i*2],count=bounds[i*2+1];
    TORCH_CHECK(count>0&&start<uint32_t(kBins[f])&&uint64_t(start)+count<=uint64_t(kBins[f]),"legal candidate bounds escaped attribute head");
    TORCH_CHECK(std::isfinite(uniforms[i])&&uniforms[i]>=0&&uniforms[i]<1,"uniform must be finite in[0,1)");
  }
}
uint32_t sample_cpu(const float* conditional,const float* unconditional,int64_t choices,
    int64_t start,int64_t count,double cfg,double alpha,float uniform,int64_t offset) {
  double maximum=-std::numeric_limits<double>::infinity();
  std::vector<std::pair<double,int64_t>> ordered;ordered.reserve(count);
  for(int64_t i=0;i<choices;++i){
    TORCH_CHECK(std::isfinite(conditional[i])&&(!unconditional||std::isfinite(unconditional[i])),"MODEL_NONFINITE: raw CPU-sampler logits are not finite");
    const auto value=unconditional?double(unconditional[i])+cfg*(double(conditional[i])-double(unconditional[i])):double(conditional[i]);
    TORCH_CHECK(std::isfinite(value),"MODEL_NONFINITE: guided CPU-sampler logits overflowed");
    if(i>=start&&i<start+count){maximum=std::max(maximum,value);ordered.emplace_back(value,i);}
  }
  std::sort(ordered.begin(),ordered.end(),[](const auto&a,const auto&b){return a.first==b.first?a.second<b.second:a.first>b.first;});
  double z=0,za=0;for(const auto&[value,id]:ordered){z+=std::exp(value-maximum);za+=std::exp(alpha*(value-maximum));}
  const double h=std::clamp(std::log(z)+(std::log(z)-std::log(za))/(alpha-1),0.,std::log(double(count)));
  TORCH_CHECK(std::isfinite(h),"MODEL_NONFINITE: CPU Renyi entropy is not finite");
  const auto support=std::clamp<int64_t>(int64_t(std::floor(std::exp(h)+.5)),1,count);
  const auto rank=std::min<int64_t>(support-1,int64_t(std::floor(double(uniform)*support)));
  return uint32_t(offset+ordered[rank].second);
}
}

std::string json_quote(const std::string& value){
  std::string out="\"";for(unsigned char c:value){if(c=='"'||c=='\\'){out+='\\';out+=c;}else if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";else if(c=='\t')out+="\\t";else if(c>=32)out+=c;}return out+'"';
}
bool execution_uncertain(const std::string& message){
  auto upper=message;std::transform(upper.begin(),upper.end(),upper.begin(),[](unsigned char c){return char(std::toupper(c));});
  for(const auto* token:{"EXECUTION_UNCERTAIN","CAPTURE_STATE_UNCERTAIN","ILLEGAL ADDRESS","ILLEGAL_ADDRESS","ILLEGAL MEMORY ACCESS","ILLEGALADDRESS",
    "LAUNCHFAILURE","LAUNCH_FAILURE","LAUNCH FAILURE","LAUNCH FAILED","LAUNCH_FAILED","LAUNCH_TIMEOUT","LAUNCH TIMEOUT","LAUNCHTIMEDOUT",
    "DEVICE LOST","DEVICE-LOST","DEVICE_LOST","DEVICE-SIDE ASSERT","DEVICE SIDE ASSERT","DEVICE_ASSERT","DEVICEASSERT","DEVICE_SIDE_ASSERT",
    "HIPERRORASSERT","CUDAERRORASSERT","CUDA_ERROR_ASSERT","CUDAERRORDEVICEASSERT","HIP_ERROR_ASSERT","HSA_STATUS_ERROR_ASSERT",
    "MISALIGNED ADDRESS","MISALIGNED_ADDRESS","MISALIGNEDADDRESS","ILLEGAL INSTRUCTION","ILLEGAL_INSTRUCTION","ILLEGALINSTRUCTION",
    "HARDWARE_STACK_ERROR","HARDWARE STACK ERROR","HARDWARESTACKERROR","INVALID_ADDRESS_SPACE","INVALID ADDRESS SPACE","INVALIDADDRESSSPACE",
    "CUBLAS_STATUS_EXECUTION_FAILED","CUBLAS_STATUS_INTERNAL_ERROR","ROCBLAS_STATUS_INTERNAL_ERROR",
    "CONTEXT_IS_DESTROYED","CONTEXT IS DESTROYED","CONTEXT HAS BEEN DESTROYED","DRIVER SHUTTING DOWN","HSA_STATUS_ERROR_EXCEPTION"})if(upper.find(token)!=std::string::npos)return true;
  return false;
}
bool recoverable_private_preparation(const std::string& message){
  auto upper=message;std::transform(upper.begin(),upper.end(),upper.begin(),[](unsigned char c){return char(std::toupper(c));});
  // Fatal classification always wins over text such as 'not supported' in an
  // appended traceback. Unknown execution failures cannot authorize more GPU work.
  if(execution_uncertain(message))return false;
  for(const auto* token:{"OUT OF MEMORY","OUT_OF_MEMORY","MEMORY ALLOCATION","NOT IMPLEMENTED","NOT_IMPLEMENTED","NOT SUPPORTED","NOT_SUPPORTED","UNSUPPORTED","NO AVAILABLE KERNEL","NO VIABLE BACKEND"})
    if(upper.find(token)!=std::string::npos)return true;
  return false;
}

SamplingResult sample_gpu(const at::Tensor& raw,const at::Tensor& uniforms,const at::Tensor& bounds,
    int64_t guidance,float cfg,float alpha,int64_t offset){
  const auto choices=raw.size(1),batch=uniforms.size(0);
  auto valid=at::isfinite(raw).view({batch,guidance,choices}).all(-1).all(-1);
  auto grouped=raw.view({batch,guidance,choices});
  auto guided=guidance==1?grouped.select(1,0):grouped.select(1,1)+(grouped.select(1,0)-grouped.select(1,1))*cfg;
  valid=valid.logical_and(at::isfinite(guided).all(-1));
  const auto start=bounds.select(1,0),count=bounds.select(1,1);
  const auto positions=at::arange(choices,bounds.options()).unsqueeze(0);
  const auto legal=positions.ge(start.unsqueeze(1)).logical_and(positions.lt((start+count).unsqueeze(1)));
  const auto safe=at::where(valid.unsqueeze(1),guided,at::zeros_like(guided));
  auto masked=at::where(legal,safe,at::full_like(safe,-std::numeric_limits<float>::infinity()));
  auto centered=masked-std::get<0>(masked.max(-1,true));
  const auto z=at::logsumexp(centered,{-1}),za=at::logsumexp(centered*alpha,{-1});
  auto entropy=(z+(z-za)/(alpha-1.f)).clamp_min(0).minimum(count.to(at::kFloat).log());
  valid=valid.logical_and(at::isfinite(entropy));
  entropy=at::where(at::isfinite(entropy),entropy,at::zeros_like(entropy));
  auto support=(entropy.exp()+.5).floor().to(at::kLong).clamp_min(1).minimum(count);
  // stable=true gives deterministic attribute-index ordering when logits tie.
  const auto ordered=at::sort(masked,true,-1,true);
  const auto rank=(uniforms.view({batch})*support.to(at::kFloat)).floor().to(at::kLong).minimum(support-1);
  const auto tokens=std::get<1>(ordered).gather(1,rank.unsqueeze(1)).squeeze(1)+offset;
  return {tokens,valid,support};
}

struct Engine::Session {
  std::shared_ptr<Model> model;
  std::shared_ptr<CacheOwner> owner;
  std::vector<CacheSnapshot> source;
  PpAtenFrameOptions options{};
  int64_t logical=0,physical=0,bucket=0;
  at::Tensor records,partial,physical_map,pending,terms,valid,packed,uniforms,bounds;
  std::unique_ptr<at::cuda::CUDAGraph> graph;
  bool graph_ready=false;

  Session(std::shared_ptr<Model> m,const std::vector<CacheSnapshot>& input,const PpAtenFrameOptions& o,int64_t capacity=kContext)
    :model(std::move(m)),source(input),options(o),logical(o.logical_rows),physical(logical*o.guidance){
    auto d=model->device();auto integers=at::TensorOptions().device(d).dtype(at::kLong);
    auto floats=integers.dtype(at::kFloat);
    int64_t maximum=0;for(const auto& c:input)maximum=std::max(maximum,c.length);
    bucket=std::min<int64_t>(capacity,((maximum+64+63)/64)*64);
    owner=std::make_shared<CacheOwner>();owner->cache=model->make_cache(physical,capacity);
    owner->model_id=input[0].owner->model_id;owner->id=next_id.fetch_add(1);
    records=at::empty({logical,9},integers);
    std::vector<int64_t> flat;for(int64_t row=0;row<logical;++row)flat.insert(flat.end(),kPartial.begin(),kPartial.end());
    partial=long_on(flat,d).view({logical,9});
    std::vector<int64_t> rowmap;for(int64_t row=0;row<logical;++row)for(uint32_t g=0;g<o.guidance;++g)rowmap.push_back(row);
    physical_map=long_on(rowmap,d);
    pending=at::empty({physical,kBins[0]},floats);terms=at::empty({physical,2},floats);
    valid=at::ones({logical},integers.dtype(at::kBool));
    uniforms=at::empty({9,logical},floats);bounds=at::empty({9,logical,2},integers);
    packed=at::empty({logical,9+o.guidance*2+1},floats);
    reset_to_sources();
  }
  void reset_to_sources(){
    owner->cache.valid.zero_();
    std::vector<int64_t> lengths,absolute;
    for(int64_t row=0;row<physical;++row){const auto& in=source[row];
      for(size_t layer=0;layer<owner->cache.keys.size();++layer){
        TORCH_CHECK(owner->cache.int8==in.owner->cache.int8,"KV precision changed without rebuilding prefix");
        owner->cache.keys[layer].select(0,row).narrow(1,0,in.length).copy_(in.owner->cache.keys[layer].select(0,in.row).narrow(1,0,in.length));
        owner->cache.values[layer].select(0,row).narrow(1,0,in.length).copy_(in.owner->cache.values[layer].select(0,in.row).narrow(1,0,in.length));
        if(owner->cache.int8){
          owner->cache.key_scales[layer].select(0,row).narrow(1,0,in.length).copy_(in.owner->cache.key_scales[layer].select(0,in.row).narrow(1,0,in.length));
          owner->cache.value_scales[layer].select(0,row).narrow(1,0,in.length).copy_(in.owner->cache.value_scales[layer].select(0,in.row).narrow(1,0,in.length));
        }
      }
      owner->cache.valid.select(0,row).narrow(0,0,in.length).copy_(in.owner->cache.valid.select(0,in.row).narrow(0,0,in.length));
      pending.select(0,row).copy_(in.pending->attributes.select(0,in.pending_row));
      terms.select(0,row).copy_(in.pending->termination.select(0,in.pending_row));
      lengths.push_back(in.length);absolute.push_back(in.absolute);
    }
    owner->current_lengths=lengths;owner->current_absolute=absolute;
    owner->cache.lengths.copy_(long_on(lengths,model->device()));
    owner->cache.absolute_positions.copy_(long_on(absolute,model->device()));
    owner->poisoned=false;
  }
  bool compatible(const std::vector<CacheSnapshot>& input,const PpAtenFrameOptions& o)const{
    if(owner->poisoned||physical!=int64_t(input.size())||o.logical_rows!=options.logical_rows
      ||o.guidance!=options.guidance||o.cfg!=options.cfg||o.alpha!=options.alpha
      ||o.sampling_gpu!=options.sampling_gpu||o.efficient_attention!=options.efficient_attention
      ||o.graph_allowed!=options.graph_allowed)return false;
    for(int64_t row=0;row<physical;++row){const auto& in=input[row];
      if(owner->cache.int8!=in.owner->cache.int8)return false;
      if(in.owner!=owner||in.row!=row||in.length!=owner->current_lengths[row]
        ||in.absolute!=owner->current_absolute[row]||in.length>=bucket)return false;
    }
    return true;
  }
  bool replacement_plan(const std::vector<CacheSnapshot>& input,const PpAtenFrameOptions& o,std::vector<size_t>& changed)const{
    if(owner->poisoned||physical!=int64_t(input.size())||o.logical_rows!=options.logical_rows||o.guidance!=options.guidance
      ||o.cfg!=options.cfg||o.alpha!=options.alpha||o.sampling_gpu!=options.sampling_gpu
      ||o.efficient_attention!=options.efficient_attention||o.graph_allowed!=options.graph_allowed)return false;
    int64_t maximum=0;
    for(size_t row=0;row<input.size();++row){const auto& in=input[row];maximum=std::max(maximum,in.length);
      if(in.owner->cache.int8!=owner->cache.int8)return false;
      const bool same=in.owner==owner&&in.row==int64_t(row)&&in.length==owner->current_lengths[row]&&in.absolute==owner->current_absolute[row];
      if(same)continue;
      // No stale snapshot of a replaced bank row may remain alive. Incoming
      // aliases of this bank require a fresh COW session, never an overwrite.
      if(in.owner==owner||!row_unreferenced(*owner,row))return false;
      changed.push_back(row);
    }
    const auto next_bucket=maximum<bucket?bucket:std::min<int64_t>(owner->cache.capacity,((maximum+64+63)/64)*64);
    if(graph_ready&&next_bucket!=bucket)return false;
    return !changed.empty()||next_bucket!=bucket;
  }
  void replace_rows(const std::vector<CacheSnapshot>& input,const std::vector<size_t>& changed){
    int64_t maximum=0;for(const auto& in:input)maximum=std::max(maximum,in.length);
    if(!graph_ready&&maximum>=bucket)bucket=std::min<int64_t>(owner->cache.capacity,((maximum+64+63)/64)*64);
    for(const auto row:changed){const auto& in=input[row];
      copy_prefix_row(owner->cache,int64_t(row),in.owner->cache,in.row,in.length);
      pending.select(0,row).copy_(in.pending->attributes.select(0,in.pending_row));
      terms.select(0,row).copy_(in.pending->termination.select(0,in.pending_row));
      owner->current_lengths[row]=in.length;owner->current_absolute[row]=in.absolute;
      owner->cache.lengths.select(0,row).fill_(in.length);owner->cache.absolute_positions.select(0,row).fill_(in.absolute);
    }
  }
  void upload(const uint32_t* legal,const float* random){
    auto cpu_bounds=at::from_blob(const_cast<uint32_t*>(legal),{9,logical,2},at::TensorOptions().dtype(at::kUInt32)).to(at::kLong);
    auto cpu_uniforms=at::from_blob(const_cast<float*>(random),{9,logical},at::TensorOptions().dtype(at::kFloat));
    bounds.copy_(cpu_bounds);uniforms.copy_(cpu_uniforms);
  }
  void step_gpu(){
    trace("step_gpu_begin");
    records.copy_(partial);valid.fill_(true);
    for(int64_t field=0;field<9;++field){
      if(std::getenv("PAINTER_ATEN_TRACE")){std::fprintf(stderr,"ATEN_FIELD %lld\n",static_cast<long long>(field));std::fflush(stderr);}
      at::Tensor raw;
      if(field==0)raw=pending;
      else raw=model->decode(records.index_select(0,physical_map),owner->cache,Scope::field(field),bucket,
        options.efficient_attention?AttentionMode::Efficient:AttentionMode::Math).attributes;
      const auto sample=sample_gpu(raw,uniforms.select(0,field),bounds.select(0,field),options.guidance,float(options.cfg),float(options.alpha),kOffsets[field]);
      records.select(1,field).copy_(sample.tokens);valid.logical_and_(sample.valid);
    }
    const auto next=model->decode(records.index_select(0,physical_map),owner->cache,Scope::next(),bucket,
      options.efficient_attention?AttentionMode::Efficient:AttentionMode::Math);
    pending.copy_(next.attributes);terms.copy_(next.termination);
    valid.logical_and_(at::isfinite(pending).all(-1).logical_and(at::isfinite(terms).all(-1)).view({logical,int64_t(options.guidance)}).all(-1));
    Model::advance_cache(owner->cache);
    packed.copy_(at::cat({records.to(at::kFloat),terms.view({logical,int64_t(options.guidance*2)}),valid.to(at::kFloat).unsqueeze(1)},1));
    trace("step_gpu_end");
  }
  void step_cpu(const uint32_t* legal,const float* random){
    std::vector<int64_t> row_values;for(int64_t row=0;row<logical;++row)row_values.insert(row_values.end(),kPartial.begin(),kPartial.end());
    for(int64_t field=0;field<9;++field){
      at::Tensor raw=field==0?pending:model->decode(long_on(row_values,model->device()).view({logical,9}).index_select(0,physical_map),
        owner->cache,Scope::field(field),bucket,options.efficient_attention?AttentionMode::Efficient:AttentionMode::Math).attributes;
      auto host=raw.to(at::kCPU).contiguous();const float* data=host.data_ptr<float>();
      for(int64_t row=0;row<logical;++row){const auto base=(field*logical+row)*2;
        row_values[row*9+field]=sample_cpu(data+row*options.guidance*kBins[field],options.guidance==2?data+(row*2+1)*kBins[field]:nullptr,
          kBins[field],legal[base],legal[base+1],options.cfg,options.alpha,random[field*logical+row],kOffsets[field]);
      }
    }
    records.copy_(long_on(row_values,model->device()).view({logical,9}));
    const auto next=model->decode(records.index_select(0,physical_map),owner->cache,Scope::next(),bucket,
      options.efficient_attention?AttentionMode::Efficient:AttentionMode::Math);
    pending.copy_(next.attributes);terms.copy_(next.termination);
    valid.copy_(at::isfinite(pending).all(-1).logical_and(at::isfinite(terms).all(-1)).view({logical,int64_t(options.guidance)}).all(-1));
    Model::advance_cache(owner->cache);
    packed.copy_(at::cat({records.to(at::kFloat),terms.view({logical,int64_t(options.guidance*2)}),valid.to(at::kFloat).unsqueeze(1)},1));
  }
};

Engine::Engine(const std::string& file,const std::string& index,uint32_t ordinal,bool prefer_efficient)
  :selection_(choose_device(ordinal)),ordinal_(selection_.ordinal),id_(next_id.fetch_add(1)),
   stream_(c10::hip::getStreamFromPoolMasqueradingAsCUDA(false,ordinal_)){
  c10::InferenceMode inference;c10::StreamGuard guard(stream_.unwrap());
  trace("load_index");
  TORCH_CHECK(hipGetDeviceCount(&device_count_)==hipSuccess,"HIP count query failed");
  auto checkpoint=load_checkpoint(file,index);
  model_=std::make_shared<Model>(checkpoint.state,at::Device(at::kCUDA,ordinal_),at::kBFloat16);
  trace("weights_on_gpu");
  checkpoint.state.clear();
  optional_preparation_attempted_=prefer_efficient;
  verify_capabilities(prefer_efficient);
  efficient_prefill_=efficient_available_;
  trace("capabilities_done");
  update_info();update_diagnostics();
}
void Engine::update_info(){
  hipDeviceProp_t properties{};int runtime=0,driver=0;char pci[64]{};
  TORCH_CHECK(hipGetDeviceProperties(&properties,ordinal_)==hipSuccess&&hipRuntimeGetVersion(&runtime)==hipSuccess
    &&hipDriverGetVersion(&driver)==hipSuccess&&hipDeviceGetPCIBusId(pci,sizeof(pci),ordinal_)==hipSuccess,"HIP actual device identity unavailable");
  TORCH_CHECK(std::string(properties.gcnArchName).rfind("gfx",0)==0,"selected native Torch provider did not report an AMD gfx compute target");
  const auto& a=model_->architecture();
  const char* runtime_id=std::getenv("PAINTER_ATEN_RUNTIME_ID");
  std::ostringstream out;
  out<<"{\"provider\":\"aten_hip\",\"abi_version\":1,\"runtime_id\":"<<json_quote(runtime_id?runtime_id:"")<<",\"build_id\":"<<json_quote(PP_ATEN_BUILD_ID)<<",\"vendor\":\"amd\",\"device_ordinal\":"<<ordinal_<<",\"device_count\":"<<device_count_
    <<",\"device_name\":"<<json_quote(properties.name)<<",\"architecture\":"<<json_quote(properties.gcnArchName)
    <<",\"pci_bus_id\":"<<json_quote(pci)<<",\"hip_version\":"<<json_quote(std::to_string(runtime))
    <<",\"hip_driver_version\":"<<json_quote(std::to_string(driver))<<",\"torch_version\":"<<json_quote(TORCH_VERSION)
    <<",\"dtype\":\"BF16\",\"layers\":"<<a.layers<<",\"width\":"<<a.width<<",\"heads\":"<<a.heads
    <<",\"ffn_hidden\":"<<a.ffn_hidden<<",\"head_width\":"<<a.head_width
    <<",\"efficient_attention_available\":"<<(efficient_available_?"true":"false")
    <<",\"gpu_sampling_available\":"<<(sampling_available_?"true":"false")
    <<",\"hip_graph_available\":"<<(graph_available_&&efficient_available_&&sampling_available_&&!model_->int8_kv_enabled()?"true":"false")
    <<",\"kv_int8\":"<<kv_info()
    <<",\"optional_preparation_attempted\":"<<(optional_preparation_attempted_?"true":"false")
    <<",\"graph_preparation_attempted\":"<<(graph_preparation_attempted_?"true":"false")
    <<",\"hip_graph_implemented\":true,\"auto_device_selection\":"<<(selection_.automatic?"true":"false")
    <<",\"device_selection_reason\":"<<json_quote(selection_.reason)<<",\"device_candidates\":"<<selection_.candidates_json
    <<",\"disabled_reasons\":{\"efficient_attention\":"<<(efficient_available_?"null":json_quote(efficient_disabled_.empty()?"optional preparation not requested":efficient_disabled_))
    <<",\"gpu_sampling\":"<<(sampling_available_?"null":json_quote(sampling_disabled_.empty()?"optional preparation not requested":sampling_disabled_))
    <<",\"hip_graph\":"<<(model_->int8_kv_enabled()?json_quote("INT8 KV fused read/commit uses ordinary GPU execution; Graph intentionally disabled"):
      graph_available_?"null":json_quote(graph_disabled_.empty()?"deferred until explicit Graph request; Auto uses measured eager paths":graph_disabled_))<<"}"
    <<",\"graph_capture_deferred\":true,\"cpu_model_fallback\":false,\"native_weight_index\":true"
    <<",\"loaded_libraries\":{\"torch_hip\":"<<json_quote(module_path(L"torch_hip.dll"))
    <<",\"rocblas\":"<<json_quote(module_path(L"rocblas.dll"))
    <<",\"hip_runtime\":"<<json_quote(module_path(L"amdhip64_7.dll"))<<"}}";
  info_=out.str();
}
Engine::~Engine(){
  try{c10::InferenceMode inference;c10::StreamGuard guard(stream_.unwrap());stream_.synchronize();session_.reset();model_.reset();stream_.synchronize();}catch(...){}
}
void Engine::verify_capabilities(bool prepare_optional){
  trace("verify_math");
  const auto d=model_->device();auto f=at::TensorOptions().device(d).dtype(at::kFloat);
  auto a=at::arange(4,f).view({2,2})+1,b=a+4;
  for(auto type:{at::kFloat,at::kBFloat16}){
    auto actual=a.to(type).matmul(b.to(type)).to(at::kFloat).to(at::kCPU).contiguous();
    const float* p=actual.data_ptr<float>();TORCH_CHECK(p[0]==19&&p[1]==22&&p[2]==43&&p[3]==50,"ordinary HIP GEMM validation failed");
  }
  if(prepare_optional)try{
    trace("verify_efficient");
    auto q=(at::arange(64,f)*.031).sin().view({1,1,1,64}).to(at::kBFloat16);
    auto k=(at::arange(33*64,f)*.017).cos().view({1,1,33,64}).to(at::kBFloat16);
    auto v=(at::arange(33*64,f)*.021).sin().view({1,1,33,64}).to(at::kBFloat16);
    auto reference=(q.to(at::kFloat).matmul(k.to(at::kFloat).transpose(-2,-1))*.125).softmax(-1).matmul(v.to(at::kFloat));
    auto actual=std::get<0>(at::_scaled_dot_product_efficient_attention(q,k,v,std::nullopt,false,0.,false,std::nullopt)).to(at::kFloat);
    const auto error=(reference-actual).abs().max().item<float>();TORCH_CHECK(std::isfinite(error)&&error<.01,"efficient attention numeric check failed");
    efficient_available_=true;
  }catch(const std::exception& e){if(execution_uncertain(e.what()))throw;efficient_available_=false;efficient_disabled_=e.what();last_graph_error_=std::string("efficient preparation: ")+e.what();}
  if(prepare_optional)try{
    trace("verify_sampler");
    auto raw=at::zeros({8,128},f);auto u=at::full({8},.25,f);auto bounds=long_on(std::vector<int64_t>{17,20,17,20,17,20,17,20,17,20,17,20,17,20,17,20},d).view({8,2});
    const auto s=sample_gpu(raw,u,bounds,1,1,4,0);
    auto packed=at::stack({s.tokens,s.valid.to(at::kLong),s.support},1).to(at::kCPU).contiguous();
    auto p=packed.data_ptr<int64_t>();for(int row=0;row<8;++row)TORCH_CHECK(p[row*3]==22&&p[row*3+1]==1&&p[row*3+2]==20,"GPU Renyi numerical check failed");sampling_available_=true;
  }catch(const std::exception& e){
    if(execution_uncertain(e.what()))throw;
    sampling_available_=false;sampling_disabled_=e.what();last_graph_error_=std::string("GPU sampler preparation unavailable; CPU sampler retained: ")+e.what();
  }
  {
    trace("verify_full_model");
    Prefix prefix;prefix.text_ids={2,3};prefix.records={{{7068,7054,7054,7054,7054,7054,7054,7054,7054}},{{0,1024,2048,2318,4366,6414,6700,6800,7000}}};
    auto cache=model_->make_cache(1);auto pending=model_->prefill_rows(cache,{prefix},Scope::next(),64,AttentionMode::Math);
    auto known=kPartial;known[0]=510;
    auto row=long_on(std::vector<int64_t>(known.begin(),known.end()),d).view({1,9});
    auto reference=model_->decode(row,cache,Scope::field(1),64,AttentionMode::Math);
    TORCH_CHECK(at::isfinite(pending.attributes).all().item<bool>()&&at::isfinite(reference.attributes).all().item<bool>(),"actual full model returned nonfinite ordinary logits");
    if(efficient_available_)try{
      auto actual=model_->decode(row,cache,Scope::field(1),64,AttentionMode::Efficient);
      const auto error=(reference.attributes-actual.attributes).abs().max().item<float>();
      if(!(std::isfinite(error)&&error<=.25f)){efficient_available_=false;efficient_disabled_="efficient full-model numeric mismatch";last_graph_error_=efficient_disabled_;}
    }catch(const std::exception& e){if(execution_uncertain(e.what()))throw;efficient_available_=false;efficient_disabled_=e.what();last_graph_error_="efficient full-model optional preparation unavailable: "+efficient_disabled_;}
  }
  stream_.synchronize();
}
CacheSnapshot Engine::prefill(const Prefix& prefix,float* out_terms){
  std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(!poisoned_,"EXECUTION_UNCERTAIN: reload model required");
  validate_prefix(prefix);c10::InferenceMode inference;c10::StreamGuard guard(stream_.unwrap());
  trace("prefill_begin");
  // Keep the bank while one independent regional prefix is prepared. Once
  // its old row lease expires, only that row needs replacing in the bank.
  auto owner=std::make_shared<CacheOwner>();owner->model_id=id_;owner->id=next_id.fetch_add(1);owner->cache=model_->make_cache(1);
  const auto p=model_->prefill_rows(owner->cache,{prefix},Scope::next(),64,efficient_prefill_?AttentionMode::Efficient:AttentionMode::Math);
  auto pending=std::make_shared<PendingSnapshot>();pending->attributes=p.attributes.clone();pending->termination=p.termination.clone();
  auto host=p.termination.to(at::kCPU).contiguous();const auto* terms=host.data_ptr<float>();
  TORCH_CHECK(std::isfinite(terms[0])&&std::isfinite(terms[1])&&at::isfinite(p.attributes).all().item<bool>(),"MODEL_NONFINITE: prefill output is invalid");
  if(out_terms){out_terms[0]=terms[0];out_terms[1]=terms[1];}
  const auto length=int64_t(prefix.text_ids.size()+prefix.records.size());const auto absolute=length+prefix.position_offset;
  owner->current_lengths={length};owner->current_absolute={absolute};
  ++full_prefill_calls_;update_diagnostics();return leased({owner,pending,0,0,length,absolute});
}
CacheSnapshot Engine::extend(const CacheSnapshot& source,int64_t retained,const std::vector<std::array<int64_t,9>>& records,float* out_terms){
  std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(!poisoned_,"EXECUTION_UNCERTAIN: reload model required");
  TORCH_CHECK(source.owner&&source.pending&&source.owner->model_id==id_&&!source.owner->poisoned,"incremental prefix source is invalid or poisoned");
  TORCH_CHECK(source.row>=0&&source.row<source.owner->cache.batch&&retained>=0&&retained<=source.length&&!records.empty(),"incremental prefix retained length invalid");
  TORCH_CHECK(source.owner->cache.int8==model_->int8_kv_enabled(),"incremental prefix precision changed; rebuild required");
  const auto offset=source.absolute-source.length;
  TORCH_CHECK(offset>=0&&retained+int64_t(records.size())+offset<kContext,"incremental prefix exceeds physical/absolute context");
  c10::InferenceMode inference;c10::StreamGuard guard(stream_.unwrap());bool executed=false;
  try{
    auto owner=std::make_shared<CacheOwner>();owner->model_id=id_;owner->id=next_id.fetch_add(1);owner->cache=model_->make_cache(1);
    executed=true;copy_prefix_row(owner->cache,0,source.owner->cache,source.row,retained);
    Prefix suffix;suffix.records=records;suffix.position_offset=offset;
    const auto prediction=model_->append_prefix(owner->cache,retained,suffix,64,efficient_prefill_?AttentionMode::Efficient:AttentionMode::Math);
    auto pending=std::make_shared<PendingSnapshot>();pending->attributes=prediction.attributes.clone();pending->termination=prediction.termination.clone();
    auto host=prediction.termination.to(at::kCPU).contiguous();const auto* terms=host.data_ptr<float>();
    TORCH_CHECK(std::isfinite(terms[0])&&std::isfinite(terms[1])&&at::isfinite(prediction.attributes).all().item<bool>(),"MODEL_NONFINITE: incremental prefill output invalid");
    if(out_terms){out_terms[0]=terms[0];out_terms[1]=terms[1];}
    const auto length=retained+int64_t(records.size());owner->current_lengths={length};owner->current_absolute={length+offset};
    stream_.synchronize();++extended_prefill_calls_;reused_prefix_tokens_+=retained;update_diagnostics();
    return leased({owner,pending,0,0,length,length+offset});
  }catch(const std::exception& error){const std::string message=error.what();
    if((executed&&message.find("MODEL_NONFINITE")==std::string::npos)||execution_uncertain(message)){poisoned_=true;update_diagnostics();throw std::runtime_error("EXECUTION_UNCERTAIN: "+message);}throw;
  }
}
void Engine::configure(bool efficient_prefill){
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!poisoned_,"EXECUTION_UNCERTAIN: reload model required");
  TORCH_CHECK(!efficient_prefill||efficient_available_,"efficient prefill was not validated for this model load");
  if(efficient_prefill_!=efficient_prefill){c10::StreamGuard guard(stream_.unwrap());stream_.synchronize();session_.reset();}
  efficient_prefill_=efficient_prefill;update_diagnostics();
}
void Engine::begin_request(){
  std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(!poisoned_,"EXECUTION_UNCERTAIN: reload model required");
  c10::StreamGuard guard(stream_.unwrap());stream_.synchronize();session_.reset();update_diagnostics();
}
std::string Engine::kv_info()const{
  const bool enabled=model_->int8_kv_enabled();std::ostringstream out;
  out<<"{\"requested\":"<<json_quote(kv_preference_==0?"auto":kv_preference_==1?"enabled":"disabled")
    <<",\"available\":"<<(kv_available_?"true":"false")<<",\"enabled\":"<<(enabled?"true":"false")
    <<",\"reason\":"<<json_quote(kv_reason_)<<",\"storage\":"<<json_quote(enabled?"int8_per_token_head":"bf16")
    <<",\"head_width\":"<<model_->architecture().head_width<<",\"scale_dtype\":\"fp32\",\"scratch_dtype\":\"bf16\""
    <<",\"history_bytes_per_token_head_kv\":"<<(enabled?136:256)<<",\"bf16_history_bytes_per_token_head_kv\":256"
    <<",\"fused_dequantization\":"<<(enabled?"true":"false")<<",\"persistent_bf16_history\":"<<(enabled?"false":"true")
    <<",\"commit_complete_primitive_only\":true,\"graph_supported\":false}";
  return out.str();
}
void Engine::configure_kv(uint32_t preference){
  std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(!poisoned_,"EXECUTION_UNCERTAIN: reload model required");
  TORCH_CHECK(preference<=2,"INT8 KV preference must be0 Auto,1 Enabled or2 Disabled");kv_preference_=preference;
  c10::InferenceMode inference;c10::StreamGuard guard(stream_.unwrap());
  if(preference!=2&&!kv_preparation_attempted_){
    kv_preparation_attempted_=true;
    try{model_->prepare_int8_kv();stream_.synchronize();kv_available_=true;}
    catch(const std::exception& e){
      kv_available_=false;kv_reason_=std::string("INT8 KV preparation unavailable; same-GPU BF16 retained: ")+e.what();
      kv_unavailable_reason_=kv_reason_;
      if(execution_uncertain(e.what())){poisoned_=true;update_info();update_diagnostics();throw;}
      try{stream_.synchronize();}catch(const std::exception& health){poisoned_=true;update_info();update_diagnostics();throw std::runtime_error(std::string("EXECUTION_UNCERTAIN: INT8 KV preparation left unhealthy GPU stream: ")+health.what());}
    }
  }
  const bool enabled=preference!=2&&kv_available_;
  if(model_->int8_kv_enabled()!=enabled){stream_.synchronize();session_.reset();model_->configure_int8_kv(enabled);}
  if(enabled)kv_reason_="INT8 historical KV, FP32 per-token/head scales and BF16 scratch; fused read and complete-primitive commit validated; Graph disabled for this path";
  else if(preference==2)kv_reason_="INT8 KV explicitly disabled (including ordinary compatibility mode); BF16 cache retained";
  else kv_reason_=kv_unavailable_reason_.empty()?"INT8 KV unavailable; same-GPU BF16 retained":kv_unavailable_reason_;
  update_info();update_diagnostics();
}
void Engine::prepare_optional(){
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!poisoned_,"EXECUTION_UNCERTAIN: reload model required");
  if(optional_preparation_attempted_)return;
  // Latch before touching the backend: an unsupported optional path must not
  // rerun preparation on every click. Model load is the retry/session boundary.
  optional_preparation_attempted_=true;
  c10::InferenceMode inference;c10::StreamGuard guard(stream_.unwrap());
  try{verify_capabilities(true);}
  catch(const std::exception& e){
    const std::string message=e.what();
    efficient_available_=false;sampling_available_=false;graph_available_=false;
    efficient_disabled_=message;sampling_disabled_=message;graph_disabled_=message;
    if(execution_uncertain(message)){
      poisoned_=true;update_diagnostics();throw std::runtime_error("EXECUTION_UNCERTAIN: "+message);
    }
    last_graph_error_="optional preparation unavailable; ordinary same-GPU path retained: "+message;
    if(message.find("MODEL_NONFINITE")!=std::string::npos){update_info();update_diagnostics();throw;}
  }
  update_info();update_diagnostics();
}
void Engine::prepare_graph(){
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!poisoned_,"EXECUTION_UNCERTAIN: reload model required");
  if(model_->int8_kv_enabled()){graph_disabled_="INT8 KV uses non-captured fused read/commit; Graph is safely disabled";update_info();update_diagnostics();return;}
  if(graph_preparation_attempted_)return;
  TORCH_CHECK(efficient_available_&&sampling_available_,"Graph requires numerically validated efficient attention and GPU sampling");
  graph_preparation_attempted_=true;
  c10::InferenceMode inference;c10::StreamGuard guard(stream_.unwrap());
  bool began=false;
  try{
    auto x=at::ones({8},at::TensorOptions().device(model_->device()).dtype(at::kFloat));at::Tensor y;at::cuda::CUDAGraph graph;
    began=true;graph.capture_begin();y=x+2;graph.capture_end();graph.replay();stream_.synchronize();
    TORCH_CHECK((y-3).abs().max().item<float>()==0,"Graph preparation replay numeric mismatch");graph_available_=true;
  }catch(const std::exception& e){
    graph_available_=false;graph_disabled_=e.what();last_graph_error_=std::string("explicit Graph preparation: ")+e.what();
    if(began||execution_uncertain(e.what())){poisoned_=true;update_diagnostics();
      throw std::runtime_error(std::string("EXECUTION_UNCERTAIN: CAPTURE_STATE_UNCERTAIN: restart the application; explicit Graph preparation state is uncertain: ")+e.what());}
  }
  update_info();update_diagnostics();
}
void Engine::probe_auto(const PpAtenAutoProbeOptions& o){
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!poisoned_,"EXECUTION_UNCERTAIN: reload model required");
  TORCH_CHECK(o.struct_size>=sizeof(PpAtenAutoProbeOptions)&&o.prefix_records>=32&&o.prefix_records<=256
    &&std::isfinite(o.cfg)&&o.cfg>=0&&std::isfinite(o.alpha)&&o.alpha>1
    &&(o.guidance==2||o.cfg==1),"invalid private adaptive probe options");
  TORCH_CHECK(!o.candidate_efficient||efficient_available_,"private candidate efficient attention is not numerically validated");
  TORCH_CHECK(!o.candidate_gpu_sampling||(sampling_available_&&std::isfinite(float(o.cfg))
    &&std::isfinite(float(o.alpha))&&float(o.alpha)>1),"private candidate GPU sampling is not validated for these scalars");
  const auto geometry=row_geometry(o.logical_rows,o.guidance);
  const int64_t physical=int64_t(geometry.physical);
  const int64_t capacity=std::min<int64_t>(kContext,((int64_t(o.prefix_records)+33+64+63)/64)*64);
  const auto& arch=model_->architecture();
  uint64_t required=1;
  for(const auto factor:{uint64_t(arch.layers),uint64_t(arch.width),uint64_t(capacity),uint64_t(4),uint64_t(physical+o.guidance)})
    required=factor&&required>std::numeric_limits<uint64_t>::max()/factor?std::numeric_limits<uint64_t>::max():required*factor;
  const auto reserve=512ull<<20;
  required=required>std::numeric_limits<uint64_t>::max()-reserve?std::numeric_limits<uint64_t>::max():required+reserve;
  size_t free_bytes=0,total_bytes=0;
  c10::InferenceMode inference;c10::StreamGuard guard(stream_.unwrap());
  if(hipMemGetInfo(&free_bytes,&total_bytes)!=hipSuccess||free_bytes<required){
    std::ostringstream out;out<<"{\"status\":\"deferred\",\"method\":\"private_eager_abba_v1\",\"reason\":\"insufficient private-probe memory headroom; no cache eviction or batch reduction\",\"available_bytes\":"
      <<free_bytes<<",\"required_bytes\":"<<required<<",\"private_probe\":true,\"user_rng_consumed\":false,\"user_cache_touched\":false}";
    probe_info_=out.str();++auto_probe_calls_;update_diagnostics();return;
  }
  try{
    Prefix prefix;prefix.text_ids={2,262,398,356,264,302,452,291,350,304,954,427,673,347,260,434,585,894,260,281,286,274,260,1340,965,874,819,376,267,448,618,261,3};
    prefix.records.push_back({7068,7054,7054,7054,7054,7054,7054,7054,7054});
    prefix.records.push_back({0,1024,2048,2318,4366,6414,6702,6830,6958});
    while(prefix.records.size()<o.prefix_records)prefix.records.push_back({512,1536,2048,2518,4566,6415,6702,6830,6958});
    const auto make_source=[&](const Prefix& p){
      auto owner=std::make_shared<CacheOwner>();owner->model_id=id_;owner->id=next_id.fetch_add(1);owner->cache=model_->make_cache(1,capacity);
      const auto pred=model_->prefill_rows(owner->cache,{p},Scope::next(),64,AttentionMode::Math);
      auto pending=std::make_shared<PendingSnapshot>();pending->attributes=pred.attributes.clone();pending->termination=pred.termination.clone();
      const auto length=int64_t(p.text_ids.size()+p.records.size());owner->current_lengths={length};owner->current_absolute={length+p.position_offset};
      return CacheSnapshot{owner,pending,0,0,length,length+p.position_offset};
    };
    const auto conditional=make_source(prefix);CacheSnapshot unconditional;
    if(o.guidance==2){auto empty=prefix;empty.text_ids={2,3};empty.position_offset=31;unconditional=make_source(empty);}
    std::vector<CacheSnapshot> inputs;inputs.reserve(geometry.physical);
    for(size_t row=0;row<geometry.logical;++row){inputs.push_back(conditional);if(o.guidance==2)inputs.push_back(unconditional);}
    std::vector<uint32_t> bounds(geometry.bounds);std::vector<float> uniforms(geometry.attributes);
    for(size_t f=0;f<9;++f)for(size_t row=0;row<geometry.logical;++row){const auto i=f*geometry.logical+row;
      bounds[i*2]=(f==5?1:0);bounds[i*2+1]=(f==5?2:(f<2?(o.regional?127:1023):uint32_t(kBins[f])));
      uniforms[i]=float((f*17+row*11)%97+.25f)/97.f;}
    std::vector<double> baseline,candidate;
    const auto check_output=[&](const at::Tensor& host){const auto* p=host.data_ptr<float>();const auto cols=9+o.guidance*2+1;
      for(size_t row=0;row<geometry.logical;++row){const auto* q=p+row*cols;
        TORCH_CHECK(q[cols-1]==1,"MODEL_NONFINITE: private adaptive probe output invalid");
        for(size_t f=0;f<9;++f){const auto i=(f*geometry.logical+row)*2;TORCH_CHECK(std::isfinite(q[f])&&q[f]==std::floor(q[f])
          &&q[f]>=kOffsets[f]+bounds[i]&&q[f]<kOffsets[f]+uint64_t(bounds[i])+bounds[i+1],"MODEL_NONFINITE: private adaptive probe legality failed");}
        for(uint32_t g=0;g<o.guidance*2;++g)TORCH_CHECK(std::isfinite(q[9+g]),"MODEL_NONFINITE: private adaptive probe term invalid");}
    };
    // No Engine.session_, user snapshots or user RNG are used in this path.
    // ABBA limits drift; each pair has one warm and one timed full primitive.
    for(int trial=0;trial<4;++trial){const bool optimized=trial==1||trial==2;
      PpAtenFrameOptions options{sizeof(PpAtenFrameOptions),o.logical_rows,o.guidance,0,
        optimized?o.candidate_gpu_sampling:0,optimized?o.candidate_efficient:0,o.cfg,o.alpha};
      Session local(model_,inputs,options,capacity);local.upload(bounds.data(),uniforms.data());
      const auto step=[&]{if(options.sampling_gpu)local.step_gpu();else local.step_cpu(bounds.data(),uniforms.data());
        auto host=local.packed.to(at::kCPU).contiguous();check_output(host);};
      step();stream_.synchronize();const auto begin=std::chrono::steady_clock::now();step();stream_.synchronize();
      const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
      TORCH_CHECK(std::isfinite(seconds)&&seconds>0,"invalid private adaptive timing");
      (optimized?candidate:baseline).push_back(seconds);
    }
    std::ostringstream out;out.precision(12);out<<"{\"status\":\"measured\",\"method\":\"private_eager_abba_v1\",\"private_probe\":true,\"user_rng_consumed\":false,\"user_cache_touched\":false,\"graph_attempted\":false,\"generated_statistics\":false,"
      <<"\"logical_rows\":"<<o.logical_rows<<",\"physical_rows\":"<<physical<<",\"guidance\":"<<o.guidance
      <<",\"scope\":"<<json_quote(o.regional?"regional":"global")<<",\"prefix_records\":"<<o.prefix_records<<",\"text_tokens\":33,\"cache_capacity\":"<<capacity
      <<",\"warm_frames_per_trial\":1,\"timed_frames_per_trial\":1,\"cfg\":"<<o.cfg<<",\"alpha\":"<<o.alpha
      <<",\"candidate_efficient\":"<<(o.candidate_efficient?"true":"false")<<",\"candidate_gpu_sampling\":"<<(o.candidate_gpu_sampling?"true":"false")
      <<",\"baseline_seconds\":["<<baseline[0]<<','<<baseline[1]<<"],\"candidate_seconds\":["<<candidate[0]<<','<<candidate[1]<<"]}";
    probe_info_=out.str();++auto_probe_calls_;update_diagnostics();
  }catch(const std::exception& e){
    if(!recoverable_private_preparation(e.what())){poisoned_=true;update_diagnostics();throw std::runtime_error(std::string("EXECUTION_UNCERTAIN: private adaptive probe failed with fatal or unclassified execution state; no ordinary retry: ")+e.what());}
    probe_info_="{\"status\":\"disabled\",\"method\":\"private_eager_abba_v1\",\"private_probe\":true,\"user_rng_consumed\":false,\"user_cache_touched\":false,\"reason\":"+json_quote(e.what())+"}";
    ++auto_probe_calls_;update_diagnostics();
  }
}
void Engine::update_diagnostics(){
  uint64_t history_bytes=0,scratch_bytes=0;
  if(session_){const auto& cache=session_->owner->cache;
    for(const auto& tensor:cache.keys)history_bytes+=tensor.numel()*tensor.element_size();
    for(const auto& tensor:cache.values)history_bytes+=tensor.numel()*tensor.element_size();
    for(const auto& tensor:cache.key_scales)history_bytes+=tensor.numel()*tensor.element_size();
    for(const auto& tensor:cache.value_scales)history_bytes+=tensor.numel()*tensor.element_size();
    if(cache.packed_scratch.defined())scratch_bytes=cache.packed_scratch.numel()*cache.packed_scratch.element_size();
  }
  std::ostringstream out;out<<"{\"graph_captures\":"<<graph_captures_<<",\"graph_replays\":"<<graph_replays_<<",\"frame_calls\":"<<frame_calls_
    <<",\"ordinary_gpu_forwards\":"<<ordinary_forwards_<<",\"last_graph_error\":"<<(last_graph_error_.empty()?"null":json_quote(last_graph_error_))
    <<",\"device_poisoned\":"<<(poisoned_?"true":"false")<<",\"auto_probe_calls\":"<<auto_probe_calls_
    <<",\"auto_probe_result\":"<<probe_info_<<",\"kv_int8\":"<<kv_info()<<",\"active_session_history_bytes\":"<<history_bytes
    <<",\"active_session_scratch_bytes\":"<<scratch_bytes<<",\"full_prefill_calls\":"<<full_prefill_calls_
    <<",\"extended_prefill_calls\":"<<extended_prefill_calls_<<",\"reused_prefix_tokens\":"<<reused_prefix_tokens_
    <<",\"session_allocations\":"<<session_allocations_<<",\"session_retargets\":"<<session_retargets_
    <<",\"session_retarget_rows\":"<<session_retarget_rows_<<"}";diagnostics_=out.str();
}
FrameResult Engine::frame(const std::vector<CacheSnapshot>& input,const PpAtenFrameOptions& requested,
    const uint32_t* bounds,const float* uniforms){
  if(requested.graph_allowed)prepare_graph();
  std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(!poisoned_,"EXECUTION_UNCERTAIN: reload model required");
  validate_frame(input,requested,bounds,uniforms,id_);
  TORCH_CHECK(!requested.sampling_gpu||sampling_available_,"GPU sampling was not prepared and numerically validated for this model load");
  auto o=requested;o.efficient_attention=o.efficient_attention&&efficient_available_;
  o.graph_allowed=o.graph_allowed&&o.sampling_gpu&&graph_available_&&o.efficient_attention&&!model_->int8_kv_enabled();
  c10::InferenceMode inference;c10::StreamGuard guard(stream_.unwrap());
  const auto geometry=row_geometry(o.logical_rows,o.guidance);
  FrameResult result;result.tokens.resize(geometry.attributes);result.terms.resize(geometry.terms);
  result.valid.resize(geometry.logical);result.caches.reserve(geometry.physical);
  bool executed=false;
  try{
    trace("frame_begin");
    bool reusable=session_&&session_->compatible(input,o);
    if(!reusable&&session_){std::vector<size_t> changed;
      if(session_->replacement_plan(input,o,changed)){executed=true;session_->replace_rows(input,changed);reusable=true;++session_retargets_;session_retarget_rows_+=changed.size();}}
    if(!reusable){
      trace("session_allocate");
      session_.reset();
      auto next=std::make_shared<Session>(model_,input,o);next->upload(bounds,uniforms);
      if(o.graph_allowed){
        trace("graph_prepare");
        bool capture_begin_attempted=false;
        try{
          for(int warm=0;warm<2;++warm){next->reset_to_sources();next->step_gpu();stream_.synchronize();}
          next->reset_to_sources();stream_.synchronize();
          next->graph=std::make_unique<at::cuda::CUDAGraph>();capture_begin_attempted=true;next->graph->capture_begin();next->step_gpu();trace("capture_body_done");next->graph->capture_end();trace("capture_end_done");stream_.synchronize();
          next->reset_to_sources();stream_.synchronize();next->graph_ready=true;++graph_captures_;last_graph_error_.clear();
        }catch(const std::exception&e){
          // PyTorch's allocator and generator capture bookkeeping cannot be
          // reliably repaired by hipStreamEndCapture/reset after an attempted
          // capture_begin. Stop instead of replaying via an ordinary path.
          if(capture_begin_attempted)throw std::runtime_error(std::string("EXECUTION_UNCERTAIN: CAPTURE_STATE_UNCERTAIN: restart the application; reloading weights cannot repair complete-primitive capture state: ")+e.what());
          if(execution_uncertain(e.what()))throw;
          last_graph_error_=e.what();
          next->graph.reset();next->graph_ready=false;next->reset_to_sources();stream_.synchronize();
        }
      }
      session_=std::move(next);
      ++session_allocations_;
    }
    auto& s=*session_;s.upload(bounds,uniforms);executed=true;
    if(s.graph_ready){s.graph->replay();++graph_replays_;}
    else if(o.sampling_gpu){s.step_gpu();ordinary_forwards_+=9;}
    else{s.step_cpu(bounds,uniforms);ordinary_forwards_+=9;}
    auto host=s.packed.to(at::kCPU).contiguous();const auto* values=host.data_ptr<float>();
    const auto columns=9+o.guidance*2+1;
    for(size_t row=0;row<geometry.logical;++row){
      const float* p=values+row*columns;
      TORCH_CHECK(p[columns-1]==1,"MODEL_NONFINITE: completed primitive was not delivered");result.valid[row]=1;
      for(size_t field=0;field<9;++field){const auto token=p[field];const auto base=(field*geometry.logical+row)*2;
        TORCH_CHECK(std::isfinite(token)&&token==std::floor(token)&&token>=kOffsets[field]+bounds[base]
          &&token<kOffsets[field]+uint64_t(bounds[base])+bounds[base+1],"MODEL_NONFINITE: generated token escaped its legal field");result.tokens[row*9+field]=uint32_t(token);
      }
      for(uint32_t g=0;g<o.guidance;++g)for(int t=0;t<2;++t){TORCH_CHECK(std::isfinite(p[9+g*2+t]),"MODEL_NONFINITE: termination head is invalid");result.terms[(row*o.guidance+g)*2+t]=p[9+g*2+t];}
    }
    auto snapshot=std::make_shared<PendingSnapshot>();snapshot->attributes=s.pending.clone();snapshot->termination=s.terms.clone();
    for(size_t row=0;row<input.size();++row){++s.owner->current_lengths[row];++s.owner->current_absolute[row];
      result.caches.push_back(leased({s.owner,snapshot,int64_t(row),int64_t(row),s.owner->current_lengths[row],s.owner->current_absolute[row]}));}
    // The session owns independent prefix storage now. Do not retain every
    // preceding region/bucket's owner through a chain of immutable sources.
    s.source.clear();
    stream_.synchronize();++frame_calls_;update_diagnostics();return result;
  }catch(const std::exception& e){
    if(session_)session_->owner->poisoned=true;
    const std::string message=e.what();
    const bool invalid_output=message.find("MODEL_NONFINITE")!=std::string::npos;
    if((executed&&!invalid_output)||execution_uncertain(message))poisoned_=true;
    update_diagnostics();
    if(poisoned_)throw std::runtime_error("EXECUTION_UNCERTAIN: "+message);
    throw;
  }
}
}
