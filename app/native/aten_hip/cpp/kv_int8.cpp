#include "kv_int8.h"
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace pp {
namespace {
const char* source=R"HIP(
#include <hip/hip_runtime.h>
__device__ __forceinline__ float bf(unsigned short v){union{unsigned int u;float f;} x;x.u=(unsigned int)v<<16;return x.f;}
__device__ __forceinline__ unsigned short rb(float v){union{unsigned int u;float f;} x;x.f=v;x.u+=0x7fff+((x.u>>16)&1);return (unsigned short)(x.u>>16);}
extern "C" __global__ void kv_i8_commit(const unsigned short* S,signed char* K,float* SCALE,const long long* LENGTH,int B,int H,int C){
 const long long row=blockIdx.x;const int d=threadIdx.x;const int b=(row/H)%B;const long long t=LENGTH[b];
 __shared__ float reduce[64];const float x=bf(S[row*64+d]);reduce[d]=fabsf(x);__syncthreads();
 for(int s=32;s;s>>=1){if(d<s)reduce[d]=fmaxf(reduce[d],reduce[d+s]);__syncthreads();}
 const float scale=fmaxf(reduce[0]*(1.f/127.f),1.e-12f);
 K[(row*C+t)*64+d]=(signed char)fminf(127.f,fmaxf(-127.f,nearbyintf(x/scale)));
 if(d==0)SCALE[row*C+t]=scale;
}
extern "C" __global__ void kv_i8_partial(const unsigned short* Q,const signed char* K,const signed char* V,
 const float* KS,const float* VS,const unsigned short* SK,const unsigned short* SV,const unsigned char* VALID,
 const long long* LENGTH,float* O,float* M,float* L,int H,int C,int END,int PARTS){
 const long long bh=blockIdx.x,b=bh/H;const int lane=threadIdx.x,part=blockIdx.y,t=part*128+lane;
 const long long index=LENGTH[b],base=bh*C;__shared__ float score[128],prob[128],reduce[128],q[64];
 if(lane<64)q[lane]=bf(Q[bh*64+lane]);__syncthreads();
 const bool legal=t<END&&t<=index&&VALID[b*C+t];float dot=0.f;
 if(legal){
  const bool old=t<index;const float scale=old?KS[base+t]:1.f;
  #pragma unroll
  for(int d=0;d<64;++d){const float k=old?bf(rb(float(K[(base+t)*64+d])*scale)):bf(SK[bh*64+d]);dot+=q[d]*k;}
 }
 score[lane]=legal?dot*.125f:-__builtin_inff();reduce[lane]=score[lane];__syncthreads();
 for(int s=64;s;s>>=1){if(lane<s)reduce[lane]=fmaxf(reduce[lane],reduce[lane+s]);__syncthreads();}
 const float maximum=isfinite(reduce[0])?reduce[0]:0.f;__syncthreads();
 prob[lane]=legal?expf(score[lane]-maximum):0.f;reduce[lane]=prob[lane];__syncthreads();
 for(int s=64;s;s>>=1){if(lane<s)reduce[lane]+=reduce[lane+s];__syncthreads();}
 if(lane==0){M[bh*PARTS+part]=maximum;L[bh*PARTS+part]=reduce[0];}
 if(lane<64){float value=0.f;
  for(int j=0;j<128;++j){const int token=part*128+j;
   if(prob[j]>0){const float v=token<index?bf(rb(float(V[(base+token)*64+lane])*VS[base+token])):bf(SV[bh*64+lane]);value+=prob[j]*v;}}
  O[(bh*PARTS+part)*64+lane]=value;
 }
}
extern "C" __global__ void kv_i8_merge(const float* O,const float* M,const float* L,unsigned short* OUT,int PARTS){
 const long long bh=blockIdx.x;const int d=threadIdx.x;float maximum=-__builtin_inff();
 for(int p=0;p<PARTS;++p)if(L[bh*PARTS+p]>0)maximum=fmaxf(maximum,M[bh*PARTS+p]);
 float numerator=0,denominator=0;
 for(int p=0;p<PARTS;++p){const float factor=L[bh*PARTS+p]>0?expf(M[bh*PARTS+p]-maximum):0.f;
  numerator+=O[(bh*PARTS+p)*64+d]*factor;denominator+=L[bh*PARTS+p]*factor;}
 OUT[bh*64+d]=rb(denominator>0?numerator/denominator:0.f);
}
)HIP";
void hip_check(hipError_t status,const char* operation){TORCH_CHECK(status==hipSuccess,operation,": ",hipGetErrorString(status));}
}

struct Int8KvKernels::Impl {
  hipModule_t module=nullptr;hipFunction_t partial=nullptr,merge=nullptr,commit=nullptr;
  int device=-1;
  ~Impl(){if(module)hipModuleUnload(module);}
};
Int8KvKernels::Int8KvKernels():impl_(std::make_unique<Impl>()){}
Int8KvKernels::~Int8KvKernels()=default;
bool Int8KvKernels::ready()const noexcept{return impl_->module&&impl_->partial&&impl_->merge&&impl_->commit;}
void Int8KvKernels::prepare(at::Device device,int64_t head_width){
  TORCH_CHECK(device.is_cuda()&&head_width==64,"INT8 KV kernel requires a HIP GPU with64-channel heads");
  if(ready()){TORCH_CHECK(impl_->device==device.index(),"INT8 KV kernel belongs to another HIP device");return;}
#ifdef _WIN32
  const auto rtc=LoadLibraryExW(L"hiprtc0701.dll",nullptr,0x400|0x800);
  TORCH_CHECK(rtc,"INT8 KV preparation unavailable: paired HIPRTC library missing (",GetLastError(),")");
  const auto get=[&](const char* name){const auto p=GetProcAddress(rtc,name);TORCH_CHECK(p,"INT8 KV HIPRTC symbol missing: ",name);return p;};
  const auto create=reinterpret_cast<decltype(&hiprtcCreateProgram)>(get("hiprtcCreateProgram"));
  const auto compile=reinterpret_cast<decltype(&hiprtcCompileProgram)>(get("hiprtcCompileProgram"));
  const auto log_size=reinterpret_cast<decltype(&hiprtcGetProgramLogSize)>(get("hiprtcGetProgramLogSize"));
  const auto log_data=reinterpret_cast<decltype(&hiprtcGetProgramLog)>(get("hiprtcGetProgramLog"));
  const auto code_size=reinterpret_cast<decltype(&hiprtcGetCodeSize)>(get("hiprtcGetCodeSize"));
  const auto code_data=reinterpret_cast<decltype(&hiprtcGetCode)>(get("hiprtcGetCode"));
  const auto destroy=reinterpret_cast<decltype(&hiprtcDestroyProgram)>(get("hiprtcDestroyProgram"));
  hipDeviceProp_t properties{};hip_check(hipGetDeviceProperties(&properties,device.index()),"INT8 KV device properties");
  const std::string architecture=std::string("--gpu-architecture=")+properties.gcnArchName;
  const char* options[]={architecture.c_str(),"--std=c++17","-O3"};hiprtcProgram program=nullptr;
  TORCH_CHECK(create(&program,source,"painter_kv_int8.hip",0,nullptr,nullptr)==HIPRTC_SUCCESS,"INT8 KV HIPRTC create failed");
  const auto status=compile(program,3,options);size_t n=0;log_size(program,&n);std::string log(n,'\0');if(n)log_data(program,log.data());
  if(status!=HIPRTC_SUCCESS){destroy(&program);throw std::runtime_error("INT8 KV HIPRTC compile unavailable: "+log);}
  auto result=code_size(program,&n);std::vector<char> code(n);if(result==HIPRTC_SUCCESS)result=code_data(program,code.data());destroy(&program);
  TORCH_CHECK(result==HIPRTC_SUCCESS&&!code.empty(),"INT8 KV HIPRTC returned no executable code");
  hip_check(hipModuleLoadData(&impl_->module,code.data()),"INT8 KV module load");
  hip_check(hipModuleGetFunction(&impl_->partial,impl_->module,"kv_i8_partial"),"INT8 KV partial symbol");
  hip_check(hipModuleGetFunction(&impl_->merge,impl_->module,"kv_i8_merge"),"INT8 KV merge symbol");impl_->device=device.index();
  hip_check(hipModuleGetFunction(&impl_->commit,impl_->module,"kv_i8_commit"),"INT8 KV commit symbol");
#else
  TORCH_CHECK(false,"this native INT8 KV provider requires Windows HIPRTC");
#endif
}

at::Tensor Int8KvKernels::attention(const at::Tensor& query,const at::Tensor& keys,const at::Tensor& values,
    const at::Tensor& key_scale,const at::Tensor& value_scale,const at::Tensor& scratch_key,
    const at::Tensor& scratch_value,const at::Tensor& valid,const at::Tensor& lengths,int64_t bucket_end)const{
  TORCH_CHECK(ready()&&query.scalar_type()==at::kBFloat16&&query.dim()==4&&query.size(2)==1&&query.size(3)==64,"INT8 KV Q1 kernel unavailable for this query");
  TORCH_CHECK(keys.scalar_type()==at::kChar&&values.scalar_type()==at::kChar&&key_scale.scalar_type()==at::kFloat&&value_scale.scalar_type()==at::kFloat,"invalid INT8 KV storage/scales");
  const auto batch=keys.size(0),heads=keys.size(1),capacity=keys.size(2),bh=batch*heads,parts=(bucket_end+127)/128;
  TORCH_CHECK(keys.is_contiguous()&&values.is_contiguous()&&key_scale.is_contiguous()&&value_scale.is_contiguous()&&valid.is_contiguous()&&lengths.is_contiguous(),"INT8 KV history must be contiguous");
  TORCH_CHECK(bucket_end>0&&bucket_end<=capacity&&capacity<=std::numeric_limits<int>::max()&&heads<=std::numeric_limits<int>::max()&&bh<=std::numeric_limits<unsigned>::max(),"INT8 KV launch dimensions exceed device representation");
  const auto q=query.contiguous(),sk=scratch_key.contiguous(),sv=scratch_value.contiguous();
  auto temporary=at::empty({bh,parts,64},query.options().dtype(at::kFloat));
  auto maximum=at::empty({bh,parts},temporary.options()),denominator=at::empty_like(maximum);
  auto output=at::empty({batch,heads,1,64},query.options());
  const void *qp=q.data_ptr(),*kp=keys.data_ptr(),*vp=values.data_ptr(),*ks=key_scale.data_ptr(),*vs=value_scale.data_ptr(),
    *skp=sk.data_ptr(),*svp=sv.data_ptr(),*mask=valid.data_ptr(),*index=lengths.data_ptr();
  void *o=temporary.data_ptr(),*m=maximum.data_ptr(),*l=denominator.data_ptr(),*out=output.data_ptr();
  int h=int(heads),c=int(capacity),end=int(bucket_end),p=int(parts);
  void* args[]={&qp,&kp,&vp,&ks,&vs,&skp,&svp,&mask,&index,&o,&m,&l,&h,&c,&end,&p};
  auto stream=c10::hip::getCurrentHIPStream(query.device().index()).stream();
  hip_check(hipModuleLaunchKernel(impl_->partial,unsigned(bh),unsigned(parts),1,128,1,1,0,stream,args,nullptr),"INT8 KV partial launch");
  void* merge_args[]={&o,&m,&l,&out,&p};
  hip_check(hipModuleLaunchKernel(impl_->merge,unsigned(bh),1,1,64,1,1,0,stream,merge_args,nullptr),"INT8 KV merge launch");
  return output;
}
void Int8KvKernels::commit(const at::Tensor& packed,const at::Tensor& scales,const at::Tensor& scratch,const at::Tensor& lengths)const{
  TORCH_CHECK(ready()&&packed.dim()==6&&packed.size(1)==2&&packed.size(5)==64&&packed.is_contiguous()&&scales.is_contiguous()&&scratch.is_contiguous(),"INT8 KV fused-commit layout invalid");
  const auto rows=packed.size(0)*2*packed.size(2)*packed.size(3);
  TORCH_CHECK(rows<=std::numeric_limits<unsigned>::max()&&packed.size(2)<=std::numeric_limits<int>::max(),"INT8 KV fused-commit launch dimensions exceed device representation");
  const void* s=scratch.data_ptr();void* k=packed.data_ptr();void* scale=scales.data_ptr();const void* index=lengths.data_ptr();
  int b=int(packed.size(2)),h=int(packed.size(3)),c=int(packed.size(4));void* args[]={&s,&k,&scale,&index,&b,&h,&c};
  auto stream=c10::hip::getCurrentHIPStream(packed.device().index()).stream();
  hip_check(hipModuleLaunchKernel(impl_->commit,unsigned(rows),1,1,64,1,1,0,stream,args,nullptr),"INT8 KV fused-commit launch");
}
void quantize_kv_chunk(const at::Tensor& source,const at::Tensor& packed,const at::Tensor& scales){
  auto fp=source.to(at::kFloat);auto scale=std::get<0>(fp.abs().max(-1,false)).div(127.).clamp_min(1e-12);
  packed.copy_(fp.div(scale.unsqueeze(-1)).round().clamp(-127,127).to(at::kChar));scales.copy_(scale);
}
at::Tensor dequantize_kv_chunk(const at::Tensor& packed,const at::Tensor& scales,at::ScalarType dtype){return (packed.to(at::kFloat)*scales.unsqueeze(-1)).to(dtype);}
void Int8KvKernels::validate(at::Device device)const{
  auto f=at::TensorOptions().device(device).dtype(at::kFloat);constexpr int B=2,H=2,C=257;
  auto k=(at::arange(B*H*C*64,f)*.013).sin().reshape({B,H,C,64}).to(at::kBFloat16);
  auto v=(at::arange(B*H*C*64,f)*.019).cos().reshape({B,H,C,64}).to(at::kBFloat16);
  auto q=(at::arange(B*H*64,f)*.017).cos().reshape({B,H,1,64}).to(at::kBFloat16);
  auto sk=k.narrow(2,C-1,1).contiguous(),sv=v.narrow(2,C-1,1).contiguous();
  auto ki=at::empty_like(k,k.options().dtype(at::kChar)),vi=at::empty_like(ki);
  auto ks=at::empty({B,H,C},f),vs=at::empty_like(ks);quantize_kv_chunk(k,ki,ks);quantize_kv_chunk(v,vi,vs);
  auto valid=at::ones({B,C},f.dtype(at::kBool));valid.select(0,0).narrow(0,0,13).fill_(false);
  auto lengths=at::full({B},C-1,f.dtype(at::kLong));
  auto dk=dequantize_kv_chunk(ki,ks,at::kBFloat16),dv=dequantize_kv_chunk(vi,vs,at::kBFloat16);
  dk.narrow(2,C-1,1).copy_(sk);dv.narrow(2,C-1,1).copy_(sv);
  auto score=q.to(at::kFloat).matmul(dk.to(at::kFloat).transpose(-2,-1))*.125;
  score.masked_fill_(valid.logical_not().reshape({B,1,1,C}),-std::numeric_limits<float>::infinity());
  auto reference=score.softmax(-1).matmul(dv.to(at::kFloat)).to(at::kBFloat16);
  auto actual=attention(q,ki,vi,ks,vs,sk,sv,valid,lengths,C);
  const auto error=(reference.to(at::kFloat)-actual.to(at::kFloat)).abs().max().item<float>();
  TORCH_CHECK(std::isfinite(error)&&error<=.005,"INT8 KV fused-read numeric validation failed: ",error);
  auto packed=at::zeros({2,2,B,H,C,64},f.dtype(at::kChar));auto scales=at::ones({2,2,B,H,C},f);
  auto scratch=(at::arange(2*2*B*H*64,f)*.031).sin().reshape({2,2,B,H,1,64}).to(at::kBFloat16);
  auto index=at::arange(B,f.dtype(at::kLong))*(-2)+(C-1);commit(packed,scales,scratch,index);
  for(int layer=0;layer<2;++layer)for(int kind=0;kind<2;++kind)for(int row=0;row<B;++row){
    auto item=scratch[layer][kind].narrow(0,row,1);auto fp=item.to(at::kFloat);
    auto scale=std::get<0>(fp.abs().max(-1,false)).div(127.).clamp_min(1e-12);
    auto expected=fp.div(scale.unsqueeze(-1)).round().clamp(-127,127).to(at::kChar);
    auto stored=packed[layer][kind].narrow(0,row,1).narrow(2,C-1-row*2,1);
    auto stored_scale=scales[layer][kind].narrow(0,row,1).narrow(2,C-1-row*2,1);
    TORCH_CHECK((stored!=expected).sum().item<int64_t>()==0&&(stored_scale-scale).abs().max().item<float>()<1e-8,"INT8 KV fused-commit numeric validation failed");
  }
}
}
