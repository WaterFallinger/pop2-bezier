// Native HIP F32-accumulating attention over unmodified BF16/F32 KV storage.
// v2: all 256 threads participate in token-parallel PV accumulation.
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>

template<class T> __device__ __forceinline__ float pp_load(const T* p,unsigned long long i){return p[i];}
template<> __device__ __forceinline__ float pp_load<unsigned short>(const unsigned short* p,unsigned long long i){return __uint_as_float(((unsigned int)p[i])<<16);}
template<class T> __device__ void pp_attention_part(
 const float*q,const T*k,const T*v,const unsigned int*lengths,float*out,
 unsigned int heads,unsigned int size,unsigned int split,unsigned int splits,unsigned int dynamic,
 unsigned long long qs,unsigned long long ks,unsigned long long vs){
  const unsigned int h=blockIdx.x,part=blockIdx.y,tid=threadIdx.x;
  const unsigned int length=dynamic?lengths[h/heads]:size;
  const bool valid=length>0&&length<=size;
  const unsigned int begin=part*split,end=valid?min(begin+split,length):0;
  extern __shared__ float shared[];
  float*scores=shared;float*scratch=shared+split;
  // A fixed width-32 subgroup works on both wave32 and wave64 HIP targets.
  const unsigned int lane=tid&31,group=tid>>5;
  for(unsigned int t=group;t<split;t+=8){
    const unsigned int token=begin+t;float dot=0.f;
    if(token<end){
      dot=q[h*qs+lane]*pp_load(k,h*ks+token*64+lane);
      dot+=q[h*qs+lane+32]*pp_load(k,h*ks+token*64+lane+32);
      for(int d=16;d>0;d>>=1)dot+=__shfl_down(dot,d,32);
    }
    if(lane==0)scores[t]=token<end?dot*0.125f:-FLT_MAX;
  }
  __syncthreads();
  float m=-FLT_MAX;
  for(unsigned int t=tid;t<split;t+=256)m=fmaxf(m,scores[t]);
  scratch[tid]=m;__syncthreads();
  for(int d=128;d>0;d>>=1){if(tid<(unsigned int)d)scratch[tid]=fmaxf(scratch[tid],scratch[tid+d]);__syncthreads();}
  m=scratch[0];float z=0.f;
  __syncthreads();
  for(unsigned int t=tid;t<split;t+=256){float e=begin+t<end?expf(scores[t]-m):0.f;scores[t]=e;z+=e;}
  scratch[tid]=z;__syncthreads();
  for(int d=128;d>0;d>>=1){if(tid<(unsigned int)d)scratch[tid]+=scratch[tid+d];__syncthreads();}
  z=scratch[0];
  __syncthreads();
  const unsigned long long base=((unsigned long long)h*splits+part)*66;
  if(splits>1&&tid==0){out[base]=m;out[base+1]=z;}
  const unsigned int coordinate=tid&63,token_group=tid>>6;
  float partial_value=0.f;
  for(unsigned int t=begin+token_group;t<end;t+=4)partial_value+=scores[t-begin]*pp_load(v,h*vs+t*64+coordinate);
  scratch[tid]=partial_value;__syncthreads();
  if(tid<64){
    const float value=(scratch[tid]+scratch[tid+64])+(scratch[tid+128]+scratch[tid+192]);
    if(splits==1)out[(unsigned long long)h*64+tid]=valid&&z>0.f?value/z:__uint_as_float(0x7fc00000);
    else out[base+2+tid]=valid?value:__uint_as_float(0x7fc00000);
  }
}
extern "C" __global__ void pp_attention_f32(const float*q,const float*k,const float*v,const unsigned int*l,float*o,unsigned int h,unsigned int n,unsigned int s,unsigned int ns,unsigned int dyn,unsigned long long qs,unsigned long long ks,unsigned long long vs){pp_attention_part(q,k,v,l,o,h,n,s,ns,dyn,qs,ks,vs);}
extern "C" __global__ void pp_attention_bf16(const float*q,const unsigned short*k,const unsigned short*v,const unsigned int*l,float*o,unsigned int h,unsigned int n,unsigned int s,unsigned int ns,unsigned int dyn,unsigned long long qs,unsigned long long ks,unsigned long long vs){pp_attention_part(q,k,v,l,o,h,n,s,ns,dyn,qs,ks,vs);}
extern "C" __global__ void pp_attention_merge(const float*parts,float*out,unsigned int splits){
  const unsigned int h=blockIdx.x,d=threadIdx.x;if(d>=64)return;
  float m=-FLT_MAX;
  for(unsigned int s=0;s<splits;s++)m=fmaxf(m,parts[((unsigned long long)h*splits+s)*66]);
  float z=0.f,value=0.f;
  for(unsigned int s=0;s<splits;s++){
    const unsigned long long b=((unsigned long long)h*splits+s)*66;
    const float weight=expf(parts[b]-m);z+=weight*parts[b+1];value+=weight*parts[b+2+d];
  }
  out[(unsigned long long)h*64+d]=z>0.f?value/z:__uint_as_float(0x7fc00000);
}
