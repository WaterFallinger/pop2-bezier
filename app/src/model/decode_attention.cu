// NVIDIA FP32-accumulating decode attention. No model/weight quantization.
__device__ __forceinline__ float bf16_load(const unsigned short* p, unsigned long long i) {
  return __uint_as_float(((unsigned int)p[i]) << 16);
}
template<class T> __device__ __forceinline__ float kv_load(const T* p, unsigned long long i) { return p[i]; }
template<> __device__ __forceinline__ float kv_load<unsigned short>(const unsigned short* p, unsigned long long i) { return bf16_load(p,i); }
template<class T> __device__ void attention_part(
 const float* q,const T* k,const T* v,const unsigned int* lengths,float* out,
 unsigned int heads,unsigned int size,unsigned int split,unsigned int splits,unsigned int dynamic,
 unsigned long long qs,unsigned long long ks,unsigned long long vs) {
  const unsigned int h=blockIdx.x, part=blockIdx.y, tid=threadIdx.x;
  const unsigned int length=dynamic?lengths[h/heads]:size;
  const bool valid=length>0 && length<=size;
  const unsigned int begin=part*split, end=valid?min(begin+split,length):0;
  extern __shared__ float shared[];
  float* scores=shared; float* scratch=shared+split;
  const unsigned int lane=tid&31, warp=tid>>5;
  for(unsigned int t=warp;t<split;t+=4) {
    const unsigned int token=begin+t;
    float dot=0.f;
    if(token<end) {
      dot=q[h*qs+lane]*kv_load(k,h*ks+token*64+lane);
      dot+=q[h*qs+lane+32]*kv_load(k,h*ks+token*64+lane+32);
      for(int d=16;d>0;d>>=1) dot+=__shfl_down_sync(0xffffffff,dot,d);
    }
    if(lane==0) scores[t]=token<end?dot*0.125f:-3.402823466e38f;
  }
  __syncthreads();
  float m=-3.402823466e38f;
  for(unsigned int t=tid;t<split;t+=128) m=fmaxf(m,scores[t]);
  scratch[tid]=m; __syncthreads();
  for(int d=64;d>0;d>>=1){if(tid<(unsigned int)d)scratch[tid]=fmaxf(scratch[tid],scratch[tid+d]);__syncthreads();}
  m=scratch[0]; float z=0.f;
  for(unsigned int t=tid;t<split;t+=128){float e=(begin+t<end)?expf(scores[t]-m):0.f;scores[t]=e;z+=e;}
  scratch[tid]=z;__syncthreads();
  for(int d=64;d>0;d>>=1){if(tid<(unsigned int)d)scratch[tid]+=scratch[tid+d];__syncthreads();}
  z=scratch[0];
  const unsigned long long base=((unsigned long long)h*splits+part)*66;
  if(splits>1 && tid==0){out[base]=m;out[base+1]=z;}
  if(tid<64){
    float value=0.f;
    for(unsigned int t=begin;t<end;t++) value+=scores[t-begin]*kv_load(v,h*vs+t*64+tid);
    if(splits==1) out[h*64+tid]=(valid&&z>0.f)?value/z:__uint_as_float(0x7fc00000);
    else out[base+2+tid]=valid?value:__uint_as_float(0x7fc00000);
  }
}
extern "C" __global__ void pp_attention_f32(const float*q,const float*k,const float*v,const unsigned int*l,float*o,unsigned int h,unsigned int n,unsigned int s,unsigned int ns,unsigned int dyn,unsigned long long qs,unsigned long long ks,unsigned long long vs){attention_part(q,k,v,l,o,h,n,s,ns,dyn,qs,ks,vs);}
extern "C" __global__ void pp_attention_bf16(const float*q,const unsigned short*k,const unsigned short*v,const unsigned int*l,float*o,unsigned int h,unsigned int n,unsigned int s,unsigned int ns,unsigned int dyn,unsigned long long qs,unsigned long long ks,unsigned long long vs){attention_part(q,k,v,l,o,h,n,s,ns,dyn,qs,ks,vs);}
extern "C" __global__ void pp_attention_merge(const float*parts,float*out,unsigned int splits){
  unsigned int h=blockIdx.x,d=threadIdx.x; if(d>=64)return;
  float m=-3.402823466e38f;
  for(unsigned int s=0;s<splits;s++)m=fmaxf(m,parts[((unsigned long long)h*splits+s)*66]);
  float z=0.f,value=0.f;
  for(unsigned int s=0;s<splits;s++){
    unsigned long long b=((unsigned long long)h*splits+s)*66;
    float weight=expf(parts[b]-m);z+=weight*parts[b+1];value+=weight*parts[b+2+d];
  }
  out[h*64+d]=z>0.f?value/z:__uint_as_float(0x7fc00000);
}
