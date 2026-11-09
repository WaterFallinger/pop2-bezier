// Persistent historical KV: symmetric INT8, one FP32 scale per head/token.
// The uncommitted query token stays F32. Attention never expands the full KV.
__device__ __forceinline__ float pp_round_bf16(float x) {
  unsigned int u=__float_as_uint(x);
  u += 0x7fffU+((u>>16)&1U);
  return __uint_as_float(u&0xffff0000U);
}
extern "C" __global__ void pp_i8_scale(const float* x,float* s,unsigned long long count) {
  unsigned long long row=blockIdx.x; unsigned int d=threadIdx.x;
  if(row>=count)return;
  __shared__ float m[64];m[d]=fabsf(x[row*64+d]);__syncthreads();
  for(unsigned int step=32;step;step>>=1){if(d<step)m[d]=fmaxf(m[d],m[d+step]);__syncthreads();}
  if(d==0)s[row]=fmaxf(m[0]/127.f,1.e-12f);
}
extern "C" __global__ void pp_i8_pack(const float* x,const float* s,unsigned char* y,unsigned long long n) {
  unsigned long long i=(unsigned long long)blockIdx.x*256+threadIdx.x;
  if(i<n){int q=__float2int_rn(x[i]/s[i/64]);q=max(-127,min(127,q));y[i]=(unsigned char)(signed char)q;}
}
extern "C" __global__ void pp_i8_unpack(const unsigned char* x,const float* s,float* y,unsigned long long n) {
  unsigned long long i=(unsigned long long)blockIdx.x*256+threadIdx.x;
  if(i<n)y[i]=(float)(signed char)x[i]*s[i/64];
}
__device__ __forceinline__ float pp_i8_load(const unsigned char* p,const float* scale,
 unsigned int h,unsigned int t,unsigned int d,unsigned int capacity) {
  unsigned long long row=(unsigned long long)h*capacity+t;
  return pp_round_bf16((float)(signed char)p[row*64+d]*scale[row]);
}
extern "C" __global__ void pp_i8_attention(const float* q,const unsigned char* k,const unsigned char* v,
 const float* ks,const float* vs,const float* sk,const float* sv,const unsigned int* past,float* out,
 unsigned int heads,unsigned int capacity,unsigned int split,unsigned int splits) {
  unsigned int h=blockIdx.x,part=blockIdx.y,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
  unsigned int p=past[h/heads],length=p+1,begin=part*split;
  bool valid=p<=capacity;unsigned int end=valid?min(begin+split,length):0;
  extern __shared__ float mem[];float* scores=mem;float* scratch=mem+split;
  for(unsigned int t=warp;t<split;t+=4){unsigned int at=begin+t;float dot=0.f;
    if(at<end){float a=at==p?sk[(unsigned long long)h*64+lane]:pp_i8_load(k,ks,h,at,lane,capacity);
      float b=at==p?sk[(unsigned long long)h*64+lane+32]:pp_i8_load(k,ks,h,at,lane+32,capacity);
      dot=q[(unsigned long long)h*64+lane]*a+q[(unsigned long long)h*64+lane+32]*b;
      for(int d=16;d;d>>=1)dot+=__shfl_down_sync(0xffffffff,dot,d);}
    if(lane==0)scores[t]=at<end?dot*0.125f:-3.402823466e38f;
  }__syncthreads();float m=-3.402823466e38f;
  for(unsigned int t=tid;t<split;t+=128)m=fmaxf(m,scores[t]);
  scratch[tid]=m;__syncthreads();for(int d=64;d;d>>=1){if(tid<(unsigned)d)scratch[tid]=fmaxf(scratch[tid],scratch[tid+d]);__syncthreads();}
  m=scratch[0];float z=0.f;for(unsigned int t=tid;t<split;t+=128){float e=begin+t<end?expf(scores[t]-m):0.f;scores[t]=e;z+=e;}
  scratch[tid]=z;__syncthreads();for(int d=64;d;d>>=1){if(tid<(unsigned)d)scratch[tid]+=scratch[tid+d];__syncthreads();}
  z=scratch[0];unsigned long long base=((unsigned long long)h*splits+part)*66;
  if(tid==0){out[base]=m;out[base+1]=z;}
  if(tid<64){float sum=0.f;for(unsigned int at=begin;at<end;at++){
      float value=at==p?sv[(unsigned long long)h*64+tid]:pp_i8_load(v,vs,h,at,tid,capacity);
      sum+=scores[at-begin]*value;}
    out[base+2+tid]=valid?sum:__uint_as_float(0x7fc00000);}
}
extern "C" __global__ void pp_i8_merge(const float* parts,float* out,unsigned int splits){
  unsigned int h=blockIdx.x,d=threadIdx.x;float m=-3.402823466e38f;
  for(unsigned int p=0;p<splits;p++)m=fmaxf(m,parts[((unsigned long long)h*splits+p)*66]);
  float z=0.f,y=0.f;for(unsigned int p=0;p<splits;p++){unsigned long long at=((unsigned long long)h*splits+p)*66;
    float w=expf(parts[at]-m);z+=parts[at+1]*w;y+=parts[at+2+d]*w;}
  out[(unsigned long long)h*64+d]=z>0.f?y/z:__uint_as_float(0x7fc00000);
}
