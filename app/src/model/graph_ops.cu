// CUDA-only graph-safe operations. No runtime header dependency.
__device__ __forceinline__ float rd(const void* p, unsigned long long i, unsigned type) {
    return type ? __uint_as_float(((const unsigned short*)p)[i] << 16) : ((const float*)p)[i];
}
__device__ __forceinline__ unsigned short bf(float x) {
    unsigned bits = __float_as_uint(x);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return (unsigned short)((bits >> 16) | 0x40U);
    return (unsigned short)((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
__device__ __forceinline__ float rnd(float x, unsigned type) {
    return type ? __uint_as_float((unsigned)bf(x) << 16) : x;
}
__device__ __forceinline__ void wr(void* p, unsigned long long i, float x, unsigned type) {
    if (type) ((unsigned short*)p)[i] = bf(x); else ((float*)p)[i] = x;
}
extern "C" __global__ void pp_graph_embedding(const unsigned* records, const void* embedding,
    const unsigned* rowmap, void* out, unsigned physical, unsigned width, unsigned type) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= physical * width) return;
    unsigned lane = rowmap[i / width], channel = i % width;
    float values[16];
    #pragma unroll
    for (unsigned field=0; field<16; ++field)
        values[field] = field<9 ? rd(embedding, (unsigned long long)records[lane*9+field] * width + channel, type) : 0.f;
    // Candle's nine-lookups sum is a 16-thread BF16 tree, not a sequential
    // float sum. Preserve those intermediate BF16 rounding boundaries.
    #pragma unroll
    for (unsigned stride=8;stride;stride>>=1)
        for(unsigned j=0;j<stride;++j)values[j]=rnd(values[j]+values[j+stride],type);
    wr(out, i, values[0], type);
}
// One warp per head, with two channels per lane. RMS reduction follows the
// ordinary 32-thread RMSNorm kernel; intermediate BF16 operations stay rounded.
extern "C" __global__ void pp_graph_qkv(const void* qkv, const void* parameters,
    const unsigned* positions, void* out, unsigned physical, unsigned heads,
    unsigned dim, unsigned max_context, unsigned type) {
    unsigned headrow=blockIdx.x, t=threadIdx.x;
    unsigned row=headrow/heads, head=headrow%heads, width=heads*dim;
    float q0=0.f,q1=0.f,k0=0.f,k1=0.f;
    unsigned c0=t, c1=t+32;
    if(c0<dim){q0=rd(qkv,(unsigned long long)row*3*width+head*dim+c0,type);k0=rd(qkv,(unsigned long long)row*3*width+width+head*dim+c0,type);}
    if(c1<dim){q1=rd(qkv,(unsigned long long)row*3*width+head*dim+c1,type);k1=rd(qkv,(unsigned long long)row*3*width+width+head*dim+c1,type);}
    float qs=q0*q0+q1*q1,ks=k0*k0+k1*k1;
    #pragma unroll
    for(unsigned delta=16;delta;delta>>=1){qs+=__shfl_down_sync(0xffffffff,qs,delta);ks+=__shfl_down_sync(0xffffffff,ks,delta);}
    float qi=rsqrtf(__shfl_sync(0xffffffff,qs,0)/dim+1.e-6f),ki=rsqrtf(__shfl_sync(0xffffffff,ks,0)/dim+1.e-6f);
    __shared__ float q[64],k[64];
    if(c0<dim){q[c0]=rnd(q0*qi*rd(parameters,c0,type),type);k[c0]=rnd(k0*ki*rd(parameters,dim+c0,type),type);}
    if(c1<dim){q[c1]=rnd(q1*qi*rd(parameters,c1,type),type);k[c1]=rnd(k1*ki*rd(parameters,dim+c1,type),type);}
    __syncwarp();
    unsigned pos=positions[row];
    for(unsigned c=t;c<dim;c+=32){
        unsigned pair=c>>1;
        float co=rd(parameters,2*dim+(unsigned long long)pos*(dim/2)+pair,type);
        float si=rd(parameters,2*dim+(unsigned long long)max_context*(dim/2)+(unsigned long long)pos*(dim/2)+pair,type);
        float qo,ko;
        if(c&1){qo=rnd(rnd(q[c-1]*si,type)+rnd(q[c]*co,type),type);ko=rnd(rnd(k[c-1]*si,type)+rnd(k[c]*co,type),type);}
        else{qo=rnd(rnd(q[c]*co,type)-rnd(q[c+1]*si,type),type);ko=rnd(rnd(k[c]*co,type)-rnd(k[c+1]*si,type),type);}
        unsigned long long at=(unsigned long long)headrow*dim+c,plane=(unsigned long long)physical*width;
        wr(out,at,qo,type);wr(out,plane+at,ko,type);
        wr(out,2*plane+at,rd(qkv,(unsigned long long)row*3*width+2*width+head*dim+c,type),type);
    }
}
extern "C" __global__ void pp_graph_swiglu(const void* input, void* output,
    unsigned rows,unsigned hidden,unsigned type){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*hidden)return;
    unsigned row=i/hidden,c=i%hidden;
    float gate=rd(input,(unsigned long long)row*hidden*2+c,type);
    float up=rd(input,(unsigned long long)row*hidden*2+hidden+c,type);
    // Candle's BF16 silu rounds exp, addition and division separately.
    // Fusing kernels must not silently replace that with FP32 silu.
    float exponential=rnd(expf(-gate),type);
    float denominator=rnd(1.f+exponential,type);
    float silu=rnd(gate/denominator,type);
    wr(output,i,rnd(silu*up,type),type);
}
extern "C" __global__ void pp_graph_indices(const unsigned* lengths,unsigned* out,
    unsigned physical,unsigned heads,unsigned dim){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<physical*heads*dim)out[i]=lengths[i/(heads*dim)];
}
extern "C" __global__ void pp_graph_mask(const unsigned* lengths,float* out,
    unsigned physical,unsigned heads,unsigned capacity){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<physical*heads*capacity)out[i]=(i%capacity<=lengths[i/(heads*capacity)])?0.f:-__int_as_float(0x7f800000);
}
extern "C" __global__ void pp_graph_finite(const float* input,unsigned* output,
    unsigned logical,unsigned guidance,unsigned width){
    unsigned row=blockIdx.x,t=threadIdx.x;
    unsigned valid=1;
    for(unsigned i=t;i<guidance*width;i+=blockDim.x){float x=input[(unsigned long long)row*guidance*width+i];if(!isfinite(x))valid=0;}
    __shared__ unsigned values[256];values[t]=valid;__syncthreads();
    for(unsigned d=blockDim.x/2;d;d>>=1){if(t<d)values[t]&=values[t+d];__syncthreads();}
    if(!t)output[row]=values[0];
}
extern "C" __global__ void pp_graph_pack(const unsigned* records,const float* terms,
    const unsigned* valid,float* output,unsigned logical,unsigned guidance){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,stride=10+guidance*2;
    if(i>=logical*stride)return;
    unsigned row=i/stride,c=i%stride;
    if(c<9)output[i]=(float)records[row*9+c];
    else if(c<9+guidance*2)output[i]=terms[(unsigned long long)row*guidance*2+c-9];
    else{
        unsigned ok=valid[row];
        for(unsigned j=0;j<guidance*2;++j)if(!isfinite(terms[(unsigned long long)row*guidance*2+j]))ok=0;
        output[i]=(float)ok;
    }
}
