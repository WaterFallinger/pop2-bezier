// One logical candidate per block. No warp-size-dependent primitives.
// This file intentionally has no toolkit header dependency: NVRTC + libdevice
// supplied with the application is sufficient.

__device__ __forceinline__ bool pp_finite(float value) {
    union { float f; unsigned int u; } bits;
    bits.f = value;
    return (bits.u & 0x7fffffffU) < 0x7f800000U;
}

__device__ __forceinline__ float pp_negative_infinity() {
    union { float f; unsigned int u; } bits;
    bits.u = 0xff800000U;
    return bits.f;
}

__device__ __forceinline__ bool pp_before(float a, unsigned int ai, float b, unsigned int bi) {
    return a > b || (a == b && ai < bi);
}

extern "C" __global__ void pp_renyi_sample(
    const float* raw,
    const float* uniforms,
    const unsigned int* bounds,
    unsigned int* output,
    unsigned int batch,
    unsigned int choices,
    unsigned int guidance,
    float alpha,
    float cfg,
    unsigned int token_offset,
    unsigned int padded_choices
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (row >= batch) return;

    extern __shared__ float shared_floats[];
    float* values = shared_floats;
    unsigned int* ids = reinterpret_cast<unsigned int*>(values + padded_choices);
    float* reduce_a = reinterpret_cast<float*>(ids + padded_choices);
    float* reduce_b = reduce_a + blockDim.x;
    unsigned int* reduce_id = reinterpret_cast<unsigned int*>(reduce_b + blockDim.x);
    unsigned int* reduce_bad = reduce_id + blockDim.x;
    __shared__ unsigned int start;
    __shared__ unsigned int count;
    __shared__ unsigned int invalid;
    __shared__ unsigned int support;
    __shared__ unsigned int rank;

    if (tid == 0) {
        start = bounds[2 * row];
        count = bounds[2 * row + 1];
        const float u = uniforms[row];
        invalid = !pp_finite(u) || u < 0.0f || u >= 1.0f
            || start >= choices || count == 0 || count > choices - (start < choices ? start : 0);
        if (invalid) {
            output[row] = token_offset + (start < choices ? start : 0);
            output[(unsigned long long)batch + row] = 0;
            output[2ULL * batch + row] = 1;
        }
    }
    __syncthreads();
    if (invalid) return;

    unsigned int sort_width = 1;
    while (sort_width < count) sort_width <<= 1;
    const unsigned int conditional_row = row * guidance;
    const float neg_inf = pp_negative_infinity();
    float local_max = neg_inf;
    unsigned int local_id = 0xffffffffU;
    unsigned int local_bad = 0;

    // Inspect the entire projected head, including currently illegal choices.
    // Compact only legal candidates into shared memory for the later sort.
    for (unsigned int col = tid; col < choices; col += blockDim.x) {
        const float c = raw[(unsigned long long)conditional_row * choices + col];
        float value = c;
        bool finite = pp_finite(c);
        if (guidance == 2) {
            const float u = raw[(unsigned long long)(conditional_row + 1) * choices + col];
            finite = finite && pp_finite(u);
            value = u + cfg * (c - u);
        }
        finite = finite && pp_finite(value);
        local_bad |= !finite;
        if (!finite) value = 0.0f;
        if (col >= start && col - start < count) {
            const unsigned int compact = col - start;
            values[compact] = value;
            ids[compact] = col;
            if (pp_before(value, col, local_max, local_id)) {
                local_max = value;
                local_id = col;
            }
        }
    }
    for (unsigned int col = tid; col < sort_width; col += blockDim.x) {
        if (col >= count) { values[col] = neg_inf; ids[col] = 0xffffffffU; }
    }
    reduce_a[tid] = local_max;
    reduce_id[tid] = local_id;
    reduce_bad[tid] = local_bad;
    __syncthreads();
    for (unsigned int step = blockDim.x >> 1; step; step >>= 1) {
        if (tid < step) {
            if (pp_before(reduce_a[tid + step], reduce_id[tid + step], reduce_a[tid], reduce_id[tid])) {
                reduce_a[tid] = reduce_a[tid + step];
                reduce_id[tid] = reduce_id[tid + step];
            }
            reduce_bad[tid] |= reduce_bad[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) {
        invalid = reduce_bad[0];
        if (invalid) {
            output[row] = token_offset + start;
            output[(unsigned long long)batch + row] = 0;
            output[2ULL * batch + row] = 1;
        }
    }
    __syncthreads();
    if (invalid) return;
    const float maximum = reduce_a[0];
    const unsigned int maximum_id = reduce_id[0];
    float sum = 0.0f;
    float sum_alpha = 0.0f;
    for (unsigned int col = tid; col < count; col += blockDim.x) {
        const float shifted = values[col] - maximum;
        sum += __expf(shifted);
        sum_alpha += __expf(alpha * shifted);
    }
    // All threads must finish reading maximum_id before reusing reduce_id.
    __syncthreads();
    reduce_a[tid] = sum;
    reduce_b[tid] = sum_alpha;
    __syncthreads();
    for (unsigned int step = blockDim.x >> 1; step; step >>= 1) {
        if (tid < step) { reduce_a[tid] += reduce_a[tid + step]; reduce_b[tid] += reduce_b[tid + step]; }
        __syncthreads();
    }
    if (tid == 0) {
        const float log_z = __logf(reduce_a[0]);
        const float log_za = __logf(reduce_b[0]);
        float correction = log_z - log_za;
        if (correction < 0.0f) correction = 0.0f;
        float entropy = log_z + correction / (alpha - 1.0f);
        invalid = !pp_finite(log_z) || !pp_finite(log_za) || !pp_finite(entropy);
        if (invalid) {
            output[row] = token_offset + start;
            output[(unsigned long long)batch + row] = 0;
            output[2ULL * batch + row] = 1;
        } else {
            const float maximum_entropy = __logf((float)count);
            if (entropy < 0.0f) entropy = 0.0f;
            if (entropy > maximum_entropy) entropy = maximum_entropy;
            const float rounded = __expf(entropy) + 0.5f;
            support = (unsigned int)rounded;
            if (support < 1) support = 1;
            if (support > count) support = count;
            rank = (unsigned int)(uniforms[row] * (float)support);
            if (rank >= support) rank = support - 1;
        }
    }
    __syncthreads();
    if (invalid) return;
    if (rank == 0) {
        if (tid == 0) {
            output[row] = token_offset + maximum_id;
            output[(unsigned long long)batch + row] = 1;
            output[2ULL * batch + row] = support;
        }
        return;
    }

    // Deterministic descending value / ascending original-index ordering.
    // Only the legal compact region participates, padded to its next power of 2.
    for (unsigned int stage = 2; stage <= sort_width; stage <<= 1) {
        for (unsigned int distance = stage >> 1; distance; distance >>= 1) {
            for (unsigned int col = tid; col < sort_width; col += blockDim.x) {
                const unsigned int other = col ^ distance;
                if (other > col) {
                    const bool descending = (col & stage) == 0;
                    const bool swap = descending
                        ? pp_before(values[other], ids[other], values[col], ids[col])
                        : pp_before(values[col], ids[col], values[other], ids[other]);
                    if (swap) {
                        const float value = values[col]; values[col] = values[other]; values[other] = value;
                        const unsigned int id = ids[col]; ids[col] = ids[other]; ids[other] = id;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        output[row] = token_offset + ids[rank];
        output[(unsigned long long)batch + row] = 1;
        output[2ULL * batch + row] = support;
    }
}
