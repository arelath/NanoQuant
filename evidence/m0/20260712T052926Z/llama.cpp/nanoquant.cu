#include "nanoquant.cuh"

static __device__ __forceinline__ float nanoquant_load_f32(const void * data, const ggml_type type, const int64_t offset) {
    const char * ptr = (const char *) data + offset;

    switch (type) {
        case GGML_TYPE_F32:
            return *(const float *) ptr;
        case GGML_TYPE_F16:
            return __half2float(*(const half *) ptr);
        case GGML_TYPE_BF16:
            return __bfloat162float(*(const nv_bfloat16 *) ptr);
        case GGML_TYPE_I8:
            return (float) *(const int8_t *) ptr;
        default:
            return 0.0f;
    }
}

static __device__ __forceinline__ float nanoquant_apply_sign(
        const int32_t * bits,
        const int64_t   nb0,
        const int64_t   nb1,
        const int64_t   row,
        const int64_t   col,
        const float     value) {
    const int64_t word_idx = col/32;
    const int64_t bit_idx  = col%32;
    const uint32_t word = *(const uint32_t *) ((const char *) bits + word_idx*nb0 + row*nb1);

    return (word & (uint32_t(1) << bit_idx)) ? -value : value;
}

static __device__ __forceinline__ uint32_t nanoquant_load_sign_word(
        const int32_t * bits,
        const int64_t   nb0,
        const int64_t   nb1,
        const int64_t   row,
        const int64_t   word_idx) {
    return *(const uint32_t *) ((const char *) bits + word_idx*nb0 + row*nb1);
}

static __device__ __forceinline__ uint4 nanoquant_load_sign_word4(
        const int32_t * bits,
        const int64_t   nb0,
        const int64_t   nb1,
        const int64_t   row,
        const int64_t   word_idx) {
    // Only used when the row stride and base pointer are 16-byte aligned.
    return *(const uint4 *) ((const char *) bits + word_idx*nb0 + row*nb1);
}

static __device__ __forceinline__ float nanoquant_apply_sign_word(
        const uint32_t word,
        const int      lane,
        const float    value) {
    // Flip the IEEE sign bit instead of branching; this keeps sign application cheap.
    const uint32_t sign = ((word >> lane) & 1u) << 31;
    return __uint_as_float(__float_as_uint(value) ^ sign);
}

static __device__ __forceinline__ float nanoquant_fmaf_sign_word(
        const uint32_t word,
        const int      lane,
        const float    a,
        const float    b,
        const float    c) {
    // Apply the NanoQuant sign to one multiplicand so stage 1 can use a single FMA.
    const uint32_t sign = ((word >> lane) & 1u) << 31;
    return fmaf(a, __uint_as_float(__float_as_uint(b) ^ sign), c);
}

template <int block_size, bool full_warps, bool fast_f32>
static __global__ void nanoquant_stage1_warp_kernel(
        const void * x,
        const ggml_type x_type,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * v_bits,
        const int64_t nbv0,
        const int64_t nbv1,
        const bool vector_sign_loads,
        const void * scale_pre,
        const ggml_type scale_pre_type,
        const int64_t nb_pre0,
        const void * scale_mid,
        const ggml_type scale_mid_type,
        const int64_t nb_mid0,
        float * tmp,
        const int64_t n_in,
        const int64_t n_rank,
        const int64_t ne1,
        const int64_t ne2) {
    static_assert(block_size % WARP_SIZE == 0, "block size must be a whole number of warps");
    constexpr int rows_per_block = block_size / WARP_SIZE;

    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = threadIdx.x / WARP_SIZE;
    const int64_t r = (int64_t) blockIdx.x*rows_per_block + row;
    // The warp kernels are launched only for n_col == 1 decode, so column coordinates are always zero.
    const int64_t c = 0;
    if (r >= n_rank) {
        return;
    }

    const int64_t i1 = 0;
    const int64_t i2 = 0;
    const int64_t i3 = 0;
    (void) ne1;
    (void) ne2;
    (void) i1;
    (void) i2;
    (void) i3;

    float acc = 0.0f;
    if constexpr (fast_f32) {
        // The fast path is selected only for contiguous F32 decode tensors and scales.
        const float * x_f         = (const float *) x;
        const float * scale_pre_f = (const float *) scale_pre;
        if constexpr (full_warps) {
            // Multiple accumulators hide FMA latency better than one long dependency chain.
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            const int64_t n_words = n_in/WARP_SIZE;
            int64_t word_idx = 0;
            for (; word_idx + 3 < n_words; word_idx += 4) {
                // Lane 0 owns the sign-word fetch and broadcasts it; aligned rows can use one 16-byte load.
                uint32_t word0;
                uint32_t word1;
                uint32_t word2;
                uint32_t word3;
                if (vector_sign_loads) {
                    uint4 words = make_uint4(0, 0, 0, 0);
                    if (lane == 0) {
                        words = nanoquant_load_sign_word4(v_bits, nbv0, nbv1, r, word_idx);
                    }
                    word0 = words.x;
                    word1 = words.y;
                    word2 = words.z;
                    word3 = words.w;
                } else {
                    word0 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 0) : 0;
                    word1 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 1) : 0;
                    word2 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 2) : 0;
                    word3 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 3) : 0;
                }
                word0 = __shfl_sync(0xffffffff, word0, 0);
                word1 = __shfl_sync(0xffffffff, word1, 0);
                word2 = __shfl_sync(0xffffffff, word2, 0);
                word3 = __shfl_sync(0xffffffff, word3, 0);

                const int64_t i0 = (word_idx + 0)*WARP_SIZE + lane;
                const int64_t i1 = (word_idx + 1)*WARP_SIZE + lane;
                const int64_t i2 = (word_idx + 2)*WARP_SIZE + lane;
                const int64_t i3 = (word_idx + 3)*WARP_SIZE + lane;
                acc0 = nanoquant_fmaf_sign_word(word0, lane, x_f[i0], scale_pre_f[i0], acc0);
                acc1 = nanoquant_fmaf_sign_word(word1, lane, x_f[i1], scale_pre_f[i1], acc1);
                acc2 = nanoquant_fmaf_sign_word(word2, lane, x_f[i2], scale_pre_f[i2], acc2);
                acc3 = nanoquant_fmaf_sign_word(word3, lane, x_f[i3], scale_pre_f[i3], acc3);
            }
            acc = (acc0 + acc1) + (acc2 + acc3);
            for (; word_idx < n_words; ++word_idx) {
                uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx) : 0;
                word = __shfl_sync(0xffffffff, word, 0);
                const int64_t i = word_idx*WARP_SIZE + lane;
                acc = nanoquant_fmaf_sign_word(word, lane, x_f[i], scale_pre_f[i], acc);
            }
        } else {
            for (int64_t base = 0; base < n_in; base += WARP_SIZE) {
                const int64_t i = base + lane;
                uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, base/WARP_SIZE) : 0;
                word = __shfl_sync(0xffffffff, word, 0);
                if (i >= n_in) {
                    continue;
                }
                acc = nanoquant_fmaf_sign_word(word, lane, x_f[i], scale_pre_f[i], acc);
            }
        }
    } else if (x_type == GGML_TYPE_F32 && scale_pre_type == GGML_TYPE_F32) {
        for (int64_t base = 0; base < n_in; base += WARP_SIZE) {
            const int64_t i = base + lane;
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, base/WARP_SIZE) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            if constexpr (!full_warps) {
                if (i >= n_in) {
                    continue;
                }
            }
            const float xv = *(const float *) ((const char *) x + i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = *(const float *) ((const char *) scale_pre + i*nb_pre0);
            acc = nanoquant_fmaf_sign_word(word, lane, xv, pre, acc);
        }
    } else if (x_type == GGML_TYPE_F32 && scale_pre_type == GGML_TYPE_F16) {
        for (int64_t base = 0; base < n_in; base += WARP_SIZE) {
            const int64_t i = base + lane;
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, base/WARP_SIZE) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            if constexpr (!full_warps) {
                if (i >= n_in) {
                    continue;
                }
            }
            const float xv = *(const float *) ((const char *) x + i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = __half2float(*(const half *) ((const char *) scale_pre + i*nb_pre0));
            acc = nanoquant_fmaf_sign_word(word, lane, xv, pre, acc);
        }
    } else if (x_type == GGML_TYPE_F32 && scale_pre_type == GGML_TYPE_BF16) {
        for (int64_t base = 0; base < n_in; base += WARP_SIZE) {
            const int64_t i = base + lane;
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, base/WARP_SIZE) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            if constexpr (!full_warps) {
                if (i >= n_in) {
                    continue;
                }
            }
            const float xv = *(const float *) ((const char *) x + i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = __bfloat162float(*(const nv_bfloat16 *) ((const char *) scale_pre + i*nb_pre0));
            acc = nanoquant_fmaf_sign_word(word, lane, xv, pre, acc);
        }
    } else {
        for (int64_t base = 0; base < n_in; base += WARP_SIZE) {
            const int64_t i = base + lane;
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, base/WARP_SIZE) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            if constexpr (!full_warps) {
                if (i >= n_in) {
                    continue;
                }
            }
            const float xv = nanoquant_load_f32(x, x_type, i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = nanoquant_load_f32(scale_pre, scale_pre_type, i*nb_pre0);
            acc = nanoquant_fmaf_sign_word(word, lane, xv, pre, acc);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        float mid;
        if constexpr (fast_f32) {
            mid = ((const float *) scale_mid)[r];
        } else {
            mid = nanoquant_load_f32(scale_mid, scale_mid_type, r*nb_mid0);
        }
        tmp[c*n_rank + r] = acc*mid;
    }
}

template <int block_size, bool full_warps, bool fast_f32>
static __global__ void nanoquant_stage1_fused_warp_kernel(
        const void * x,
        const ggml_type x_type,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * v0,
        const int64_t nbv00,
        const int64_t nbv01,
        const bool vector_v0,
        const void * scale_pre0,
        const ggml_type scale_pre0_type,
        const int64_t nb_pre00,
        const void * scale_mid0,
        const ggml_type scale_mid0_type,
        const int64_t nb_mid00,
        const int32_t * v1,
        const int64_t nbv10,
        const int64_t nbv11,
        const bool vector_v1,
        const void * scale_pre1,
        const ggml_type scale_pre1_type,
        const int64_t nb_pre10,
        const void * scale_mid1,
        const ggml_type scale_mid1_type,
        const int64_t nb_mid10,
        const int32_t * v2,
        const int64_t nbv20,
        const int64_t nbv21,
        const bool vector_v2,
        const void * scale_pre2,
        const ggml_type scale_pre2_type,
        const int64_t nb_pre20,
        const void * scale_mid2,
        const ggml_type scale_mid2_type,
        const int64_t nb_mid20,
        float * tmp,
        const int64_t n_in,
        const int64_t n_rank0,
        const int64_t n_rank1,
        const int64_t n_rank2,
        const int64_t ne1,
        const int64_t ne2) {
    static_assert(block_size % WARP_SIZE == 0, "block size must be a whole number of warps");
    constexpr int rows_per_block = block_size / WARP_SIZE;

    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = threadIdx.x / WARP_SIZE;
    const int64_t r_global = (int64_t) blockIdx.x*rows_per_block + row;
    const int64_t n_rank_total = n_rank0 + n_rank1 + n_rank2;
    if (r_global >= n_rank_total) {
        return;
    }

    const int64_t c = 0;
    const int64_t i1 = 0;
    const int64_t i2 = 0;
    const int64_t i3 = 0;
    (void) ne1;
    (void) ne2;
    (void) i1;
    (void) i2;
    (void) i3;

    const int32_t * v_bits;
    int64_t nbv0;
    int64_t nbv1;
    [[maybe_unused]] bool vector_sign_loads;
    const void * scale_pre;
    [[maybe_unused]] ggml_type scale_pre_type;
    [[maybe_unused]] int64_t nb_pre0;
    const void * scale_mid;
    [[maybe_unused]] ggml_type scale_mid_type;
    [[maybe_unused]] int64_t nb_mid0;
    int64_t r;

    if (r_global < n_rank0) {
        v_bits = v0; nbv0 = nbv00; nbv1 = nbv01; vector_sign_loads = vector_v0;
        scale_pre = scale_pre0; scale_pre_type = scale_pre0_type; nb_pre0 = nb_pre00;
        scale_mid = scale_mid0; scale_mid_type = scale_mid0_type; nb_mid0 = nb_mid00;
        r = r_global;
    } else if (r_global < n_rank0 + n_rank1) {
        v_bits = v1; nbv0 = nbv10; nbv1 = nbv11; vector_sign_loads = vector_v1;
        scale_pre = scale_pre1; scale_pre_type = scale_pre1_type; nb_pre0 = nb_pre10;
        scale_mid = scale_mid1; scale_mid_type = scale_mid1_type; nb_mid0 = nb_mid10;
        r = r_global - n_rank0;
    } else {
        v_bits = v2; nbv0 = nbv20; nbv1 = nbv21; vector_sign_loads = vector_v2;
        scale_pre = scale_pre2; scale_pre_type = scale_pre2_type; nb_pre0 = nb_pre20;
        scale_mid = scale_mid2; scale_mid_type = scale_mid2_type; nb_mid0 = nb_mid20;
        r = r_global - n_rank0 - n_rank1;
    }

    float acc = 0.0f;
    if constexpr (fast_f32) {
        const float * x_f = (const float *) x;
        const float * scale_pre_f = (const float *) scale_pre;
        if constexpr (full_warps) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            const int64_t n_words = n_in/WARP_SIZE;
            int64_t word_idx = 0;
            for (; word_idx + 3 < n_words; word_idx += 4) {
                uint32_t word0;
                uint32_t word1;
                uint32_t word2;
                uint32_t word3;
                if (vector_sign_loads) {
                    uint4 words = make_uint4(0, 0, 0, 0);
                    if (lane == 0) {
                        words = nanoquant_load_sign_word4(v_bits, nbv0, nbv1, r, word_idx);
                    }
                    word0 = words.x;
                    word1 = words.y;
                    word2 = words.z;
                    word3 = words.w;
                } else {
                    word0 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 0) : 0;
                    word1 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 1) : 0;
                    word2 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 2) : 0;
                    word3 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 3) : 0;
                }
                word0 = __shfl_sync(0xffffffff, word0, 0);
                word1 = __shfl_sync(0xffffffff, word1, 0);
                word2 = __shfl_sync(0xffffffff, word2, 0);
                word3 = __shfl_sync(0xffffffff, word3, 0);

                const int64_t ii0 = (word_idx + 0)*WARP_SIZE + lane;
                const int64_t ii1 = (word_idx + 1)*WARP_SIZE + lane;
                const int64_t ii2 = (word_idx + 2)*WARP_SIZE + lane;
                const int64_t ii3 = (word_idx + 3)*WARP_SIZE + lane;
                acc0 = nanoquant_fmaf_sign_word(word0, lane, x_f[ii0], scale_pre_f[ii0], acc0);
                acc1 = nanoquant_fmaf_sign_word(word1, lane, x_f[ii1], scale_pre_f[ii1], acc1);
                acc2 = nanoquant_fmaf_sign_word(word2, lane, x_f[ii2], scale_pre_f[ii2], acc2);
                acc3 = nanoquant_fmaf_sign_word(word3, lane, x_f[ii3], scale_pre_f[ii3], acc3);
            }
            acc = (acc0 + acc1) + (acc2 + acc3);
            for (; word_idx < n_words; ++word_idx) {
                uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx) : 0;
                word = __shfl_sync(0xffffffff, word, 0);
                const int64_t i = word_idx*WARP_SIZE + lane;
                acc = nanoquant_fmaf_sign_word(word, lane, x_f[i], scale_pre_f[i], acc);
            }
        } else {
            for (int64_t base = 0; base < n_in; base += WARP_SIZE) {
                const int64_t i = base + lane;
                uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, base/WARP_SIZE) : 0;
                word = __shfl_sync(0xffffffff, word, 0);
                if (i >= n_in) {
                    continue;
                }
                acc = nanoquant_fmaf_sign_word(word, lane, x_f[i], scale_pre_f[i], acc);
            }
        }
    } else {
        for (int64_t base = 0; base < n_in; base += WARP_SIZE) {
            const int64_t i = base + lane;
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, base/WARP_SIZE) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            if constexpr (!full_warps) {
                if (i >= n_in) {
                    continue;
                }
            }
            const float xv = nanoquant_load_f32(x, x_type, i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = nanoquant_load_f32(scale_pre, scale_pre_type, i*nb_pre0);
            acc = nanoquant_fmaf_sign_word(word, lane, xv, pre, acc);
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        float mid;
        if constexpr (fast_f32) {
            mid = ((const float *) scale_mid)[r];
        } else {
            mid = nanoquant_load_f32(scale_mid, scale_mid_type, r*nb_mid0);
        }
        tmp[c*n_rank_total + r_global] = acc*mid;
    }
}

template <int block_size, int cols_per_block>
static __global__ void nanoquant_stage1_coltile_kernel(
        const void * x,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * v_bits,
        const int64_t nbv0,
        const int64_t nbv1,
        const bool vector_sign_loads,
        const float * scale_pre,
        const float * scale_mid,
        float * tmp,
        const int64_t n_in,
        const int64_t n_rank,
        const int64_t n_col,
        const int64_t ne1,
        const int64_t ne2) {
    static_assert(block_size == cols_per_block*WARP_SIZE, "one warp per token column");

    // Prefill shapes are too small to fill the GPU with one full block per dot.
    // This packs several token columns into a block while keeping one warp per dot product.
    const int lane = threadIdx.x % WARP_SIZE;
    const int col  = threadIdx.x / WARP_SIZE;
    const int64_t r = blockIdx.x;
    const int64_t c = (int64_t) blockIdx.y*cols_per_block + col;
    if (c >= n_col) {
        return;
    }

    const int64_t i3 = c/(ne1*ne2);
    const int64_t rem = c - i3*ne1*ne2;
    const int64_t i2 = rem/ne1;
    const int64_t i1 = rem - i2*ne1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const int64_t n_words = n_in/WARP_SIZE;
    int64_t word_idx = 0;
    for (; word_idx + 3 < n_words; word_idx += 4) {
        uint32_t word0;
        uint32_t word1;
        uint32_t word2;
        uint32_t word3;
        if (vector_sign_loads) {
            uint4 words = make_uint4(0, 0, 0, 0);
            if (lane == 0) {
                words = nanoquant_load_sign_word4(v_bits, nbv0, nbv1, r, word_idx);
            }
            word0 = words.x;
            word1 = words.y;
            word2 = words.z;
            word3 = words.w;
        } else {
            word0 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 0) : 0;
            word1 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 1) : 0;
            word2 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 2) : 0;
            word3 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 3) : 0;
        }
        word0 = __shfl_sync(0xffffffff, word0, 0);
        word1 = __shfl_sync(0xffffffff, word1, 0);
        word2 = __shfl_sync(0xffffffff, word2, 0);
        word3 = __shfl_sync(0xffffffff, word3, 0);

        const int64_t i0 = (word_idx + 0)*WARP_SIZE + lane;
        const int64_t i1_in = (word_idx + 1)*WARP_SIZE + lane;
        const int64_t i2_in = (word_idx + 2)*WARP_SIZE + lane;
        const int64_t i3_in = (word_idx + 3)*WARP_SIZE + lane;
        const float x0 = *(const float *) ((const char *) x + i0*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
        const float x1 = *(const float *) ((const char *) x + i1_in*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
        const float x2 = *(const float *) ((const char *) x + i2_in*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
        const float x3 = *(const float *) ((const char *) x + i3_in*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
        acc0 = nanoquant_fmaf_sign_word(word0, lane, x0, scale_pre[i0], acc0);
        acc1 = nanoquant_fmaf_sign_word(word1, lane, x1, scale_pre[i1_in], acc1);
        acc2 = nanoquant_fmaf_sign_word(word2, lane, x2, scale_pre[i2_in], acc2);
        acc3 = nanoquant_fmaf_sign_word(word3, lane, x3, scale_pre[i3_in], acc3);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (; word_idx < n_words; ++word_idx) {
        uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx) : 0;
        word = __shfl_sync(0xffffffff, word, 0);
        const int64_t i = word_idx*WARP_SIZE + lane;
        const float xv = *(const float *) ((const char *) x + i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
        acc = nanoquant_fmaf_sign_word(word, lane, xv, scale_pre[i], acc);
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        tmp[c*n_rank + r] = acc*scale_mid[r];
    }
}

template <int block_size, int cols_per_block>
static __global__ void nanoquant_stage1_fused_coltile_kernel(
        const void * x,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * v0,
        const int64_t nbv00,
        const int64_t nbv01,
        const bool vector_v0,
        const float * scale_pre0,
        const float * scale_mid0,
        const int32_t * v1,
        const int64_t nbv10,
        const int64_t nbv11,
        const bool vector_v1,
        const float * scale_pre1,
        const float * scale_mid1,
        const int32_t * v2,
        const int64_t nbv20,
        const int64_t nbv21,
        const bool vector_v2,
        const float * scale_pre2,
        const float * scale_mid2,
        float * tmp,
        const int64_t n_in,
        const int64_t n_rank0,
        const int64_t n_rank1,
        const int64_t n_rank2,
        const int64_t n_col,
        const int64_t ne1,
        const int64_t ne2) {
    static_assert(block_size == cols_per_block*WARP_SIZE, "one warp per token column");

    // Same column-tiled prefill strategy as the single-projection kernel, but
    // with Q/K/V or gate/up ranks concatenated in tmp.
    const int lane = threadIdx.x % WARP_SIZE;
    const int col  = threadIdx.x / WARP_SIZE;
    const int64_t r_global = blockIdx.x;
    const int64_t c = (int64_t) blockIdx.y*cols_per_block + col;
    const int64_t n_rank_total = n_rank0 + n_rank1 + n_rank2;
    if (r_global >= n_rank_total || c >= n_col) {
        return;
    }

    const int64_t i3 = c/(ne1*ne2);
    const int64_t rem = c - i3*ne1*ne2;
    const int64_t i2 = rem/ne1;
    const int64_t i1 = rem - i2*ne1;

    const int32_t * v_bits;
    int64_t nbv0;
    int64_t nbv1;
    bool vector_sign_loads;
    const float * scale_pre;
    const float * scale_mid;
    int64_t r;

    if (r_global < n_rank0) {
        v_bits = v0; nbv0 = nbv00; nbv1 = nbv01; vector_sign_loads = vector_v0;
        scale_pre = scale_pre0; scale_mid = scale_mid0; r = r_global;
    } else if (r_global < n_rank0 + n_rank1) {
        v_bits = v1; nbv0 = nbv10; nbv1 = nbv11; vector_sign_loads = vector_v1;
        scale_pre = scale_pre1; scale_mid = scale_mid1; r = r_global - n_rank0;
    } else {
        v_bits = v2; nbv0 = nbv20; nbv1 = nbv21; vector_sign_loads = vector_v2;
        scale_pre = scale_pre2; scale_mid = scale_mid2; r = r_global - n_rank0 - n_rank1;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const int64_t n_words = n_in/WARP_SIZE;
    int64_t word_idx = 0;
    for (; word_idx + 3 < n_words; word_idx += 4) {
        uint32_t word0;
        uint32_t word1;
        uint32_t word2;
        uint32_t word3;
        if (vector_sign_loads) {
            uint4 words = make_uint4(0, 0, 0, 0);
            if (lane == 0) {
                words = nanoquant_load_sign_word4(v_bits, nbv0, nbv1, r, word_idx);
            }
            word0 = words.x;
            word1 = words.y;
            word2 = words.z;
            word3 = words.w;
        } else {
            word0 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 0) : 0;
            word1 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 1) : 0;
            word2 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 2) : 0;
            word3 = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx + 3) : 0;
        }
        word0 = __shfl_sync(0xffffffff, word0, 0);
        word1 = __shfl_sync(0xffffffff, word1, 0);
        word2 = __shfl_sync(0xffffffff, word2, 0);
        word3 = __shfl_sync(0xffffffff, word3, 0);

        const int64_t i0 = (word_idx + 0)*WARP_SIZE + lane;
        const int64_t i1_in = (word_idx + 1)*WARP_SIZE + lane;
        const int64_t i2_in = (word_idx + 2)*WARP_SIZE + lane;
        const int64_t i3_in = (word_idx + 3)*WARP_SIZE + lane;
        const float x0 = *(const float *) ((const char *) x + i0*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
        const float x1 = *(const float *) ((const char *) x + i1_in*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
        const float x2 = *(const float *) ((const char *) x + i2_in*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
        const float x3 = *(const float *) ((const char *) x + i3_in*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
        acc0 = nanoquant_fmaf_sign_word(word0, lane, x0, scale_pre[i0], acc0);
        acc1 = nanoquant_fmaf_sign_word(word1, lane, x1, scale_pre[i1_in], acc1);
        acc2 = nanoquant_fmaf_sign_word(word2, lane, x2, scale_pre[i2_in], acc2);
        acc3 = nanoquant_fmaf_sign_word(word3, lane, x3, scale_pre[i3_in], acc3);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (; word_idx < n_words; ++word_idx) {
        uint32_t word = lane == 0 ? nanoquant_load_sign_word(v_bits, nbv0, nbv1, r, word_idx) : 0;
        word = __shfl_sync(0xffffffff, word, 0);
        const int64_t i = word_idx*WARP_SIZE + lane;
        const float xv = *(const float *) ((const char *) x + i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
        acc = nanoquant_fmaf_sign_word(word, lane, xv, scale_pre[i], acc);
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        tmp[c*n_rank_total + r_global] = acc*scale_mid[r];
    }
}

template <int block_size>
static __global__ void nanoquant_stage1_fused_block_kernel(
        const void * x,
        const ggml_type x_type,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * v0,
        const int64_t nbv00,
        const int64_t nbv01,
        const void * scale_pre0,
        const ggml_type scale_pre0_type,
        const int64_t nb_pre00,
        const void * scale_mid0,
        const ggml_type scale_mid0_type,
        const int64_t nb_mid00,
        const int32_t * v1,
        const int64_t nbv10,
        const int64_t nbv11,
        const void * scale_pre1,
        const ggml_type scale_pre1_type,
        const int64_t nb_pre10,
        const void * scale_mid1,
        const ggml_type scale_mid1_type,
        const int64_t nb_mid10,
        const int32_t * v2,
        const int64_t nbv20,
        const int64_t nbv21,
        const void * scale_pre2,
        const ggml_type scale_pre2_type,
        const int64_t nb_pre20,
        const void * scale_mid2,
        const ggml_type scale_mid2_type,
        const int64_t nb_mid20,
        float * tmp,
        const int64_t n_in,
        const int64_t n_rank0,
        const int64_t n_rank1,
        const int64_t n_rank2,
        const int64_t ne1,
        const int64_t ne2) {
    const int64_t r_global = blockIdx.x;
    const int64_t c = blockIdx.y;
    const int tid = threadIdx.x;
    const int64_t n_rank_total = n_rank0 + n_rank1 + n_rank2;

    const int32_t * v_bits;
    int64_t nbv0;
    int64_t nbv1;
    const void * scale_pre;
    ggml_type scale_pre_type;
    int64_t nb_pre0;
    const void * scale_mid;
    ggml_type scale_mid_type;
    int64_t nb_mid0;
    int64_t r;

    if (r_global < n_rank0) {
        v_bits = v0; nbv0 = nbv00; nbv1 = nbv01;
        scale_pre = scale_pre0; scale_pre_type = scale_pre0_type; nb_pre0 = nb_pre00;
        scale_mid = scale_mid0; scale_mid_type = scale_mid0_type; nb_mid0 = nb_mid00;
        r = r_global;
    } else if (r_global < n_rank0 + n_rank1) {
        v_bits = v1; nbv0 = nbv10; nbv1 = nbv11;
        scale_pre = scale_pre1; scale_pre_type = scale_pre1_type; nb_pre0 = nb_pre10;
        scale_mid = scale_mid1; scale_mid_type = scale_mid1_type; nb_mid0 = nb_mid10;
        r = r_global - n_rank0;
    } else {
        v_bits = v2; nbv0 = nbv20; nbv1 = nbv21;
        scale_pre = scale_pre2; scale_pre_type = scale_pre2_type; nb_pre0 = nb_pre20;
        scale_mid = scale_mid2; scale_mid_type = scale_mid2_type; nb_mid0 = nb_mid20;
        r = r_global - n_rank0 - n_rank1;
    }

    const int64_t i3 = c/(ne1*ne2);
    const int64_t rem = c - i3*ne1*ne2;
    const int64_t i2 = rem/ne1;
    const int64_t i1 = rem - i2*ne1;

    float acc = 0.0f;
    for (int64_t i = tid; i < n_in; i += block_size) {
        const float xv = nanoquant_load_f32(x, x_type, i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
        const float pre = nanoquant_load_f32(scale_pre, scale_pre_type, i*nb_pre0);
        acc += nanoquant_apply_sign(v_bits, nbv0, nbv1, r, i, xv*pre);
    }

    extern __shared__ float shared[];
    acc = block_reduce<block_reduce_method::SUM, block_size>(acc, shared);
    if (tid == 0) {
        const float mid = nanoquant_load_f32(scale_mid, scale_mid_type, r*nb_mid0);
        tmp[c*n_rank_total + r_global] = acc*mid;
    }
}

template <int block_size, bool full_warps, int fast_salient, bool no_salient>
static __global__ void nanoquant_stage2_warp_kernel(
        const void * x,
        const ggml_type x_type,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * u_bits,
        const int64_t nbu0,
        const int64_t nbu1,
        const bool vector_sign_loads,
        const void * scale_post,
        const ggml_type scale_post_type,
        const int64_t nb_post0,
        const int32_t * salient_idx,
        const int64_t nb_salient_idx0,
        const void * salient_weight,
        const ggml_type salient_weight_type,
        const int64_t nb_salient_weight0,
        const int64_t nb_salient_weight1,
        const void * salient_scale,
        const ggml_type salient_scale_type,
        const int64_t nb_salient_scale0,
        const int64_t n_salient,
        const float * tmp,
        float * dst,
        const int64_t n_rank,
        const int64_t n_out,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t nbd0,
        const int64_t nbd1,
        const int64_t nbd2,
        const int64_t nbd3) {
    static_assert(block_size % WARP_SIZE == 0, "block size must be a whole number of warps");
    constexpr int rows_per_block = block_size / WARP_SIZE;

    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = threadIdx.x / WARP_SIZE;
    const int64_t o = (int64_t) blockIdx.x*rows_per_block + row;
    // The warp kernels are launched only for n_col == 1 decode, so column coordinates are always zero.
    const int64_t c = 0;
    if (o >= n_out) {
        return;
    }

    const int64_t i1 = 0;
    const int64_t i2 = 0;
    const int64_t i3 = 0;
    (void) ne1;
    (void) ne2;
    if constexpr (fast_salient >= 0) {
        (void) i1;
        (void) i2;
        (void) i3;
    }

    float acc = 0.0f;
    if constexpr (full_warps) {
        // Multiple accumulators hide add latency in the rank reduction. Shared-memory staging of tmp was slower.
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        const int64_t n_words = n_rank/WARP_SIZE;
        int64_t word_idx = 0;
        for (; word_idx + 3 < n_words; word_idx += 4) {
            // Lane 0 sign-word loads plus shuffles measured faster than cooperative word loading here.
            uint32_t word0;
            uint32_t word1;
            uint32_t word2;
            uint32_t word3;
            if (vector_sign_loads) {
                uint4 words = make_uint4(0, 0, 0, 0);
                if (lane == 0) {
                    words = nanoquant_load_sign_word4(u_bits, nbu0, nbu1, o, word_idx);
                }
                word0 = words.x;
                word1 = words.y;
                word2 = words.z;
                word3 = words.w;
            } else {
                word0 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 0) : 0;
                word1 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 1) : 0;
                word2 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 2) : 0;
                word3 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 3) : 0;
            }
            word0 = __shfl_sync(0xffffffff, word0, 0);
            word1 = __shfl_sync(0xffffffff, word1, 0);
            word2 = __shfl_sync(0xffffffff, word2, 0);
            word3 = __shfl_sync(0xffffffff, word3, 0);

            const int64_t r0 = (word_idx + 0)*WARP_SIZE + lane;
            const int64_t r1 = (word_idx + 1)*WARP_SIZE + lane;
            const int64_t r2 = (word_idx + 2)*WARP_SIZE + lane;
            const int64_t r3 = (word_idx + 3)*WARP_SIZE + lane;
            acc0 += nanoquant_apply_sign_word(word0, lane, tmp[c*n_rank + r0]);
            acc1 += nanoquant_apply_sign_word(word1, lane, tmp[c*n_rank + r1]);
            acc2 += nanoquant_apply_sign_word(word2, lane, tmp[c*n_rank + r2]);
            acc3 += nanoquant_apply_sign_word(word3, lane, tmp[c*n_rank + r3]);
        }
        acc = (acc0 + acc1) + (acc2 + acc3);
        for (; word_idx < n_words; ++word_idx) {
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            const int64_t r = word_idx*WARP_SIZE + lane;
            acc += nanoquant_apply_sign_word(word, lane, tmp[c*n_rank + r]);
        }
    } else {
        for (int64_t base = 0; base < n_rank; base += WARP_SIZE) {
            const int64_t r = base + lane;
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, base/WARP_SIZE) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            if (r >= n_rank) {
                continue;
            }
            acc += nanoquant_apply_sign_word(word, lane, tmp[c*n_rank + r]);
        }
    }

    float salient_acc = 0.0f;
    if constexpr (no_salient) {
    } else if constexpr (fast_salient >= 0) {
        // Fast salient paths require contiguous F16 weights in [out, salient] order.
        const int32_t * salient_idx_i = (const int32_t *) salient_idx;
        const half * salient_weight_h = (const half *) salient_weight;
        const float * x_f = (const float *) x;
        if (lane == 0) {
            for (int s = 0; s < fast_salient; ++s) {
                const int64_t idx = salient_idx_i[s];
                const float weight = __half2float(salient_weight_h[o*fast_salient + s]);
                salient_acc += x_f[idx]*weight;
            }
        }
    } else {
        for (int64_t s = lane; s < n_salient; s += WARP_SIZE) {
            const int64_t idx = *(const int32_t *) ((const char *) salient_idx + s*nb_salient_idx0);
            float weight = nanoquant_load_f32(
                salient_weight, salient_weight_type, s*nb_salient_weight0 + o*nb_salient_weight1);
            if (salient_scale != nullptr) {
                weight *= nanoquant_load_f32(salient_scale, salient_scale_type, s*nb_salient_scale0);
            }
            salient_acc += nanoquant_load_f32(x, x_type, idx*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3)*weight;
        }
    }

    acc = warp_reduce_sum(acc);
    if constexpr (!no_salient && fast_salient < 0) {
        salient_acc = warp_reduce_sum(salient_acc);
    }
    if (lane == 0) {
        float post;
        if constexpr (fast_salient >= 0) {
            post = ((const float *) scale_post)[o];
        } else {
            post = nanoquant_load_f32(scale_post, scale_post_type, o*nb_post0);
        }
        const float value = no_salient ? acc*post : acc*post + salient_acc;

        if constexpr (fast_salient >= 0) {
            dst[o] = value;
        } else {
            *(float *) ((char *) dst + o*nbd0 + i1*nbd1 + i2*nbd2 + i3*nbd3) = value;
        }
    }
}

template <int block_size, bool full_warps, int fast_salient0, int fast_salient1>
static __global__ void nanoquant_stage2_fused2_warp_kernel(
        const void * x,
        const ggml_type x_type,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const float * tmp,
        const int32_t * u0,
        const int64_t nbu00,
        const int64_t nbu01,
        const bool vector_u0,
        const void * scale_post0,
        const ggml_type scale_post0_type,
        const int64_t nb_post00,
        const int32_t * salient_idx0,
        const int64_t nb_salient_idx00,
        const void * salient_weight0,
        const ggml_type salient_weight0_type,
        const int64_t nb_salient_weight00,
        const int64_t nb_salient_weight01,
        const int32_t * u1,
        const int64_t nbu10,
        const int64_t nbu11,
        const bool vector_u1,
        const void * scale_post1,
        const ggml_type scale_post1_type,
        const int64_t nb_post10,
        const int32_t * salient_idx1,
        const int64_t nb_salient_idx10,
        const void * salient_weight1,
        const ggml_type salient_weight1_type,
        const int64_t nb_salient_weight10,
        const int64_t nb_salient_weight11,
        float * dst,
        const int64_t n_rank0,
        const int64_t n_rank1,
        const int64_t n_out0,
        const int64_t n_out_total,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t nbd0,
        const int64_t nbd1,
        const int64_t nbd2,
        const int64_t nbd3) {
    static_assert(block_size % WARP_SIZE == 0, "block size must be a whole number of warps");
    constexpr int rows_per_block = block_size / WARP_SIZE;

    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = threadIdx.x / WARP_SIZE;
    const int64_t o_global = (int64_t) blockIdx.x*rows_per_block + row;
    if (o_global >= n_out_total) {
        return;
    }

    const bool group1 = o_global >= n_out0;
    const int64_t o = group1 ? o_global - n_out0 : o_global;
    const int64_t n_rank = group1 ? n_rank1 : n_rank0;
    const int64_t rank_offset = group1 ? n_rank0 : 0;
    const int32_t * u_bits = group1 ? u1 : u0;
    const int64_t nbu0 = group1 ? nbu10 : nbu00;
    const int64_t nbu1 = group1 ? nbu11 : nbu01;
    const bool vector_sign_loads = group1 ? vector_u1 : vector_u0;

    const int64_t c = 0;
    const int64_t i1 = 0;
    const int64_t i2 = 0;
    const int64_t i3 = 0;
    (void) ne1;
    (void) ne2;
    (void) c;
    (void) i1;
    (void) i2;
    (void) i3;

    float acc = 0.0f;
    if constexpr (full_warps) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        const int64_t n_words = n_rank/WARP_SIZE;
        int64_t word_idx = 0;
        for (; word_idx + 3 < n_words; word_idx += 4) {
            uint32_t word0;
            uint32_t word1;
            uint32_t word2;
            uint32_t word3;
            if (vector_sign_loads) {
                uint4 words = make_uint4(0, 0, 0, 0);
                if (lane == 0) {
                    words = nanoquant_load_sign_word4(u_bits, nbu0, nbu1, o, word_idx);
                }
                word0 = words.x;
                word1 = words.y;
                word2 = words.z;
                word3 = words.w;
            } else {
                word0 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 0) : 0;
                word1 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 1) : 0;
                word2 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 2) : 0;
                word3 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 3) : 0;
            }
            word0 = __shfl_sync(0xffffffff, word0, 0);
            word1 = __shfl_sync(0xffffffff, word1, 0);
            word2 = __shfl_sync(0xffffffff, word2, 0);
            word3 = __shfl_sync(0xffffffff, word3, 0);

            const int64_t r0 = rank_offset + (word_idx + 0)*WARP_SIZE + lane;
            const int64_t r1 = rank_offset + (word_idx + 1)*WARP_SIZE + lane;
            const int64_t r2 = rank_offset + (word_idx + 2)*WARP_SIZE + lane;
            const int64_t r3 = rank_offset + (word_idx + 3)*WARP_SIZE + lane;
            acc0 += nanoquant_apply_sign_word(word0, lane, tmp[r0]);
            acc1 += nanoquant_apply_sign_word(word1, lane, tmp[r1]);
            acc2 += nanoquant_apply_sign_word(word2, lane, tmp[r2]);
            acc3 += nanoquant_apply_sign_word(word3, lane, tmp[r3]);
        }
        acc = (acc0 + acc1) + (acc2 + acc3);
        for (; word_idx < n_words; ++word_idx) {
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            const int64_t r = rank_offset + word_idx*WARP_SIZE + lane;
            acc += nanoquant_apply_sign_word(word, lane, tmp[r]);
        }
    } else {
        for (int64_t base = 0; base < n_rank; base += WARP_SIZE) {
            const int64_t r = base + lane;
            uint32_t word = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, base/WARP_SIZE) : 0;
            word = __shfl_sync(0xffffffff, word, 0);
            if (r >= n_rank) {
                continue;
            }
            acc += nanoquant_apply_sign_word(word, lane, tmp[rank_offset + r]);
        }
    }

    float salient_acc = 0.0f;
    if (lane == 0) {
        const int fast_salient = group1 ? fast_salient1 : fast_salient0;
        const int32_t * salient_idx = group1 ? salient_idx1 : salient_idx0;
        const void * salient_weight = group1 ? salient_weight1 : salient_weight0;
        if (fast_salient > 0) {
            const half * salient_weight_h = (const half *) salient_weight;
            const float * x_f = (const float *) x;
            for (int s = 0; s < fast_salient; ++s) {
                const int64_t idx = salient_idx[s];
                const float weight = __half2float(salient_weight_h[o*fast_salient + s]);
                salient_acc += x_f[idx]*weight;
            }
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        const void * scale_post = group1 ? scale_post1 : scale_post0;
        const ggml_type scale_post_type = group1 ? scale_post1_type : scale_post0_type;
        const int64_t nb_post0 = group1 ? nb_post10 : nb_post00;
        const float post = nanoquant_load_f32(scale_post, scale_post_type, o*nb_post0);
        *(float *) ((char *) dst + o_global*nbd0) = acc*post + salient_acc;
    }
}

template <int block_size>
static __global__ void nanoquant_stage1_block_kernel(
        const void * x,
        const ggml_type x_type,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * v_bits,
        const int64_t nbv0,
        const int64_t nbv1,
        const void * scale_pre,
        const ggml_type scale_pre_type,
        const int64_t nb_pre0,
        const void * scale_mid,
        const ggml_type scale_mid_type,
        const int64_t nb_mid0,
        float * tmp,
        const int64_t n_in,
        const int64_t n_rank,
        const int64_t ne1,
        const int64_t ne2) {
    const int64_t r = blockIdx.x;
    const int64_t c = blockIdx.y;
    const int tid = threadIdx.x;

    const int64_t i3 = c/(ne1*ne2);
    const int64_t rem = c - i3*ne1*ne2;
    const int64_t i2 = rem/ne1;
    const int64_t i1 = rem - i2*ne1;

    float acc = 0.0f;
    if (x_type == GGML_TYPE_F32 && scale_pre_type == GGML_TYPE_F32) {
        for (int64_t i = tid; i < n_in; i += block_size) {
            const float xv = *(const float *) ((const char *) x + i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = *(const float *) ((const char *) scale_pre + i*nb_pre0);
            acc += nanoquant_apply_sign(v_bits, nbv0, nbv1, r, i, xv*pre);
        }
    } else {
        for (int64_t i = tid; i < n_in; i += block_size) {
            const float xv = nanoquant_load_f32(x, x_type, i*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            const float pre = nanoquant_load_f32(scale_pre, scale_pre_type, i*nb_pre0);
            acc += nanoquant_apply_sign(v_bits, nbv0, nbv1, r, i, xv*pre);
        }
    }

    extern __shared__ float shared[];
    acc = block_reduce<block_reduce_method::SUM, block_size>(acc, shared);
    if (tid == 0) {
        const float mid = nanoquant_load_f32(scale_mid, scale_mid_type, r*nb_mid0);
        tmp[c*n_rank + r] = acc*mid;
    }
}

template <int block_size>
static __global__ void nanoquant_stage2_block_kernel(
        const void * x,
        const ggml_type x_type,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * u_bits,
        const int64_t nbu0,
        const int64_t nbu1,
        const void * scale_post,
        const ggml_type scale_post_type,
        const int64_t nb_post0,
        const int32_t * salient_idx,
        const int64_t nb_salient_idx0,
        const void * salient_weight,
        const ggml_type salient_weight_type,
        const int64_t nb_salient_weight0,
        const int64_t nb_salient_weight1,
        const void * salient_scale,
        const ggml_type salient_scale_type,
        const int64_t nb_salient_scale0,
        const int64_t n_salient,
        const float * tmp,
        const int64_t nbtmp0,
        const int64_t nbtmp1,
        const int64_t nbtmp2,
        const int64_t nbtmp3,
        float * dst,
        const int64_t n_rank,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t nbd0,
        const int64_t nbd1,
        const int64_t nbd2,
        const int64_t nbd3) {
    // Generic fallback for uncommon shapes and scalar types.  It is slower than
    // the specialized paths but handles arbitrary strides and salient formats.
    const int64_t o = blockIdx.x;
    const int64_t c = blockIdx.y;
    const int tid = threadIdx.x;

    const int64_t i3 = c/(ne1*ne2);
    const int64_t rem = c - i3*ne1*ne2;
    const int64_t i2 = rem/ne1;
    const int64_t i1 = rem - i2*ne1;

    float acc = 0.0f;
    for (int64_t r = tid; r < n_rank; r += block_size) {
        const float tmp_v = *(const float *) ((const char *) tmp + r*nbtmp0 + i1*nbtmp1 + i2*nbtmp2 + i3*nbtmp3);
        acc += nanoquant_apply_sign(u_bits, nbu0, nbu1, o, r, tmp_v);
    }

    extern __shared__ float shared[];
    acc = block_reduce<block_reduce_method::SUM, block_size>(acc, shared);
    if (tid == 0) {
        float value = acc*nanoquant_load_f32(scale_post, scale_post_type, o*nb_post0);

        for (int64_t s = 0; s < n_salient; ++s) {
            const int64_t idx = *(const int32_t *) ((const char *) salient_idx + s*nb_salient_idx0);
            float weight = nanoquant_load_f32(
                salient_weight, salient_weight_type, s*nb_salient_weight0 + o*nb_salient_weight1);
            if (salient_scale != nullptr) {
                weight *= nanoquant_load_f32(salient_scale, salient_scale_type, s*nb_salient_scale0);
            }
            value += nanoquant_load_f32(x, x_type, idx*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3)*weight;
        }

        *(float *) ((char *) dst + o*nbd0 + i1*nbd1 + i2*nbd2 + i3*nbd3) = value;
    }
}

template <int block_size, int cols_per_block, int fast_salient>
static __global__ void nanoquant_stage2_coltile_kernel(
        const void * x,
        const int64_t nbx0,
        const int64_t nbx1,
        const int64_t nbx2,
        const int64_t nbx3,
        const int32_t * u_bits,
        const int64_t nbu0,
        const int64_t nbu1,
        const bool vector_sign_loads,
        const float * scale_post,
        const int32_t * salient_idx,
        const half * salient_weight,
        const float * tmp,
        const int64_t nbtmp0,
        const int64_t nbtmp1,
        const int64_t nbtmp2,
        const int64_t nbtmp3,
        float * dst,
        const int64_t n_rank,
        const int64_t n_col,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t nbd0,
        const int64_t nbd1,
        const int64_t nbd2,
        const int64_t nbd3) {
    static_assert(block_size == cols_per_block*WARP_SIZE, "one warp per token column");

    // Fast prefill path for the current GGUF layout: F32 activation/tmp/post
    // tensors and contiguous F16 salient weights in [out, salient] order.
    const int lane = threadIdx.x % WARP_SIZE;
    const int col  = threadIdx.x / WARP_SIZE;
    const int64_t o = blockIdx.x;
    const int64_t c = (int64_t) blockIdx.y*cols_per_block + col;
    if (c >= n_col) {
        return;
    }

    const int64_t i3 = c/(ne1*ne2);
    const int64_t rem = c - i3*ne1*ne2;
    const int64_t i2 = rem/ne1;
    const int64_t i1 = rem - i2*ne1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const int64_t n_words = n_rank/WARP_SIZE;
    int64_t word_idx = 0;
    for (; word_idx + 3 < n_words; word_idx += 4) {
        uint32_t word0;
        uint32_t word1;
        uint32_t word2;
        uint32_t word3;
        if (vector_sign_loads) {
            uint4 words = make_uint4(0, 0, 0, 0);
            if (lane == 0) {
                words = nanoquant_load_sign_word4(u_bits, nbu0, nbu1, o, word_idx);
            }
            word0 = words.x;
            word1 = words.y;
            word2 = words.z;
            word3 = words.w;
        } else {
            word0 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 0) : 0;
            word1 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 1) : 0;
            word2 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 2) : 0;
            word3 = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx + 3) : 0;
        }
        word0 = __shfl_sync(0xffffffff, word0, 0);
        word1 = __shfl_sync(0xffffffff, word1, 0);
        word2 = __shfl_sync(0xffffffff, word2, 0);
        word3 = __shfl_sync(0xffffffff, word3, 0);

        const int64_t r0 = (word_idx + 0)*WARP_SIZE + lane;
        const int64_t r1 = (word_idx + 1)*WARP_SIZE + lane;
        const int64_t r2 = (word_idx + 2)*WARP_SIZE + lane;
        const int64_t r3 = (word_idx + 3)*WARP_SIZE + lane;
        const float tmp0 = *(const float *) ((const char *) tmp + r0*nbtmp0 + i1*nbtmp1 + i2*nbtmp2 + i3*nbtmp3);
        const float tmp1 = *(const float *) ((const char *) tmp + r1*nbtmp0 + i1*nbtmp1 + i2*nbtmp2 + i3*nbtmp3);
        const float tmp2 = *(const float *) ((const char *) tmp + r2*nbtmp0 + i1*nbtmp1 + i2*nbtmp2 + i3*nbtmp3);
        const float tmp3 = *(const float *) ((const char *) tmp + r3*nbtmp0 + i1*nbtmp1 + i2*nbtmp2 + i3*nbtmp3);
        acc0 += nanoquant_apply_sign_word(word0, lane, tmp0);
        acc1 += nanoquant_apply_sign_word(word1, lane, tmp1);
        acc2 += nanoquant_apply_sign_word(word2, lane, tmp2);
        acc3 += nanoquant_apply_sign_word(word3, lane, tmp3);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (; word_idx < n_words; ++word_idx) {
        uint32_t word = lane == 0 ? nanoquant_load_sign_word(u_bits, nbu0, nbu1, o, word_idx) : 0;
        word = __shfl_sync(0xffffffff, word, 0);
        const int64_t r = word_idx*WARP_SIZE + lane;
        const float tmp_v = *(const float *) ((const char *) tmp + r*nbtmp0 + i1*nbtmp1 + i2*nbtmp2 + i3*nbtmp3);
        acc += nanoquant_apply_sign_word(word, lane, tmp_v);
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        float value = acc*scale_post[o];
        for (int s = 0; s < fast_salient; ++s) {
            const int64_t idx = salient_idx[s];
            const float weight = __half2float(salient_weight[o*fast_salient + s]);
            const float xv = *(const float *) ((const char *) x + idx*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
            value += xv*weight;
        }
        *(float *) ((char *) dst + o*nbd0 + i1*nbd1 + i2*nbd2 + i3*nbd3) = value;
    }
}

static bool nanoquant_vector_sign_loads(const ggml_tensor * bits) {
    // Four packed sign words can be loaded as uint4 only when each row starts
    // on a 16-byte boundary.
    return bits != nullptr &&
        bits->nb[0] == sizeof(int32_t) &&
        ((((uintptr_t) bits->data) | (uintptr_t) bits->nb[1]) & 0xF) == 0;
}

static int nanoquant_fast_salient_count(
        const ggml_tensor * x,
        const ggml_tensor * scale_post,
        const ggml_tensor * salient_idx,
        const ggml_tensor * salient_weight,
        const ggml_tensor * salient_scale,
        const ggml_tensor * dst,
        const int64_t       n_salient) {
    // Return the template salient count for layouts with direct pointer math;
    // -1 means the generic typed/strided path must be used.
    if (n_salient == 0) {
        return 0;
    }
    if (n_salient != 2 && n_salient != 7 && n_salient != 8) {
        return -1;
    }
    if (x->type != GGML_TYPE_F32 ||
            scale_post->type != GGML_TYPE_F32 ||
            salient_idx == nullptr ||
            salient_weight == nullptr ||
            salient_weight->type != GGML_TYPE_F16 ||
            salient_scale != nullptr ||
            x->nb[0] != sizeof(float) ||
            scale_post->nb[0] != sizeof(float) ||
            salient_idx->nb[0] != sizeof(int32_t) ||
            salient_weight->nb[0] != sizeof(half) ||
            salient_weight->nb[1] != n_salient*sizeof(half) ||
            dst->nb[0] != sizeof(float)) {
        return -1;
    }
    return (int) n_salient;
}

void ggml_cuda_op_nanoquant_stage1_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * v0 = dst->src[1];
    const ggml_tensor * pre0 = dst->src[2];
    const ggml_tensor * mid0 = dst->src[3];
    const ggml_tensor * v1 = dst->src[4];
    const ggml_tensor * pre1 = dst->src[5];
    const ggml_tensor * mid1 = dst->src[6];
    const ggml_tensor * v2 = dst->src[7];
    const ggml_tensor * pre2 = dst->src[8];
    const ggml_tensor * mid2 = dst->src[9];

    const int64_t n_in = x->ne[0];
    const int64_t n_rank0 = mid0 == nullptr ? 0 : mid0->ne[0];
    const int64_t n_rank1 = mid1 == nullptr ? 0 : mid1->ne[0];
    const int64_t n_rank2 = mid2 == nullptr ? 0 : mid2->ne[0];
    const int64_t n_rank_total = n_rank0 + n_rank1 + n_rank2;
    const int64_t n_col = dst->ne[1]*dst->ne[2]*dst->ne[3];

    constexpr int block_size = 256;
    constexpr int rows_per_block = block_size / WARP_SIZE;
    cudaStream_t stream = ctx.stream();

    const auto ptr = [](const ggml_tensor * t) { return t == nullptr ? nullptr : t->data; };
    const auto type = [](const ggml_tensor * t) { return t == nullptr ? GGML_TYPE_COUNT : t->type; };
    const auto nb0 = [](const ggml_tensor * t) { return t == nullptr ? 0 : t->nb[0]; };
    const auto nb1 = [](const ggml_tensor * t) { return t == nullptr ? 0 : t->nb[1]; };

    if (n_col == 1) {
        const bool full_warps = n_in % WARP_SIZE == 0;
        const bool fast_f32 =
            x->type == GGML_TYPE_F32 &&
            x->nb[0] == sizeof(float) &&
            (pre0 == nullptr || (pre0->type == GGML_TYPE_F32 && pre0->nb[0] == sizeof(float) && mid0->type == GGML_TYPE_F32 && mid0->nb[0] == sizeof(float))) &&
            (pre1 == nullptr || (pre1->type == GGML_TYPE_F32 && pre1->nb[0] == sizeof(float) && mid1->type == GGML_TYPE_F32 && mid1->nb[0] == sizeof(float))) &&
            (pre2 == nullptr || (pre2->type == GGML_TYPE_F32 && pre2->nb[0] == sizeof(float) && mid2->type == GGML_TYPE_F32 && mid2->nb[0] == sizeof(float)));

        const dim3 blocks((n_rank_total + rows_per_block - 1)/rows_per_block, n_col, 1);
        const ggml_cuda_kernel_launch_params params(blocks, block_size, 0, stream);
        if (full_warps && fast_f32) {
            ggml_cuda_kernel_launch(nanoquant_stage1_fused_warp_kernel<block_size, true, true>, params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) ptr(v0), nb0(v0), nb1(v0), nanoquant_vector_sign_loads(v0), ptr(pre0), type(pre0), nb0(pre0), ptr(mid0), type(mid0), nb0(mid0),
                (const int32_t *) ptr(v1), nb0(v1), nb1(v1), nanoquant_vector_sign_loads(v1), ptr(pre1), type(pre1), nb0(pre1), ptr(mid1), type(mid1), nb0(mid1),
                (const int32_t *) ptr(v2), nb0(v2), nb1(v2), nanoquant_vector_sign_loads(v2), ptr(pre2), type(pre2), nb0(pre2), ptr(mid2), type(mid2), nb0(mid2),
                (float *) dst->data, n_in, n_rank0, n_rank1, n_rank2, dst->ne[1], dst->ne[2]);
        } else if (full_warps) {
            ggml_cuda_kernel_launch(nanoquant_stage1_fused_warp_kernel<block_size, true, false>, params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) ptr(v0), nb0(v0), nb1(v0), nanoquant_vector_sign_loads(v0), ptr(pre0), type(pre0), nb0(pre0), ptr(mid0), type(mid0), nb0(mid0),
                (const int32_t *) ptr(v1), nb0(v1), nb1(v1), nanoquant_vector_sign_loads(v1), ptr(pre1), type(pre1), nb0(pre1), ptr(mid1), type(mid1), nb0(mid1),
                (const int32_t *) ptr(v2), nb0(v2), nb1(v2), nanoquant_vector_sign_loads(v2), ptr(pre2), type(pre2), nb0(pre2), ptr(mid2), type(mid2), nb0(mid2),
                (float *) dst->data, n_in, n_rank0, n_rank1, n_rank2, dst->ne[1], dst->ne[2]);
        } else {
            ggml_cuda_kernel_launch(nanoquant_stage1_fused_warp_kernel<block_size, false, false>, params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) ptr(v0), nb0(v0), nb1(v0), false, ptr(pre0), type(pre0), nb0(pre0), ptr(mid0), type(mid0), nb0(mid0),
                (const int32_t *) ptr(v1), nb0(v1), nb1(v1), false, ptr(pre1), type(pre1), nb0(pre1), ptr(mid1), type(mid1), nb0(mid1),
                (const int32_t *) ptr(v2), nb0(v2), nb1(v2), false, ptr(pre2), type(pre2), nb0(pre2), ptr(mid2), type(mid2), nb0(mid2),
                (float *) dst->data, n_in, n_rank0, n_rank1, n_rank2, dst->ne[1], dst->ne[2]);
        }
    } else {
        const bool fast_f32 =
            n_in % WARP_SIZE == 0 &&
            x->type == GGML_TYPE_F32 &&
            x->nb[0] == sizeof(float) &&
            (pre0 == nullptr || (pre0->type == GGML_TYPE_F32 && pre0->nb[0] == sizeof(float) && mid0->type == GGML_TYPE_F32 && mid0->nb[0] == sizeof(float))) &&
            (pre1 == nullptr || (pre1->type == GGML_TYPE_F32 && pre1->nb[0] == sizeof(float) && mid1->type == GGML_TYPE_F32 && mid1->nb[0] == sizeof(float))) &&
            (pre2 == nullptr || (pre2->type == GGML_TYPE_F32 && pre2->nb[0] == sizeof(float) && mid2->type == GGML_TYPE_F32 && mid2->nb[0] == sizeof(float)));
        if (fast_f32 && n_col >= 8) {
            // Below eight columns, the old block kernel has enough occupancy and
            // avoids the column-tile kernel's extra grid bookkeeping.
            constexpr int cols_per_block = 8;
            const dim3 blocks(n_rank_total, (n_col + cols_per_block - 1)/cols_per_block, 1);
            const ggml_cuda_kernel_launch_params params(blocks, block_size, 0, stream);
            ggml_cuda_kernel_launch(nanoquant_stage1_fused_coltile_kernel<block_size, cols_per_block>, params,
                x->data, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) ptr(v0), nb0(v0), nb1(v0), nanoquant_vector_sign_loads(v0), (const float *) ptr(pre0), (const float *) ptr(mid0),
                (const int32_t *) ptr(v1), nb0(v1), nb1(v1), nanoquant_vector_sign_loads(v1), (const float *) ptr(pre1), (const float *) ptr(mid1),
                (const int32_t *) ptr(v2), nb0(v2), nb1(v2), nanoquant_vector_sign_loads(v2), (const float *) ptr(pre2), (const float *) ptr(mid2),
                (float *) dst->data, n_in, n_rank0, n_rank1, n_rank2, n_col, dst->ne[1], dst->ne[2]);
            return;
        }

        const size_t shmem = rows_per_block*sizeof(float);
        const dim3 blocks(n_rank_total, n_col, 1);
        const ggml_cuda_kernel_launch_params params(blocks, block_size, shmem, stream);
        ggml_cuda_kernel_launch(nanoquant_stage1_fused_block_kernel<block_size>, params,
            x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            (const int32_t *) ptr(v0), nb0(v0), nb1(v0), ptr(pre0), type(pre0), nb0(pre0), ptr(mid0), type(mid0), nb0(mid0),
            (const int32_t *) ptr(v1), nb0(v1), nb1(v1), ptr(pre1), type(pre1), nb0(pre1), ptr(mid1), type(mid1), nb0(mid1),
            (const int32_t *) ptr(v2), nb0(v2), nb1(v2), ptr(pre2), type(pre2), nb0(pre2), ptr(mid2), type(mid2), nb0(mid2),
            (float *) dst->data, n_in, n_rank0, n_rank1, n_rank2, dst->ne[1], dst->ne[2]);
    }
}

void ggml_cuda_op_nanoquant_stage2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x          = dst->src[0];
    const ggml_tensor * tmp        = dst->src[1];
    const ggml_tensor * u_bits     = dst->src[2];
    const ggml_tensor * scale_post = dst->src[3];
    const ggml_tensor * salient_idx    = dst->src[4];
    const ggml_tensor * salient_weight = dst->src[5];
    const ggml_tensor * salient_scale  = dst->src[6];

    const int64_t n_rank = tmp->ne[0];
    const int64_t n_out  = dst->ne[0];
    const int64_t n_col  = dst->ne[1]*dst->ne[2]*dst->ne[3];
    const int64_t n_salient = salient_idx == nullptr ? 0 : salient_idx->ne[0];

    constexpr int block_size = 256;
    constexpr int rows_per_block = block_size / WARP_SIZE;
    cudaStream_t stream = ctx.stream();

    if (n_col == 1) {
        const bool full_warps = n_rank % WARP_SIZE == 0;
        const dim3 blocks((n_out + rows_per_block - 1)/rows_per_block, n_col, 1);
        const ggml_cuda_kernel_launch_params params(blocks, block_size, 0, stream);
        const bool no_salient = n_salient == 0;
        const int fast_salient = nanoquant_fast_salient_count(
                x, scale_post, salient_idx, salient_weight, salient_scale, dst, n_salient);

        if (full_warps && no_salient) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, -1, true>, params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], nanoquant_vector_sign_loads(u_bits),
                scale_post->data, scale_post->type, scale_post->nb[0],
                nullptr, 0, nullptr, GGML_TYPE_COUNT, 0, 0, nullptr, GGML_TYPE_COUNT, 0,
                n_salient, (const float *) tmp->data, (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (full_warps && fast_salient == 2) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, 2, false>, params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], nanoquant_vector_sign_loads(u_bits),
                scale_post->data, scale_post->type, scale_post->nb[0],
                (const int32_t *) salient_idx->data, salient_idx->nb[0],
                salient_weight->data, salient_weight->type, salient_weight->nb[0], salient_weight->nb[1],
                nullptr, GGML_TYPE_COUNT, 0,
                n_salient, (const float *) tmp->data, (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (full_warps && fast_salient == 7) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, 7, false>, params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], nanoquant_vector_sign_loads(u_bits),
                scale_post->data, scale_post->type, scale_post->nb[0],
                (const int32_t *) salient_idx->data, salient_idx->nb[0],
                salient_weight->data, salient_weight->type, salient_weight->nb[0], salient_weight->nb[1],
                nullptr, GGML_TYPE_COUNT, 0,
                n_salient, (const float *) tmp->data, (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (full_warps && fast_salient == 8) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, 8, false>, params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], nanoquant_vector_sign_loads(u_bits),
                scale_post->data, scale_post->type, scale_post->nb[0],
                (const int32_t *) salient_idx->data, salient_idx->nb[0],
                salient_weight->data, salient_weight->type, salient_weight->nb[0], salient_weight->nb[1],
                nullptr, GGML_TYPE_COUNT, 0,
                n_salient, (const float *) tmp->data, (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (full_warps) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, -1, false>, params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], nanoquant_vector_sign_loads(u_bits),
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, (const float *) tmp->data, (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, false, -1, false>, params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], false,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, (const float *) tmp->data, (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        }
    } else {
        const int fast_salient = nanoquant_fast_salient_count(
                x, scale_post, salient_idx, salient_weight, salient_scale, dst, n_salient);
        const bool use_coltile =
            n_rank % WARP_SIZE == 0 &&
            n_col >= 8 &&
            (fast_salient == 2 || fast_salient == 7 || fast_salient == 8) &&
            tmp->nb[0] == sizeof(float);
        if (use_coltile) {
            // The col-tile path trades per-dot block-level reductions for warp
            // reductions and much denser launches during prompt processing.
            constexpr int cols_per_block = 8;
            const dim3 blocks(n_out, (n_col + cols_per_block - 1)/cols_per_block, 1);
            const ggml_cuda_kernel_launch_params params(blocks, block_size, 0, stream);
            if (fast_salient == 2) {
                ggml_cuda_kernel_launch(nanoquant_stage2_coltile_kernel<block_size, cols_per_block, 2>, params,
                    x->data, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                    (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], nanoquant_vector_sign_loads(u_bits),
                    (const float *) scale_post->data,
                    (const int32_t *) salient_idx->data,
                    (const half *) salient_weight->data,
                    (const float *) tmp->data,
                    tmp->nb[0], tmp->nb[1], tmp->nb[2], tmp->nb[3],
                    (float *) dst->data, n_rank, n_col,
                    dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            } else if (fast_salient == 7) {
                ggml_cuda_kernel_launch(nanoquant_stage2_coltile_kernel<block_size, cols_per_block, 7>, params,
                    x->data, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                    (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], nanoquant_vector_sign_loads(u_bits),
                    (const float *) scale_post->data,
                    (const int32_t *) salient_idx->data,
                    (const half *) salient_weight->data,
                    (const float *) tmp->data,
                    tmp->nb[0], tmp->nb[1], tmp->nb[2], tmp->nb[3],
                    (float *) dst->data, n_rank, n_col,
                    dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            } else {
                ggml_cuda_kernel_launch(nanoquant_stage2_coltile_kernel<block_size, cols_per_block, 8>, params,
                    x->data, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                    (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], nanoquant_vector_sign_loads(u_bits),
                    (const float *) scale_post->data,
                    (const int32_t *) salient_idx->data,
                    (const half *) salient_weight->data,
                    (const float *) tmp->data,
                    tmp->nb[0], tmp->nb[1], tmp->nb[2], tmp->nb[3],
                    (float *) dst->data, n_rank, n_col,
                    dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            }
            return;
        }

        const size_t shmem = rows_per_block*sizeof(float);
        const dim3 blocks(n_out, n_col, 1);
        const ggml_cuda_kernel_launch_params params(blocks, block_size, shmem, stream);
        ggml_cuda_kernel_launch(nanoquant_stage2_block_kernel<block_size>, params,
            x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1],
            scale_post->data, scale_post->type, scale_post->nb[0],
            salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
            salient_idx == nullptr ? 0 : salient_idx->nb[0],
            salient_weight == nullptr ? nullptr : salient_weight->data,
            salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
            salient_weight == nullptr ? 0 : salient_weight->nb[0],
            salient_weight == nullptr ? 0 : salient_weight->nb[1],
            salient_scale == nullptr ? nullptr : salient_scale->data,
            salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
            salient_scale == nullptr ? 0 : salient_scale->nb[0],
            n_salient, (const float *) tmp->data,
            tmp->nb[0], tmp->nb[1], tmp->nb[2], tmp->nb[3],
            (float *) dst->data, n_rank,
            dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    }
}

void ggml_cuda_op_nanoquant_stage2_fused2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * tmp     = dst->src[1];
    const ggml_tensor * u0      = dst->src[2];
    const ggml_tensor * post0   = dst->src[3];
    const ggml_tensor * idx0    = dst->src[4];
    const ggml_tensor * weight0 = dst->src[5];
    const ggml_tensor * u1      = dst->src[6];
    const ggml_tensor * post1   = dst->src[7];
    const ggml_tensor * idx1    = dst->src[8];
    const ggml_tensor * weight1 = dst->src[9];

    const int64_t n_rank0 = ggml_get_op_params_i32(dst, 0);
    const int64_t n_rank1 = ggml_get_op_params_i32(dst, 1);
    const int64_t n_out0  = post0->ne[0];
    const int64_t n_out1  = post1->ne[0];
    const int64_t n_col   = dst->ne[1]*dst->ne[2]*dst->ne[3];
    const int64_t n_salient0 = idx0 == nullptr ? 0 : idx0->ne[0];
    const int64_t n_salient1 = idx1 == nullptr ? 0 : idx1->ne[0];

    constexpr int block_size = 256;
    constexpr int rows_per_block = block_size / WARP_SIZE;
    cudaStream_t stream = ctx.stream();

    if (n_col == 1) {
        // This op is intentionally narrow: it exists to trim decode launch
        // count for two projections with the common 2-outlier F16 layout.
        const int fast0 = nanoquant_fast_salient_count(x, post0, idx0, weight0, nullptr, dst, n_salient0);
        const int fast1 = nanoquant_fast_salient_count(x, post1, idx1, weight1, nullptr, dst, n_salient1);
        GGML_ASSERT(fast0 == 2 && fast1 == 2);

        const bool full_warps = n_rank0 % WARP_SIZE == 0 && n_rank1 % WARP_SIZE == 0;
        GGML_ASSERT(full_warps);

        const dim3 blocks((n_out0 + n_out1 + rows_per_block - 1)/rows_per_block, 1, 1);
        const ggml_cuda_kernel_launch_params params(blocks, block_size, 0, stream);
        ggml_cuda_kernel_launch(nanoquant_stage2_fused2_warp_kernel<block_size, true, 2, 2>, params,
            x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            (const float *) tmp->data,
            (const int32_t *) u0->data, u0->nb[0], u0->nb[1], nanoquant_vector_sign_loads(u0),
            post0->data, post0->type, post0->nb[0],
            (const int32_t *) idx0->data, idx0->nb[0],
            weight0->data, weight0->type, weight0->nb[0], weight0->nb[1],
            (const int32_t *) u1->data, u1->nb[0], u1->nb[1], nanoquant_vector_sign_loads(u1),
            post1->data, post1->type, post1->nb[0],
            (const int32_t *) idx1->data, idx1->nb[0],
            weight1->data, weight1->type, weight1->nb[0], weight1->nb[1],
            (float *) dst->data, n_rank0, n_rank1, n_out0, n_out0 + n_out1,
            dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    } else {
        // Graph construction currently avoids this branch for prompt batches;
        // keep the fallback correct for backend validation and future callers.
        const size_t shmem = rows_per_block*sizeof(float);
        const ggml_cuda_kernel_launch_params params0(dim3(n_out0, n_col, 1), block_size, shmem, stream);
        ggml_cuda_kernel_launch(nanoquant_stage2_block_kernel<block_size>, params0,
            x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            (const int32_t *) u0->data, u0->nb[0], u0->nb[1],
            post0->data, post0->type, post0->nb[0],
            (const int32_t *) idx0->data, idx0->nb[0],
            weight0->data, weight0->type, weight0->nb[0], weight0->nb[1],
            nullptr, GGML_TYPE_COUNT, 0,
            n_salient0, (const float *) tmp->data,
            tmp->nb[0], tmp->nb[1], tmp->nb[2], tmp->nb[3],
            (float *) dst->data, n_rank0,
            dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);

        const ggml_cuda_kernel_launch_params params1(dim3(n_out1, n_col, 1), block_size, shmem, stream);
        ggml_cuda_kernel_launch(nanoquant_stage2_block_kernel<block_size>, params1,
            x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            (const int32_t *) u1->data, u1->nb[0], u1->nb[1],
            post1->data, post1->type, post1->nb[0],
            (const int32_t *) idx1->data, idx1->nb[0],
            weight1->data, weight1->type, weight1->nb[0], weight1->nb[1],
            nullptr, GGML_TYPE_COUNT, 0,
            n_salient1, (const float *) ((const char *) tmp->data + n_rank0*tmp->nb[0]),
            tmp->nb[0], tmp->nb[1], tmp->nb[2], tmp->nb[3],
            (float *) ((char *) dst->data + n_out0*dst->nb[0]), n_rank1,
            dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    }
}

void ggml_cuda_op_nanoquant_linear(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x          = dst->src[0];
    const ggml_tensor * v_bits     = dst->src[1];
    const ggml_tensor * u_bits     = dst->src[2];
    const ggml_tensor * scale_pre  = dst->src[3];
    const ggml_tensor * scale_mid  = dst->src[4];
    const ggml_tensor * scale_post = dst->src[5];
    const ggml_tensor * salient_idx    = dst->src[6];
    const ggml_tensor * salient_weight = dst->src[7];
    const ggml_tensor * salient_scale  = dst->src[8];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(v_bits->type == GGML_TYPE_I32);
    GGML_ASSERT(u_bits->type == GGML_TYPE_I32);

    const int64_t n_in   = x->ne[0];
    const int64_t n_rank = scale_mid->ne[0];
    const int64_t n_out  = dst->ne[0];
    const int64_t n_col  = dst->ne[1]*dst->ne[2]*dst->ne[3];
    const int64_t n_salient = salient_idx == nullptr ? 0 : salient_idx->ne[0];

    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), n_col*n_rank);

    constexpr int block_size = 256;
    constexpr int rows_per_block = block_size / WARP_SIZE;
    cudaStream_t stream = ctx.stream();

    if (n_col == 1) {
        const dim3 stage1_blocks((n_rank + rows_per_block - 1)/rows_per_block, n_col, 1);
        const dim3 stage2_blocks((n_out  + rows_per_block - 1)/rows_per_block, n_col, 1);
        // Most NanoQuant decode shapes are multiples of 32, so specialize away tail checks when possible.
        const bool full_warps = n_in % WARP_SIZE == 0 && n_rank % WARP_SIZE == 0;
        const bool vector_v_loads =
            v_bits->nb[0] == sizeof(int32_t) &&
            ((((uintptr_t) v_bits->data) | (uintptr_t) v_bits->nb[1]) & 0xF) == 0;
        const bool vector_u_loads =
            u_bits->nb[0] == sizeof(int32_t) &&
            ((((uintptr_t) u_bits->data) | (uintptr_t) u_bits->nb[1]) & 0xF) == 0;

        // Direct pointer indexing avoids byte-stride arithmetic in the hot decode path.
        const bool fast_stage1 =
            x->type == GGML_TYPE_F32 &&
            scale_pre->type == GGML_TYPE_F32 &&
            scale_mid->type == GGML_TYPE_F32 &&
            x->nb[0] == sizeof(float) &&
            scale_pre->nb[0] == sizeof(float) &&
            scale_mid->nb[0] == sizeof(float);
        const ggml_cuda_kernel_launch_params stage1_params(stage1_blocks, block_size, 0, stream);
        if (full_warps && fast_stage1) {
            ggml_cuda_kernel_launch(nanoquant_stage1_warp_kernel<block_size, true, true>, stage1_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) v_bits->data, v_bits->nb[0], v_bits->nb[1], vector_v_loads,
                scale_pre->data, scale_pre->type, scale_pre->nb[0],
                scale_mid->data, scale_mid->type, scale_mid->nb[0],
                tmp.get(), n_in, n_rank, dst->ne[1], dst->ne[2]);
        } else if (full_warps) {
            ggml_cuda_kernel_launch(nanoquant_stage1_warp_kernel<block_size, true, false>, stage1_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) v_bits->data, v_bits->nb[0], v_bits->nb[1], vector_v_loads,
                scale_pre->data, scale_pre->type, scale_pre->nb[0],
                scale_mid->data, scale_mid->type, scale_mid->nb[0],
                tmp.get(), n_in, n_rank, dst->ne[1], dst->ne[2]);
        } else if (fast_stage1) {
            ggml_cuda_kernel_launch(nanoquant_stage1_warp_kernel<block_size, false, true>, stage1_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) v_bits->data, v_bits->nb[0], v_bits->nb[1], false,
                scale_pre->data, scale_pre->type, scale_pre->nb[0],
                scale_mid->data, scale_mid->type, scale_mid->nb[0],
                tmp.get(), n_in, n_rank, dst->ne[1], dst->ne[2]);
        } else {
            ggml_cuda_kernel_launch(nanoquant_stage1_warp_kernel<block_size, false, false>, stage1_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) v_bits->data, v_bits->nb[0], v_bits->nb[1], false,
                scale_pre->data, scale_pre->type, scale_pre->nb[0],
                scale_mid->data, scale_mid->type, scale_mid->nb[0],
                tmp.get(), n_in, n_rank, dst->ne[1], dst->ne[2]);
        }

        const ggml_cuda_kernel_launch_params stage2_params(stage2_blocks, block_size, 0, stream);
        const bool no_salient = n_salient == 0;
        const int fast_salient = nanoquant_fast_salient_count(
                x, scale_post, salient_idx, salient_weight, salient_scale, dst, n_salient);
        if (full_warps && no_salient) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, -1, true>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], vector_u_loads,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (full_warps && fast_salient == 2) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, 2, false>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], vector_u_loads,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (full_warps && fast_salient == 7) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, 7, false>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], vector_u_loads,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (full_warps && fast_salient == 8) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, 8, false>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], vector_u_loads,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (full_warps) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, true, -1, false>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], vector_u_loads,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (no_salient) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, false, -1, true>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], false,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (fast_salient == 2) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, false, 2, false>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], false,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (fast_salient == 7) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, false, 7, false>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], false,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else if (fast_salient == 8) {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, false, 8, false>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], false,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        } else {
            ggml_cuda_kernel_launch(nanoquant_stage2_warp_kernel<block_size, false, -1, false>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], false,
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(), (float *) dst->data, n_rank, n_out,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        }
    } else {
        const size_t shmem = rows_per_block*sizeof(float);
        const dim3 stage2_blocks(n_out,  n_col, 1);

        const bool fast_stage1 =
            n_in % WARP_SIZE == 0 &&
            n_col >= 8 &&
            x->type == GGML_TYPE_F32 &&
            scale_pre->type == GGML_TYPE_F32 &&
            scale_mid->type == GGML_TYPE_F32 &&
            x->nb[0] == sizeof(float) &&
            scale_pre->nb[0] == sizeof(float) &&
            scale_mid->nb[0] == sizeof(float);

        if (fast_stage1) {
            // This mirrors the split stage-1 prefill op, but uses the temporary
            // buffer allocated inside the monolithic NanoQuant op.
            constexpr int cols_per_block = 8;
            const dim3 stage1_blocks(n_rank, (n_col + cols_per_block - 1)/cols_per_block, 1);
            const ggml_cuda_kernel_launch_params stage1_params(stage1_blocks, block_size, 0, stream);
            ggml_cuda_kernel_launch(nanoquant_stage1_coltile_kernel<block_size, cols_per_block>, stage1_params,
                x->data, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) v_bits->data, v_bits->nb[0], v_bits->nb[1], nanoquant_vector_sign_loads(v_bits),
                (const float *) scale_pre->data, (const float *) scale_mid->data,
                tmp.get(), n_in, n_rank, n_col, dst->ne[1], dst->ne[2]);
        } else {
            const dim3 stage1_blocks(n_rank, n_col, 1);
            const ggml_cuda_kernel_launch_params stage1_params(stage1_blocks, block_size, shmem, stream);
            ggml_cuda_kernel_launch(nanoquant_stage1_block_kernel<block_size>, stage1_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) v_bits->data, v_bits->nb[0], v_bits->nb[1],
                scale_pre->data, scale_pre->type, scale_pre->nb[0],
                scale_mid->data, scale_mid->type, scale_mid->nb[0],
                tmp.get(), n_in, n_rank, dst->ne[1], dst->ne[2]);
        }

        const int fast_salient = nanoquant_fast_salient_count(
                x, scale_post, salient_idx, salient_weight, salient_scale, dst, n_salient);
        const bool fast_stage2 =
            n_rank % WARP_SIZE == 0 &&
            n_col >= 8 &&
            (fast_salient == 2 || fast_salient == 7 || fast_salient == 8);
        if (fast_stage2) {
            // tmp is internally allocated in [column, rank] order, so pass
            // explicit byte strides instead of relying on tensor metadata.
            constexpr int cols_per_block = 8;
            const dim3 stage2_coltile_blocks(n_out, (n_col + cols_per_block - 1)/cols_per_block, 1);
            const ggml_cuda_kernel_launch_params stage2_params(stage2_coltile_blocks, block_size, 0, stream);
            if (fast_salient == 2) {
                ggml_cuda_kernel_launch(nanoquant_stage2_coltile_kernel<block_size, cols_per_block, 2>, stage2_params,
                    x->data, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                    (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], nanoquant_vector_sign_loads(u_bits),
                    (const float *) scale_post->data,
                    (const int32_t *) salient_idx->data,
                    (const half *) salient_weight->data,
                    tmp.get(),
                    (int64_t) sizeof(float),
                    (int64_t) n_rank*sizeof(float),
                    (int64_t) n_rank*dst->ne[1]*sizeof(float),
                    (int64_t) n_rank*dst->ne[1]*dst->ne[2]*sizeof(float),
                    (float *) dst->data, n_rank, n_col,
                    dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            } else if (fast_salient == 7) {
                ggml_cuda_kernel_launch(nanoquant_stage2_coltile_kernel<block_size, cols_per_block, 7>, stage2_params,
                    x->data, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                    (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], nanoquant_vector_sign_loads(u_bits),
                    (const float *) scale_post->data,
                    (const int32_t *) salient_idx->data,
                    (const half *) salient_weight->data,
                    tmp.get(),
                    (int64_t) sizeof(float),
                    (int64_t) n_rank*sizeof(float),
                    (int64_t) n_rank*dst->ne[1]*sizeof(float),
                    (int64_t) n_rank*dst->ne[1]*dst->ne[2]*sizeof(float),
                    (float *) dst->data, n_rank, n_col,
                    dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            } else {
                ggml_cuda_kernel_launch(nanoquant_stage2_coltile_kernel<block_size, cols_per_block, 8>, stage2_params,
                    x->data, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                    (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1], nanoquant_vector_sign_loads(u_bits),
                    (const float *) scale_post->data,
                    (const int32_t *) salient_idx->data,
                    (const half *) salient_weight->data,
                    tmp.get(),
                    (int64_t) sizeof(float),
                    (int64_t) n_rank*sizeof(float),
                    (int64_t) n_rank*dst->ne[1]*sizeof(float),
                    (int64_t) n_rank*dst->ne[1]*dst->ne[2]*sizeof(float),
                    (float *) dst->data, n_rank, n_col,
                    dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            }
        } else {
            const ggml_cuda_kernel_launch_params stage2_params(stage2_blocks, block_size, shmem, stream);
            ggml_cuda_kernel_launch(nanoquant_stage2_block_kernel<block_size>, stage2_params,
                x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3],
                (const int32_t *) u_bits->data, u_bits->nb[0], u_bits->nb[1],
                scale_post->data, scale_post->type, scale_post->nb[0],
                salient_idx == nullptr ? nullptr : (const int32_t *) salient_idx->data,
                salient_idx == nullptr ? 0 : salient_idx->nb[0],
                salient_weight == nullptr ? nullptr : salient_weight->data,
                salient_weight == nullptr ? GGML_TYPE_COUNT : salient_weight->type,
                salient_weight == nullptr ? 0 : salient_weight->nb[0],
                salient_weight == nullptr ? 0 : salient_weight->nb[1],
                salient_scale == nullptr ? nullptr : salient_scale->data,
                salient_scale == nullptr ? GGML_TYPE_COUNT : salient_scale->type,
                salient_scale == nullptr ? 0 : salient_scale->nb[0],
                n_salient, tmp.get(),
                (int64_t) sizeof(float),
                (int64_t) n_rank*sizeof(float),
                (int64_t) n_rank*dst->ne[1]*sizeof(float),
                (int64_t) n_rank*dst->ne[1]*dst->ne[2]*sizeof(float),
                (float *) dst->data, n_rank,
                dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        }
    }
}
