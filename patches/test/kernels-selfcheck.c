// Self-check for the PrismML-fork SIMD kernels added in patches/prism-fork-avx2-kernels.patch:
//   - ggml_vec_dot_pq2_0_q8_0  (AVX2/VNNI, restructured)
//   - ggml_vec_dot_ptq1_0_q8_0 (AVX2/VNNI, new)
// Compares both against the scalar generic references on randomized blocks.
// Requires an AVX2+AVX-VNNI machine (same ISA the fork kernels target).
// Build: gcc -O2 -mavx2 -mavxvnni -mf16c -mfma -o kernels-selfcheck kernels-selfcheck.c
// Run:   ./kernels-selfcheck
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define QK8_0 32
#define QK_PQ2_0 128
#define QK_PTQ1_0 128
typedef uint16_t ggml_half;
typedef struct { ggml_half d; uint8_t qs[QK_PQ2_0 / 4]; } block_pq2_0;
typedef struct { uint8_t qs[(QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5]; uint8_t qh[QK_PTQ1_0/64]; ggml_half d; } block_ptq1_0;
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;

static inline float f16c_to_f32(ggml_half h) { return _cvtsh_ss((unsigned short) h); }
#define UNUSED(x) (void)(x)

static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#  define GGML_DPBUSD_256(acc, a, b) _mm256_dpbusd_epi32(acc, a, b)
#elif defined(__AVXVNNI__)
#  define GGML_DPBUSD_256(acc, a, b) _mm256_dpbusd_avx_epi32(acc, a, b)
#endif

// ---- PQ2_0 generic reference (scalar) ----
static void pq2_dot_generic(int n, float * s, const block_pq2_0 * x, const block_q8_0 * y) {
    const int qk = QK_PQ2_0;
    const int nb = n / qk;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d0 = f16c_to_f32(x[i].d);
        float sumi = 0.0f;
        for (int k = 0; k < 4; k++) {
            const block_q8_0 * yb = &y[i * 4 + k];
            const float d1 = f16c_to_f32(yb->d);
            int sumi_block = 0;
            const uint8_t * qs = &x[i].qs[k * 8];
            const int8_t * qy = yb->qs;
            for (int b = 0; b < 8; ++b) {
                const uint8_t byte = qs[b];
                sumi_block += ((int)((byte >> 0) & 3) - 1) * qy[b*4 + 0];
                sumi_block += ((int)((byte >> 2) & 3) - 1) * qy[b*4 + 1];
                sumi_block += ((int)((byte >> 4) & 3) - 1) * qy[b*4 + 2];
                sumi_block += ((int)((byte >> 6) & 3) - 1) * qy[b*4 + 3];
            }
            sumi += d1 * sumi_block;
        }
        sumf += d0 * sumi;
    }
    *s = sumf;
}

// ---- PQ2_0 AVX2 kernel (same math as arch/x86/quants.c) ----
static void pq2_dot_avx2(int n, float * s, const block_pq2_0 * x, const block_q8_0 * y) {
    const int qk = QK_PQ2_0;
    const int nb = n / qk;
    float sumf = 0.0f;
    const __m256i ones8  = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i mul    = _mm256_setr_epi16(64,16,4,1, 64,16,4,1, 64,16,4,1, 64,16,4,1);
    const __m256i three  = _mm256_set1_epi16(3);
    const __m128i idxlo  = _mm_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
    const __m128i idxhi  = _mm_setr_epi8(4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7);
    for (int i = 0; i < nb; i++) {
        const float d0 = f16c_to_f32(x[i].d);
        __m256 accf = _mm256_setzero_ps();
        for (int k = 0; k < 4; k++) {
            const block_q8_0 * yb = &y[i * 4 + k];
            const __m128i src = _mm_loadl_epi64((const __m128i *) &x[i].qs[k * 8]);
            const __m256i rep = _mm256_set_m128i(_mm_shuffle_epi8(src, idxhi), _mm_shuffle_epi8(src, idxlo));
            __m256i r0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rep));
            __m256i r1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(rep, 1));
            r0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r0, mul), 6), three);
            r1 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r1, mul), 6), three);
            const __m256i codes = _mm256_permute4x64_epi64(_mm256_packus_epi16(r0, r1), 0xD8);
#if (defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVXVNNI__)
            const __m256i qy = _mm256_loadu_si256((const __m256i *) yb->qs);
            const __m256i dp = GGML_DPBUSD_256(_mm256_setzero_si256(), codes, qy);
            const __m256i sq = GGML_DPBUSD_256(_mm256_setzero_si256(), ones8, qy);
            const __m256i ss = _mm256_sub_epi32(dp, sq);
#else
            const __m256i qy = _mm256_loadu_si256((const __m256i *) yb->qs);
            const __m256i dp = _mm256_madd_epi16(ones16, _mm256_maddubs_epi16(codes, qy));
            const __m256i sq = _mm256_madd_epi16(ones16, _mm256_maddubs_epi16(ones8, qy));
            const __m256i ss = _mm256_sub_epi32(dp, sq);
#endif
            const float dd = d0 * f16c_to_f32(yb->d);
            accf = _mm256_fmadd_ps(_mm256_set1_ps(dd), _mm256_cvtepi32_ps(ss), accf);
        }
        sumf += hsum_float_8(accf);
    }
    *s = sumf;
}

// ---- PTQ1_0 generic reference (scalar) ----
static void ptq_dot_generic(int n, float * s, const block_ptq1_0 * x, const block_q8_0 * y) {
    const int qk = QK_PTQ1_0;
    const int nb = n / qk;
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    static const size_t  stages[3] = {32, 16, 8};
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        int8_t q[QK_PTQ1_0];
        int o = 0;
        size_t j = 0;
        for (size_t st = 0; st < 3; ++st) {
            const size_t c = stages[st];
            for (; j + c <= sizeof(x->qs); j += c) {
                for (size_t nn = 0; nn < 5; ++nn) {
                    for (size_t m = 0; m < c; ++m) {
                        const uint8_t v  = x[i].qs[j + m] * pow3[nn];
                        const int16_t xi = ((uint16_t) v * 3) >> 8;
                        q[o++] = (int8_t) (xi - 1);
                    }
                }
            }
        }
        for (size_t nn = 0; nn < 4; ++nn) {
            for (size_t h = 0; h < sizeof(x->qh); ++h) {
                const uint8_t v  = x[i].qh[h] * pow3[nn];
                const int16_t xi = ((uint16_t) v * 3) >> 8;
                q[o++] = (int8_t) (xi - 1);
            }
        }
        const float d0 = f16c_to_f32(x[i].d);
        float sumi = 0.0f;
        for (int k = 0; k < 4; k++) {
            const block_q8_0 * yb = &y[i * 4 + k];
            const float d1 = f16c_to_f32(yb->d);
            int sumi_block = 0;
            for (int b = 0; b < 32; ++b) {
                sumi_block += (int) q[k*32 + b] * (int) yb->qs[b];
            }
            sumi += d1 * sumi_block;
        }
        sumf += d0 * sumi;
    }
    *s = sumf;
}

// ---- PTQ1_0 AVX2 kernel (same math as arch/x86/quants.c) ----
static void ptq_dot_avx2(int n, float * s, const block_ptq1_0 * x, const block_q8_0 * y) {
    const int qk = QK_PTQ1_0;
    const int nb = n / qk;
    float sumf = 0.0f;
    const __m256i ones8  = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i bmask  = _mm256_set1_epi16(0x00FF);
    const __m128i bmaskl = _mm_set1_epi16(0x00FF);
    const __m128i three  = _mm_set1_epi16(3);

    for (int i = 0; i < nb; i++) {
        const __m128i qsA = _mm_loadu_si128((const __m128i *) x[i].qs);
        const __m256i vA  = _mm256_cvtepu8_epi16(qsA);
        const __m128i qsB = _mm_loadl_epi64((const __m128i *) (x[i].qs + 16));
        const __m128i vB  = _mm_cvtepu8_epi16(qsB);

        __m128i a[5];
        {
            __m256i t = vA;
            for (int nn = 0; nn < 5; ++nn) {
                if (nn > 0) t = _mm256_mullo_epi16(t, _mm256_set1_epi16(3));
                const __m256i lo = _mm256_and_si256(t, bmask);
                const __m256i tn = _mm256_srli_epi16(_mm256_mullo_epi16(lo, _mm256_set1_epi16(3)), 8);
                a[nn] = _mm_packus_epi16(_mm256_castsi256_si128(tn), _mm256_extracti128_si256(tn, 1));
            }
        }
        __m128i b[5];
        {
            __m128i t = vB;
            for (int nn = 0; nn < 5; ++nn) {
                if (nn > 0) t = _mm_mullo_epi16(t, three);
                const __m128i lo = _mm_and_si128(t, bmaskl);
                const __m128i tn = _mm_srli_epi16(_mm_mullo_epi16(lo, three), 8);
                b[nn] = _mm_packus_epi16(tn, tn);
            }
        }
        uint8_t qh_dec[8];
        {
            static const uint8_t pow3[4] = {1, 3, 9, 27};
            for (int nn = 0; nn < 4; ++nn) {
                for (int h = 0; h < 2; ++h) {
                    const uint8_t v  = x[i].qh[h] * pow3[nn];
                    const int16_t xi = ((uint16_t) v * 3) >> 8;
                    qh_dec[nn * 2 + h] = (uint8_t) xi;
                }
            }
        }
        const __m128i vh = _mm_loadl_epi64((const __m128i *) qh_dec);

        __m256 accf = _mm256_setzero_ps();
        const __m256i codes[4] = {
            _mm256_set_m128i(a[1], a[0]),
            _mm256_set_m128i(a[3], a[2]),
            _mm256_set_m128i(_mm_unpacklo_epi64(b[0], b[1]), a[4]),
            _mm256_set_m128i(_mm_unpacklo_epi64(b[4], vh), _mm_unpacklo_epi64(b[2], b[3])),
        };
        for (int k = 0; k < 4; k++) {
            const block_q8_0 * yb = &y[i * 4 + k];
            const __m256i qy = _mm256_loadu_si256((const __m256i *) yb->qs);
#if (defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVXVNNI__)
            const __m256i dp = GGML_DPBUSD_256(_mm256_setzero_si256(), codes[k], qy);
            const __m256i sq = GGML_DPBUSD_256(_mm256_setzero_si256(), ones8, qy);
            const __m256i ss = _mm256_sub_epi32(dp, sq);
#else
            const __m256i dp = _mm256_madd_epi16(ones16, _mm256_maddubs_epi16(codes[k], qy));
            const __m256i sq = _mm256_madd_epi16(ones16, _mm256_maddubs_epi16(ones8, qy));
            const __m256i ss = _mm256_sub_epi32(dp, sq);
#endif
            const float dd = f16c_to_f32(x[i].d) * f16c_to_f32(yb->d);
            accf = _mm256_fmadd_ps(_mm256_set1_ps(dd), _mm256_cvtepi32_ps(ss), accf);
        }
        sumf += hsum_float_8(accf);
    }
    *s = sumf;
}

static uint32_t lcg = 42;
static uint32_t rnd(void) { lcg = lcg * 1664525u + 1013904223u; return lcg >> 8; }

static int check(const char * name, float a, float b) {
    if (a != a || b != b) { fprintf(stderr, "%s: NaN\n", name); return 1; }
    const float tol = 1e-3f * (fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b)) + 1e-4f;
    if (fabsf(a - b) > tol) {
        printf("FAIL %s: generic=%.9f avx2=%.9f\n", name, a, b);
        return 1;
    }
    return 0;
}

int main(void) {
    const int n = 5120; // typical hidden size: 40 blocks
    const int nb = n / QK_PQ2_0;
    static block_pq2_0 x2[128];
    static block_ptq1_0 xt[128];
    static block_q8_0 y[512];
    int fail = 0;

    // correctness: 512 random rows; PTQ1_0 bytes constrained to valid encodings (<= 242)
    for (int trial = 0; trial < 512 && !fail; ++trial) {
        for (int i = 0; i < nb; i++) {
            x2[i].d = (ggml_half) (0x3800 + (rnd() & 0x1fff));
            xt[i].d = (ggml_half) (0x3800 + (rnd() & 0x1fff));
            for (int j = 0; j < QK_PQ2_0 / 4; j++) {
                x2[i].qs[j] = (trial & 1) ? 0xFF : (uint8_t) rnd();
            }
            for (int j = 0; j < (QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5; j++) {
                xt[i].qs[j] = (uint8_t) (rnd() % 243); // valid 5-trit encodings only
            }
            for (int j = 0; j < QK_PTQ1_0/64; j++) {
                xt[i].qh[j] = (uint8_t) rnd();
            }
        }
        for (int i = 0; i < nb * 4; i++) {
            y[i].d = (ggml_half) (0x3800 + (rnd() & 0x1FFF));
            for (int j = 0; j < QK8_0; j++) {
                y[i].qs[j] = (int8_t) (rnd() % 255 - 127);
            }
        }
        float a, b;
        pq2_dot_generic(n, &a, x2, y);
        pq2_dot_avx2(n, &b, x2, y);
        fail |= check("pq2_0", a, b);
        ptq_dot_generic(n, &a, xt, y);
        ptq_dot_avx2(n, &b, xt, y);
        fail |= check("ptq1_0", a, b);
    }
    if (fail) return 1;
    printf("PASS: 512/512 trials match for pq2_0 and ptq1_0\n");

    // microbenchmark: one 27B-style GEMV row set (7168 rows x 40 blocks)
    for (int i = 0; i < 40; i++) {
        x2[i].d = 0x3C00; xt[i].d = 0x3C00;
        for (int j = 0; j < QK_PQ2_0 / 4; j++) { x2[i].qs[j] = (uint8_t) rnd(); }
        for (int j = 0; j < (QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5; j++) { xt[i].qs[j] = (uint8_t) (rnd() % 243); }
        for (int j = 0; j < QK_PTQ1_0/64; j++) { xt[i].qh[j] = (uint8_t) rnd(); }
        for (int k = 0; k < 4; k++) { y[i*4+k].d = 0x3C00; for (int j = 0; j < QK8_0; j++) y[i*4+k].qs[j] = (int8_t)(rnd() % 255 - 127); }
    }
    static float out[7168];
    struct timespec ts0, ts1;
    float sink = 0;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int r = 0; r < 7168; r++) { ptq_dot_avx2(n, &out[0], xt, y); sink += out[0]; }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double avx2_ms = (ts1.tv_sec - ts0.tv_sec) * 1e3 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int r = 0; r < 7168; r++) { ptq_dot_generic(n, &out[0], xt, y); sink += out[0]; }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double gen_ms = (ts1.tv_nsec - ts0.tv_nsec) / 1e6 + (ts1.tv_sec - ts0.tv_sec) * 1e3;
    printf("ptq1_0 layer GEMV: avx2 %.1f ms, generic %.1f ms, ratio %.2fx (sink %g)\n", avx2_ms, gen_ms, gen_ms / avx2_ms, sink);
    (void) sink;
    return 0;
}
