/* -*- c++ -*- ----------------------------------------------------------
   LMFF AVX-512 SIMD backend.

   FP64 (default): 8-wide __m512d compute, __mmask8 predicates.
   LMFF_MIXED_PREC: 16-wide __m512 fp32 compute, FP64 force writeback,
                    __mmask16 predicates for compute / __mmask8 for fp64
                    position/charge gather and force scatter.
------------------------------------------------------------------------- */

#ifndef LMFF_SIMD_AVX512_H
#define LMFF_SIMD_AVX512_H

#include <cmath>
#include <cstdint>
#include <immintrin.h>

/* ---------------------------------------------------------------------- */
/* Vector / predicate types (mirror SVE short names)                      */
/* ---------------------------------------------------------------------- */

struct svd_t {
  __m512d v;
  svd_t() = default;
  svd_t(__m512d x) : v(x) {}
  svd_t operator+(const svd_t &o) const { return _mm512_add_pd(v, o.v); }
  svd_t operator-(const svd_t &o) const { return _mm512_sub_pd(v, o.v); }
  svd_t operator*(const svd_t &o) const { return _mm512_mul_pd(v, o.v); }
  svd_t operator/(const svd_t &o) const { return _mm512_div_pd(v, o.v); }
  svd_t &operator+=(const svd_t &o)
  {
    v = _mm512_add_pd(v, o.v);
    return *this;
  }
  svd_t &operator-=(const svd_t &o)
  {
    v = _mm512_sub_pd(v, o.v);
    return *this;
  }
  svd_t &operator*=(const svd_t &o)
  {
    v = _mm512_mul_pd(v, o.v);
    return *this;
  }
};

inline svd_t operator-(const svd_t &a)
{
  return _mm512_sub_pd(_mm512_setzero_pd(), a.v);
}

inline svd_t operator-(const svd_t &a, double s)
{
  return _mm512_sub_pd(a.v, _mm512_set1_pd(s));
}

inline svd_t operator-(double s, const svd_t &a)
{
  return _mm512_sub_pd(_mm512_set1_pd(s), a.v);
}

inline svd_t operator*(double s, const svd_t &o)
{
  return _mm512_mul_pd(_mm512_set1_pd(s), o.v);
}

struct svf_t {
  __m512 v;
  svf_t() = default;
  svf_t(__m512 x) : v(x) {}
  svf_t operator+(const svf_t &o) const { return _mm512_add_ps(v, o.v); }
  svf_t operator-(const svf_t &o) const { return _mm512_sub_ps(v, o.v); }
  svf_t operator*(const svf_t &o) const { return _mm512_mul_ps(v, o.v); }
  svf_t operator/(const svf_t &o) const { return _mm512_div_ps(v, o.v); }
  svf_t &operator+=(const svf_t &o)
  {
    v = _mm512_add_ps(v, o.v);
    return *this;
  }
  svf_t &operator-=(const svf_t &o)
  {
    v = _mm512_sub_ps(v, o.v);
    return *this;
  }
  svf_t &operator*=(const svf_t &o)
  {
    v = _mm512_mul_ps(v, o.v);
    return *this;
  }
};

inline svf_t operator-(const svf_t &a)
{
  return _mm512_sub_ps(_mm512_setzero_ps(), a.v);
}

inline svf_t operator-(const svf_t &a, float s)
{
  return _mm512_sub_ps(a.v, _mm512_set1_ps(s));
}

inline svf_t operator-(float s, const svf_t &a)
{
  return _mm512_sub_ps(_mm512_set1_ps(s), a.v);
}

inline svf_t operator*(float s, const svf_t &o)
{
  return _mm512_mul_ps(_mm512_set1_ps(s), o.v);
}

#ifdef LMFF_MIXED_PREC
typedef __m512i svsi_t;
typedef svf_t MY_VEC;
typedef __mmask16 svp_t;
#else
struct svsi_t {
  __m256i v;
  svsi_t() = default;
  svsi_t(__m256i x) : v(x) {}
};
typedef svd_t MY_VEC;
typedef __mmask8 svp_t;
#endif

struct svsl_t {
  __m512i v;
  svsl_t() = default;
  svsl_t(__m512i x) : v(x) {}
};

/* ---------------------------------------------------------------------- */
/* Duplicates / predicates                                                */
/* ---------------------------------------------------------------------- */

#ifdef LMFF_MIXED_PREC
inline svsi_t svdup_s32(int x)
{
  return _mm512_set1_epi32(x);
}
#else
inline svsi_t svdup_s32(int x)
{
  return _mm256_set1_epi32(x);
}
#endif

inline svsl_t svdup_s64(int64_t x)
{
  return _mm512_set1_epi64(x);
}

inline svp_t svwhilelt_b32(int /*a*/, int b)
{
  return b >= 16 ? 0xFFFF : (__mmask16) ((1u << b) - 1u);
}

inline svp_t svwhilelt_b64(int start, int b)
{
  int end = b < start ? start : (b < start + 8 ? b : start + 8);
  int cnt = end - start;
  if (cnt <= 0) return 0;
  return (__mmask8) (cnt >= 8 ? 0xFF : ((1u << cnt) - 1u));
}

#define svptest_any(pg, a) ((a) != 0)

/* ---------------------------------------------------------------------- */
/* Loads / stores                                                         */
/* ---------------------------------------------------------------------- */

#ifdef LMFF_MIXED_PREC
inline svf_t svld1(svp_t pg, const float *p)
{
  return _mm512_mask_loadu_ps(_mm512_setzero_ps(), pg, p);
}

inline void svst1(svp_t pg, float *p, svf_t x)
{
  _mm512_mask_storeu_ps(p, pg, x.v);
}
#else
inline svd_t svld1(svp_t pg, const double *p)
{
  return _mm512_mask_loadu_pd(_mm512_setzero_pd(), pg, p);
}

inline void svst1(svp_t pg, double *p, svd_t x)
{
  _mm512_mask_storeu_pd(p, pg, x.v);
}
#endif

#ifdef LMFF_MIXED_PREC
inline svsi_t svld1(svp_t pg, const int32_t *p)
{
  return _mm512_mask_loadu_epi32(_mm512_setzero_si512(), pg, p);
}
#else
inline svsi_t svld1(svp_t pg, const int32_t *p)
{
  return _mm256_mask_loadu_epi32(_mm256_setzero_si256(), pg, p);
}
#endif

inline svsl_t svunpklo(svsi_t x)
{
#ifdef LMFF_MIXED_PREC
  return _mm512_cvtepi32_epi64(_mm512_castsi512_si256(x));
#else
  return _mm512_cvtepi32_epi64(x.v);
#endif
}

#ifdef LMFF_MIXED_PREC
inline svsl_t svunpkhi(svsi_t x)
{
  return _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(x, 1));
}
#endif

inline svd_t svld1_gather_offset(svp_t pg, const double *base, svsl_t byte_off)
{
  __m512i idx = _mm512_srli_epi64(byte_off.v, 3);
  return _mm512_mask_i64gather_pd(_mm512_setzero_pd(), pg, idx, base, 8);
}

inline svd_t svld1_gather_index(svp_t pg, const double *base, svsl_t idx)
{
  return _mm512_mask_i64gather_pd(_mm512_setzero_pd(), pg, idx.v, base, 8);
}

#ifdef LMFF_MIXED_PREC
inline svsi_t svld1_gather_index(svp_t pg, const int32_t *base, svsi_t idx)
{
  return _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), pg, idx, base, 4);
}
#else
inline svsi_t svld1_gather_index(svp_t pg, const int32_t *base, svsi_t idx)
{
  return _mm256_mmask_i32gather_epi32(_mm256_setzero_si256(), pg, idx.v, base, 4);
}
#endif

inline svsl_t svld1sw_gather_index_s64(svp_t pg, const int32_t *base, svsl_t idx)
{
  __m256i idx32 = _mm512_cvtepi64_epi32(idx.v);
  __m512i idx512 = _mm512_castsi256_si512(idx32);
  __m512i i32 = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), pg, idx512, base, 4);
  return _mm512_cvtepi32_epi64(_mm512_castsi512_si256(i32));
}

inline void svst1_scatter_offset(svp_t pg, double *base, svsl_t byte_off, svd_t val)
{
  __m512i idx = _mm512_srli_epi64(byte_off.v, 3);
  _mm512_mask_i64scatter_pd(base, pg, idx, val.v, 8);
}

/* ---------------------------------------------------------------------- */
/* Compare / select / logical (predicates)                                */
/* ---------------------------------------------------------------------- */

inline svp_t svcmpgt(svp_t pg, svsl_t a, svsl_t b)
{
  return _mm512_mask_cmp_epi64_mask(pg, a.v, b.v, _MM_CMPINT_NLE);
}

inline svp_t svcmpeq(svp_t pg, svsl_t a, svsl_t b)
{
  return _mm512_mask_cmp_epi64_mask(pg, a.v, b.v, _MM_CMPINT_EQ);
}

inline svp_t svcmpne(svp_t pg, svsl_t a, svsl_t b)
{
  return _mm512_mask_cmp_epi64_mask(pg, a.v, b.v, _MM_CMPINT_NE);
}

#ifdef LMFF_MIXED_PREC
inline svp_t svcmplt(svp_t pg, svf_t a, svf_t b)
{
  return _mm512_mask_cmp_ps_mask(pg, a.v, b.v, _CMP_LT_OQ);
}

inline svp_t svcmpeq(svp_t pg, svf_t a, svf_t b)
{
  return _mm512_mask_cmp_ps_mask(pg, a.v, b.v, _CMP_EQ_OQ);
}

inline svp_t svcmpne(svp_t pg, svf_t a, svf_t b)
{
  return _mm512_mask_cmp_ps_mask(pg, a.v, b.v, _CMP_NEQ_OQ);
}

inline svp_t svcmpgt(svp_t pg, svf_t a, svf_t b)
{
  return _mm512_mask_cmp_ps_mask(pg, a.v, b.v, _CMP_GT_OQ);
}

inline svp_t svcmpgt(svp_t pg, svsi_t a, svsi_t b)
{
  return _mm512_mask_cmp_epi32_mask(pg, a, b, _MM_CMPINT_NLE);
}

inline svp_t svcmpeq(svp_t pg, svsi_t a, svsi_t b)
{
  return _mm512_mask_cmp_epi32_mask(pg, a, b, _MM_CMPINT_EQ);
}

inline svp_t svcmpne(svp_t pg, svsi_t a, svsi_t b)
{
  return _mm512_mask_cmp_epi32_mask(pg, a, b, _MM_CMPINT_NE);
}
#else
inline svp_t svcmplt(svp_t pg, svd_t a, svd_t b)
{
  return _mm512_mask_cmp_pd_mask(pg, a.v, b.v, _CMP_LT_OQ);
}

inline svp_t svcmpeq(svp_t pg, svd_t a, svd_t b)
{
  return _mm512_mask_cmp_pd_mask(pg, a.v, b.v, _CMP_EQ_OQ);
}

inline svp_t svcmpne(svp_t pg, svd_t a, svd_t b)
{
  return _mm512_mask_cmp_pd_mask(pg, a.v, b.v, _CMP_NEQ_OQ);
}

inline svp_t svcmpgt(svp_t pg, svd_t a, svd_t b)
{
  return _mm512_mask_cmp_pd_mask(pg, a.v, b.v, _CMP_GT_OQ);
}
#endif

#ifdef LMFF_MIXED_PREC
inline svf_t svsel(svp_t pg, svf_t a, svf_t b)
{
  return _mm512_mask_blend_ps(pg, b.v, a.v);
}
#else
inline svd_t svsel(svp_t pg, svd_t a, svd_t b)
{
  return _mm512_mask_blend_pd(pg, b.v, a.v);
}
#endif

inline svp_t svand_z(svp_t pg, svp_t a, svp_t b)
{
#ifdef LMFF_MIXED_PREC
  return _kand_mask16(pg, _kand_mask16(a, b));
#else
  return _kand_mask8(pg, _kand_mask8(a, b));
#endif
}

inline svp_t svorr_z(svp_t pg, svp_t a, svp_t b)
{
#ifdef LMFF_MIXED_PREC
  return _kand_mask16(pg, _kor_mask16(a, b));
#else
  return _kand_mask8(pg, _kor_mask8(a, b));
#endif
}

inline svp_t sveor_z(svp_t pg, svp_t a, svp_t b)
{
#ifdef LMFF_MIXED_PREC
  return _kand_mask16(pg, _kxor_mask16(a, b));
#else
  return _kand_mask8(pg, _kxor_mask8(a, b));
#endif
}

#ifdef LMFF_MIXED_PREC
inline svsi_t svand_z(svp_t pg, svsi_t a, svsi_t b)
{
  return _mm512_mask_and_epi32(_mm512_setzero_si512(), pg, a, b);
}

inline svsi_t sveor_z(svp_t pg, svsi_t a, svsi_t b)
{
  return _mm512_mask_xor_epi32(_mm512_setzero_si512(), pg, a, b);
}
#else
inline svsl_t svand_z(svp_t pg, svsl_t a, svsl_t b)
{
  return _mm512_mask_and_epi64(_mm512_setzero_si512(), pg, a.v, b.v);
}

inline svsl_t sveor_z(svp_t pg, svsl_t a, svsl_t b)
{
  return _mm512_mask_xor_epi64(_mm512_setzero_si512(), pg, a.v, b.v);
}
#endif

/* ---------------------------------------------------------------------- */
/* Mixed-precision conversions / zip                                      */
/* ---------------------------------------------------------------------- */

#ifdef LMFF_MIXED_PREC

inline svf_t svcvt_f32_f64_x(svp_t pg, svd_t x)
{
  __m256 f8 = _mm512_cvtpd_ps(x.v);
  __m512 out = _mm512_castps256_ps512(f8);
  return _mm512_mask_mov_ps(_mm512_setzero_ps(), pg, out);
}

/* Force writeback uses svzip1(fxi, zero) / svzip2(fxi, zero) only.
   Keep lanes 0..7 (or 8..15 moved to 0..7) as contiguous fp32 forces. */
inline svd_t svcvt_f64_f32_x(svp_t pg, svf_t x)
{
  __m256 f8 = _mm512_castps512_ps256(x.v);
  __m512d y = _mm512_cvtps_pd(f8);
  return _mm512_mask_mov_pd(_mm512_setzero_pd(), pg, y);
}

inline svf_t svzip1(svf_t a, svf_t /*b*/)
{
  return a;
}

inline svf_t svzip2(svf_t a, svf_t /*b*/)
{
  return _mm512_insertf32x8(_mm512_setzero_ps(), _mm512_extractf32x8_ps(a.v, 1), 0);
}

inline svf_t svuzp1(svf_t a, svf_t b)
{
  __m256 lo = _mm512_castps512_ps256(a.v);
  __m256 hi = _mm512_castps512_ps256(b.v);
  return _mm512_insertf32x8(_mm512_castps256_ps512(lo), hi, 1);
}

#endif    // LMFF_MIXED_PREC

/* ---------------------------------------------------------------------- */
/* Integer helpers                                                        */
/* ---------------------------------------------------------------------- */

inline svsl_t operator*(const svsl_t &a, int64_t s)
{
  return _mm512_mullo_epi64(a.v, _mm512_set1_epi64(s));
}

inline svsl_t operator*(int64_t s, const svsl_t &a)
{
  return a * s;
}

inline svsl_t operator+(const svsl_t &a, int64_t s)
{
  return _mm512_add_epi64(a.v, _mm512_set1_epi64(s));
}

inline svsl_t operator+(const svsl_t &a, int s)
{
  return a + static_cast<int64_t>(s);
}

inline svsl_t operator+(int64_t s, const svsl_t &a)
{
  return a + s;
}

inline svsl_t operator+(int s, const svsl_t &a)
{
  return a + static_cast<int64_t>(s);
}

/* ---------------------------------------------------------------------- */
/* Vector math: svnxp_exp / svnxp_log                                     */
/* ---------------------------------------------------------------------- */

#if defined(LMFF_SVNX_MATH_MKL)

#include <svmlintrin.h>

template<int N, int P>
inline void svnxp_exp(svd_t *y, svd_t *x)
{
  for (int i = 0; i < N; ++i)
    y[i].v = _mm512_exp_pd(x[i].v);
}

template<int N, int P>
inline void svnxp_log(svd_t *y, svd_t *x)
{
  for (int i = 0; i < N; ++i)
    y[i].v = _mm512_log_pd(x[i].v);
}

#ifdef LMFF_MIXED_PREC
template<int N, int P>
inline void svnxp_expf(svf_t *y, svf_t *x)
{
  for (int i = 0; i < N; ++i)
    y[i].v = _mm512_exp_ps(x[i].v);
}

template<int N, int P>
inline void svnxp_logf(svf_t *y, svf_t *x)
{
  for (int i = 0; i < N; ++i)
    y[i].v = _mm512_log_ps(x[i].v);
}
#endif

#elif defined(LMFF_SVNX_MATH_LIBMVEC)

extern "C" __m512d _ZGVeN8v_exp(__m512d);
extern "C" __m512d _ZGVeN8v_log(__m512d);
#ifdef LMFF_MIXED_PREC
extern "C" __m512 _ZGVeN16v_exp(__m512);
extern "C" __m512 _ZGVeN16v_log(__m512);
#endif

template<int N, int P>
inline void svnxp_exp(svd_t *y, svd_t *x)
{
  for (int i = 0; i < N; ++i)
    y[i].v = _ZGVeN8v_exp(x[i].v);
}

template<int N, int P>
inline void svnxp_log(svd_t *y, svd_t *x)
{
  for (int i = 0; i < N; ++i)
    y[i].v = _ZGVeN8v_log(x[i].v);
}

#ifdef LMFF_MIXED_PREC
template<int N, int P>
inline void svnxp_expf(svf_t *y, svf_t *x)
{
  for (int i = 0; i < N; ++i)
    y[i].v = _ZGVeN16v_exp(x[i].v);
}

template<int N, int P>
inline void svnxp_logf(svf_t *y, svf_t *x)
{
  for (int i = 0; i < N; ++i)
    y[i].v = _ZGVeN16v_log(x[i].v);
}
#endif

#else

template<int N, int P>
inline void svnxp_exp(svd_t *y, svd_t *x)
{
  alignas(64) double xbuf[8];
  alignas(64) double ybuf[8];
  for (int i = 0; i < N; ++i) {
    _mm512_store_pd(xbuf, x[i].v);
    for (int j = 0; j < 8; ++j)
      ybuf[j] = std::exp(xbuf[j]);
    y[i].v = _mm512_load_pd(ybuf);
  }
}

template<int N, int P>
inline void svnxp_log(svd_t *y, svd_t *x)
{
  alignas(64) double xbuf[8];
  alignas(64) double ybuf[8];
  for (int i = 0; i < N; ++i) {
    _mm512_store_pd(xbuf, x[i].v);
    for (int j = 0; j < 8; ++j)
      ybuf[j] = std::log(xbuf[j]);
    y[i].v = _mm512_load_pd(ybuf);
  }
}

#ifdef LMFF_MIXED_PREC
template<int N, int P>
inline void svnxp_expf(svf_t *y, svf_t *x)
{
  alignas(64) float xbuf[16];
  alignas(64) float ybuf[16];
  for (int i = 0; i < N; ++i) {
    _mm512_store_ps(xbuf, x[i].v);
    for (int j = 0; j < 16; ++j)
      ybuf[j] = std::exp(xbuf[j]);
    y[i].v = _mm512_load_ps(ybuf);
  }
}

template<int N, int P>
inline void svnxp_logf(svf_t *y, svf_t *x)
{
  alignas(64) float xbuf[16];
  alignas(64) float ybuf[16];
  for (int i = 0; i < N; ++i) {
    _mm512_store_ps(xbuf, x[i].v);
    for (int j = 0; j < 16; ++j)
      ybuf[j] = std::log(xbuf[j]);
    y[i].v = _mm512_load_ps(ybuf);
  }
}
#endif

#endif

/* ---------------------------------------------------------------------- */
/* Unified MY_VEC API (match lmff_simd_sve.h)                             */
/* ---------------------------------------------------------------------- */

#ifdef LMFF_MIXED_PREC

#define svdup(x) (svf_t(_mm512_set1_ps((float) (x))))
#define svadd_m(pg, a, b) (svf_t(_mm512_mask_add_ps((a).v, (pg), (a).v, (b).v)))
#define svdiv_x(pg, a, b) (svf_t(_mm512_mask_div_ps((a).v, (pg), (a).v, (b).v)))
#define svsqrt_x(pg, x) (svf_t(_mm512_mask_sqrt_ps((x).v, (pg), (x).v)))
#define svmad_x(pg, a, b, c) (svf_t(_mm512_maskz_fmadd_ps((pg), (a).v, (b).v, (c).v)))
#define svaddv(pg, x) _mm512_reduce_add_ps(_mm512_maskz_mov_ps((pg), (x).v))
#define svwhilelt svwhilelt_b32
#define svptrue() (0xFFFF)
#define svnxp_exp svnxp_expf
#define svnxp_log svnxp_logf

#else

#define svdup(x) (svd_t(_mm512_set1_pd(x)))
#define svadd_m(pg, a, b) (svd_t(_mm512_mask_add_pd((a).v, (pg), (a).v, (b).v)))
#define svdiv_x(pg, a, b) (svd_t(_mm512_mask_div_pd((a).v, (pg), (a).v, (b).v)))
#define svsqrt_x(pg, x) (svd_t(_mm512_mask_sqrt_pd((x).v, (pg), (x).v)))
#define svmad_x(pg, a, b, c) (svd_t(_mm512_maskz_fmadd_pd((pg), (a).v, (b).v, (c).v)))
#define svaddv(pg, x) _mm512_reduce_add_pd(_mm512_maskz_mov_pd((pg), (x).v))
#define svwhilelt svwhilelt_b64
#define svptrue() (0xFF)

#endif

#endif    // LMFF_SIMD_AVX512_H
