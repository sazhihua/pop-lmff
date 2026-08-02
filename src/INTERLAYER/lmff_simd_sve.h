/* -*- c++ -*- ----------------------------------------------------------
   LMFF ARM SVE SIMD backend (512-bit).
------------------------------------------------------------------------- */

#ifndef LMFF_SIMD_SVE_H
#define LMFF_SIMD_SVE_H

#include <arm_sve.h>
#include <cmath>
#include <cstdint>
#include <utility>

typedef svfloat64_t svd_t __attribute__((arm_sve_vector_bits(512)));
typedef svfloat32_t svf_t __attribute__((arm_sve_vector_bits(512)));
typedef svuint64_t svul_t __attribute__((arm_sve_vector_bits(512)));
typedef svint64_t svsl_t __attribute__((arm_sve_vector_bits(512)));
typedef svuint32_t svui_t __attribute__((arm_sve_vector_bits(512)));
typedef svint32_t svsi_t __attribute__((arm_sve_vector_bits(512)));
typedef svbool_t svp_t __attribute__((arm_sve_vector_bits(512)));

#ifdef LMFF_MIXED_PREC
typedef svf_t MY_VEC;
#else
typedef svd_t MY_VEC;
#endif

/* ---------------------------------------------------------------------- */
/* Vector math: svnxp_exp / svnxp_log                                     */
/* ---------------------------------------------------------------------- */

template<int ...Is>
using iseq = std::integer_sequence<int, Is...>;
template<int N>
using mkiseq = std::make_integer_sequence<int, N>;
template<typename T, typename ...Ts>
T __first(T, Ts...){
  return {};
}
#define __unused_arr__(...) do {__attribute__((unused)) decltype(__first(__VA_ARGS__)) a[] = {__VA_ARGS__};} while (0)

template<int N>
struct __const_factorial {
  constexpr static double value = N * __const_factorial<N-1>::value;
  constexpr static double rvalue = 1 / value;
  constexpr static double value64 = value;
  constexpr static double rvalue64 = rvalue;
  constexpr static float value32 = value;
  constexpr static float rvalue32 = rvalue;
};
template<>
struct __const_factorial<0> {
  constexpr static double value = 1;
  constexpr static double rvalue = 1;
  constexpr static double value64 = value;
  constexpr static double rvalue64 = rvalue;
  constexpr static float value32 = value;
  constexpr static float rvalue32 = rvalue;
};

template<int P, int NP, int ...Vs>
__always_inline void __svnxp_expfma(svd_t *expval, svd_t *qqln2, iseq<Vs...>){
  if (P == 0)
    __unused_arr__(expval[Vs] = svdup_f64(__const_factorial<NP-1>::rvalue)...);
  else
    __unused_arr__(expval[Vs] = expval[Vs] * qqln2[Vs] + __const_factorial<NP-1-P>::rvalue...);
}
template<int ...Vs, int ...Ps>
void __svnxp_exp(svd_t *y, svd_t *x, iseq<Vs...> vseq, iseq<Ps...>) {
  svd_t xscaled[] = {x[Vs] * (1.0/M_LN2)...};
  svd_t x64[] = {svscale_x(svptrue_b64(), xscaled[Vs], svdup_s64(6)) ...};
  svd_t x64r[] = {svrinta_x(svptrue_b64(), x64[Vs])...};
  svsl_t a64ri[] = {svcvt_s64_x(svptrue_b64(), x64r[Vs])...};
  svsl_t arl[] = {a64ri[Vs] & 63 ...};
  svd_t stval[] = {svexpa((svul_t)(a64ri[Vs] & 63) + (0x3ff << 6))...};
  svsl_t arh[] = {a64ri[Vs] >> 6 ...};
  svd_t q[] = { xscaled[Vs] - (svd_t)svcvt_f64_x(svptrue_b64(), arh[Vs]) ...};
  svd_t qq[] = { q[Vs] - (svd_t)svscale_f64_x(svptrue_b64(), svcvt_f64_x(svptrue_b64(), arl[Vs]), svdup_s64(-6)) ...};
  svd_t qqln2[] = { qq[Vs] * M_LN2 ...};
  svd_t expval[sizeof...(Vs)];
  __unused_arr__((__svnxp_expfma<Ps, sizeof...(Ps)>(expval, qqln2, vseq), 0) ...);
  __unused_arr__(expval[Vs] = expval[Vs] * stval[Vs]...);
  __unused_arr__(y[Vs] = svscale_x(svptrue_b64(), expval[Vs], arh[Vs])...);
}
template<int V, int P>
void svnxp_exp(svd_t *y, svd_t *x) {
  __svnxp_exp(y, x, mkiseq<V>{}, mkiseq<P>{});
}

template<int N>
struct __log_coefs{
  static constexpr double value = N==0 ? 0 : ((N % 2 == 0) ? -1.0 / N : 1.0 / N);
  static constexpr double value64 = value;
  static constexpr float value32 = value;
};

struct log_tbl_t {
  uint32_t base_tbll[32], base_tblh[32], coef_tbll[32], coef_tblh[32], base_tblf[32], coef_tblf[32];
};
__attribute__((weak)) log_tbl_t log_tbl = {
  .base_tbll = {
    0x00000000, 0xc79a9a22, 0xb0fc03e4, 0x6e2af2e6, 0x0e783300, 0x6d5e3e2b, 0xf632dcfc, 0xe27390e3,
    0xc01162a6, 0xababa60e, 0x961bd1d1, 0xfbcf7966, 0x8ae56b4c, 0xbae11d31, 0x3a4ad563, 0x556945ea,
    0x6e2af2e6, 0xc21c5ec2, 0x7cd08e59, 0x872df82d, 0xf81ff523, 0xbf5809ca, 0x6ccb7d1e, 0x1e9b14af,
    0x70a793d4, 0xf6bbd007, 0x0210d909, 0x3b82afc3, 0x9cf456b4, 0x86c1425b, 0x4e80bff3, 0x1c2116fb,
  },
  .base_tblh = {
    0x00000000, 0x3fcc8ff7, 0x3f8fc0a8, 0x3fce2707, 0x3f9f829b, 0x3fcfb918, 0x3fa77458, 0x3fd0a324,
    0x3faf0a30, 0x3fd1675c, 0x3fb341d7, 0x3fd22941, 0x3fb6f0d2, 0x3fd2e8e2, 0x3fba926d, 0x3fd3a64c,
    0x3fbe2707, 0x3fd4618b, 0x3fc0d77e, 0x3fd51aad, 0x3fc29552, 0x3fd5d1bd, 0x3fc44d2b, 0x3fd686c8,
    0x3fc5ff30, 0x3fd739d7, 0x3fc7ab89, 0x3fd7eaf8, 0x3fc9525a, 0x3fd89a33, 0x3fcaf3c9, 0x3fd94794,
  },
  .coef_tbll = {
    0x00000000, 0x9999999a, 0x1f81f820, 0xfcd6e9e0, 0xf07c1f08, 0x8f9c18fa, 0xabf0b767, 0x0f6bf3aa,
    0x1e1e1e1e, 0x18618618, 0x76b981db, 0x18181818, 0x1d41d41d, 0x417d05f4, 0x89039b0b, 0x8178a4c8,
    0x1c71c71c, 0x745d1746, 0x0381c0e0, 0x5c0b8170, 0x14c1bad0, 0x16c16c17, 0xb4e81b4f, 0x16816817,
    0xbca1af28, 0x590b2164, 0x606a63be, 0x60581606, 0x1a41a41a, 0x2b931057, 0x951033d9, 0x308158ed,
  },
  .coef_tblh = {
    0x3ff00000, 0x3fe99999, 0x3fef81f8, 0x3fe948b0, 0x3fef07c1, 0x3fe8f9c1, 0x3fee9131, 0x3fe8acb9,
    0x3fee1e1e, 0x3fe86186, 0x3fedae60, 0x3fe81818, 0x3fed41d4, 0x3fe7d05f, 0x3fecd856, 0x3fe78a4c,
    0x3fec71c7, 0x3fe745d1, 0x3fec0e07, 0x3fe702e0, 0x3febacf9, 0x3fe6c16c, 0x3feb4e81, 0x3fe68168,
    0x3feaf286, 0x3fe642c8, 0x3fea98ef, 0x3fe60581, 0x3fea41a4, 0x3fe5c988, 0x3fe9ec8e, 0x3fe58ed2,
  },
  .base_tblf = {
    0x00000000, 0x3e647fbe, 0x3c7e0546, 0x3e71383b, 0x3cfc14d8, 0x3e7dc8c3, 0x3d3ba2c8, 0x3e851927,
    0x3d785186, 0x3e8b3ae5, 0x3d9a0ebd, 0x3e914a10, 0x3db78694, 0x3e974716, 0x3dd4936a, 0x3e9d3263,
    0x3df1383b, 0x3ea30c5e, 0x3e06bbf4, 0x3ea8d56c, 0x3e14aa98, 0x3eae8dee, 0x3e22695b, 0x3eb43641,
    0x3e2ff984, 0x3eb9cec0, 0x3e3d5c48, 0x3ebf57c2, 0x3e4a92d5, 0x3ec4d19c, 0x3e579e4a, 0x3eca3ca1,
  },
  .coef_tblf = {
    0x3f800000, 0x3f4ccccd, 0x3f7c0fc1, 0x3f4a4588, 0x3f783e10, 0x3f47ce0c, 0x3f74898d, 0x3f4565c8,
    0x3f70f0f1, 0x3f430c31, 0x3f6d7304, 0x3f40c0c1, 0x3f6a0ea1, 0x3f3e82fa, 0x3f66c2b4, 0x3f3c5264,
    0x3f638e39, 0x3f3a2e8c, 0x3f607038, 0x3f381703, 0x3f5d67c9, 0x3f360b61, 0x3f5a740e, 0x3f340b41,
    0x3f579436, 0x3f321643, 0x3f54c77b, 0x3f302c0b, 0x3f520d21, 0x3f2e4c41, 0x3f4f6475, 0x3f2c7692,
  }
};
template<int P, int NP, int ...Vs>
__always_inline void __svnxp_logfma(svd_t *logval, svd_t *q, svd_t *base, iseq<Vs...>){
  if (P == 0)
    __unused_arr__(logval[Vs] = svdup_f64(__log_coefs<NP - 1>::value) ...);
  else if (P == NP - 1)
    __unused_arr__(logval[Vs] = logval[Vs] * q[Vs] + base[Vs] ...);
  else {
    __unused_arr__(logval[Vs] = logval[Vs] * q[Vs] + __log_coefs<NP - 1 - P>::value ...);
  }
}
template<int ...Vs, int ...Ps>
__always_inline void __svnxp_log(svd_t *y, svd_t *x, iseq<Vs...> vseq, iseq<Ps...>){
  svsl_t expon[] = {svlogb_f64_x(svptrue_b64(), x[Vs])...};
  svd_t fract2[] = {svscale_x(svptrue_b64(), x[Vs], -expon[Vs]) ...};
  svp_t ovf[] = {svcmpgt(svptrue_b64(), fract2[Vs], svdup_f64(M_SQRT2)) ...};
  svd_t loge[] = {svadd_m(ovf[Vs], svcvt_f64_x(svptrue_b64(), (svsl_t)expon[Vs]), svdup_f64(0.5)) * M_LN2 ...};
  svd_t fract[] = {svmul_m(ovf[Vs], fract2[Vs], svdup_f64(M_SQRT1_2)) ...};
  svd_t fractm1[] = {fract[Vs] - 1.0 ...};
  svd_t fract64[] = {svrinta_x(svptrue_b64(), fractm1[Vs] * 64) ...};
  svui_t tblp[] = {reinterpret_cast<svui_t>(svcvt_u64_x(svptrue_b64(), fract64[Vs])) ...};
  svd_t anchor[] = {fract64[Vs] * (1.0/64) ...};
  svuint32x2_t base_tbll = svld2(svptrue_b32(), log_tbl.base_tbll);
  svuint32x2_t base_tblh = svld2(svptrue_b32(), log_tbl.base_tblh);
  svuint32x2_t coef_tbll = svld2(svptrue_b32(), log_tbl.coef_tbll);
  svuint32x2_t coef_tblh = svld2(svptrue_b32(), log_tbl.coef_tblh);
  svui_t basel[] = {svtbl2_u32(base_tbll, tblp[Vs]) ...};
  svui_t baseh[] = {svtbl2_u32(base_tblh, tblp[Vs]) ...};
  svui_t coefl[] = {svtbl2_u32(coef_tbll, tblp[Vs]) ...};
  svui_t coefh[] = {svtbl2_u32(coef_tblh, tblp[Vs]) ...};
  svd_t base[] = {reinterpret_cast<svd_t>(svtrn1(basel[Vs], baseh[Vs])) + loge[Vs] ...};
  svd_t coef[] = {reinterpret_cast<svd_t>(svtrn1(coefl[Vs], coefh[Vs])) ...};
  svd_t q[] = {(fractm1[Vs] - anchor[Vs]) * coef[Vs] ...};
  __unused_arr__((__svnxp_logfma<Ps, sizeof...(Ps)>(y, q, base, vseq), 0) ...);
}
template<int N, int P>
__always_inline void svnxp_log(svd_t *y, svd_t *x){
  return __svnxp_log(y, x, mkiseq<N>{}, mkiseq<P>{});
}

template<int P, int NP, int ...Vs>
__always_inline void __svnxp_expfmaf(svf_t *expval, svf_t *qqln2, iseq<Vs...>){
  if (P == 0)
    __unused_arr__(expval[Vs] = svdup_f32(__const_factorial<NP-1>::rvalue32)...);
  else
    __unused_arr__(expval[Vs] = expval[Vs] * qqln2[Vs] + __const_factorial<NP-1-P>::rvalue32...);
}

template<int ...Vs, int ...Ps>
void __svnxp_expf(svf_t *y, svf_t *x, iseq<Vs...> vseq, iseq<Ps...>) {
  svf_t xscaled[] = {x[Vs] * (float32_t)(1.0/M_LN2)...};
  svf_t x64[] = {svscale_f32_x(svptrue_b32(), xscaled[Vs], svdup_s32(6)) ...};
  svf_t x64r[] = {svrinta_f32_x(svptrue_b32(), x64[Vs])...};
  svsi_t a64ri[] = {svcvt_s32_f32_x(svptrue_b32(), x64r[Vs])...};
  svsi_t arl[] = {a64ri[Vs] & 63 ...};
  svf_t stval[] = {svexpa_f32((svui_t)(a64ri[Vs] & 63) + (0x7f << 6))...};
  svsi_t arh[] = {a64ri[Vs] >> 6 ...};
  svf_t q[] = { xscaled[Vs] - (svf_t)svcvt_f32_s32_x(svptrue_b32(), arh[Vs]) ...};
  svf_t qq[] = { q[Vs] - (svf_t)svscale_f32_x(svptrue_b32(), svcvt_f32_s32_x(svptrue_b32(), arl[Vs]), svdup_s32(-6)) ...};
  svf_t qqln2[] = { qq[Vs] * (float32_t)M_LN2 ...};
  svf_t expval[sizeof...(Vs)];
  __unused_arr__((__svnxp_expfmaf<Ps, sizeof...(Ps)>(expval, qqln2, vseq), 0) ...);
  __unused_arr__(expval[Vs] = expval[Vs] * stval[Vs]...);
  __unused_arr__(y[Vs] = svscale_f32_x(svptrue_b32(), expval[Vs], arh[Vs])...);
}

template<int V, int P>
void svnxp_expf(svf_t *y, svf_t *x) {
  __svnxp_expf(y, x, mkiseq<V>{}, mkiseq<P>{});
}

template<int P, int NP, int ...Vs>
__always_inline void __svnxp_logfmaf(svf_t *logval, svf_t *q, svf_t *base, iseq<Vs...>){
  if (P == 0)
    __unused_arr__(logval[Vs] = svdup_f32(__log_coefs<NP - 1>::value32) ...);
  else if (P == NP - 1)
    __unused_arr__(logval[Vs] = logval[Vs] * q[Vs] + base[Vs] ...);
  else
    __unused_arr__(logval[Vs] = logval[Vs] * q[Vs] + __log_coefs<NP - 1 - P>::value32 ...);
}

template<int ...Vs, int ...Ps>
__always_inline void __svnxp_logf(svf_t *y, svf_t *x, iseq<Vs...> vseq, iseq<Ps...>){
  svsi_t expon[] = {svlogb_f32_x(svptrue_b32(), x[Vs])...};
  svf_t fract2[] = {svscale_f32_x(svptrue_b32(), x[Vs], svneg_s32_x(svptrue_b32(), expon[Vs])) ...};
  svp_t ovf[] = {svcmpgt_f32(svptrue_b32(), fract2[Vs], svdup_f32((float32_t)M_SQRT2)) ...};
  svf_t loge[] = {svadd_f32_m(ovf[Vs], svcvt_f32_s32_x(svptrue_b32(), expon[Vs]), svdup_f32(0.5f)) * svdup_f32((float32_t)M_LN2) ...};
  svf_t fract[] = {svmul_f32_m(ovf[Vs], fract2[Vs], svdup_f32((float32_t)M_SQRT1_2)) ...};
  svf_t fractm1[] = {fract[Vs] - 1.0f ...};
  svf_t fract64[] = {svrinta_f32_x(svptrue_b32(), fractm1[Vs] * 64.0f) ...};
  svui_t tblp[] = {svcvt_u32_f32_x(svptrue_b32(), fract64[Vs]) ...};
  svf_t anchor[] = {fract64[Vs] * (1.0f/64.0f) ...};
  svfloat32x2_t base_tblf = svld2_f32(svptrue_b32(), reinterpret_cast<const float32_t *>(log_tbl.base_tblf));
  svfloat32x2_t coef_tblf = svld2_f32(svptrue_b32(), reinterpret_cast<const float32_t *>(log_tbl.coef_tblf));
  svf_t basef[] = {svtbl2_f32(base_tblf, tblp[Vs]) ...};
  svf_t coeff[] = {svtbl2_f32(coef_tblf, tblp[Vs]) ...};
  svf_t base[] = {basef[Vs] + loge[Vs] ...};
  svf_t coef[] = {coeff[Vs] ...};
  svf_t q[] = {(fractm1[Vs] - anchor[Vs]) * coef[Vs] ...};
  __unused_arr__((__svnxp_logfmaf<Ps, sizeof...(Ps)>(y, q, base, vseq), 0) ...);
}

template<int N, int P>
__always_inline void svnxp_logf(svf_t *y, svf_t *x){
  return __svnxp_logf(y, x, mkiseq<N>{}, mkiseq<P>{});
}

#ifdef LMFF_MIXED_PREC

#define svdup(x) svdup_f32((float)(x))
#define svld1(pg, base) svld1_f32((pg), (const float *)(base))
#define svst1(pg, base, val) svst1_f32((pg), (float *)(base), (val))
#define svaddv svaddv_f32
#define svwhilelt svwhilelt_b32
#define svptrue() svptrue_b32()
#define svnxp_exp svnxp_expf
#define svnxp_log svnxp_logf

#else

#define svdup(x) svdup_f64(x)
#define svld1(pg, base) svld1_f64((pg), (const double *)(base))
#define svst1(pg, base, val) svst1_f64((pg), (double *)(base), (val))
#define svaddv svaddv_f64
#define svwhilelt svwhilelt_b64
#define svptrue() svptrue_b64()

#endif

#endif    // LMFF_SIMD_SVE_H
