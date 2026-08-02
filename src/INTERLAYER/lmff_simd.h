/* -*- c++ -*- ----------------------------------------------------------
   LMFF SIMD abstraction: ARM SVE (512-bit) or x86 AVX-512.

   Default: FP64 compute, CLUSTERSIZE = 8.
   LMFF_MIXED_PREC: FP32 compute, FP64 force writeback, CLUSTERSIZE = 16.

   Build flags (auto-detect if unset):
     LMFF_USE_SVE     - force ARM SVE backend
     LMFF_USE_AVX512  - force AVX-512 backend
     LMFF_MIXED_PREC  - mixed-precision eval (MY_FLOAT = float)

   AVX-512 svnxp_exp/log backend (pass via CMAKE_CXX_FLAGS):
     -DLMFF_SVNX_MATH_MKL      Intel SVML _mm512_exp_pd/log_pd
     -DLMFF_SVNX_MATH_LIBMVEC  GNU libmvec _ZGVeN8v_exp/log
     (default)                 scalar std::exp/log
------------------------------------------------------------------------- */

#ifndef LMFF_SIMD_H
#define LMFF_SIMD_H

#include <cmath>
#include <cstdint>
#include <utility>

#if defined(LMFF_USE_AVX512) || (defined(__AVX512F__) && !defined(LMFF_USE_SVE))
#define LMFF_SIMD_AVX512 1
#elif defined(LMFF_USE_SVE) || defined(__ARM_FEATURE_SVE)
#define LMFF_SIMD_SVE 1
#else
#error "LMFF requires LMFF_USE_SVE / LMFF_USE_AVX512 or a compiler with SVE or AVX-512F"
#endif

#ifdef LMFF_MIXED_PREC
#undef LMFF_CLUSTER_WIDTH
#define LMFF_CLUSTER_WIDTH 16
#elif !defined(LMFF_CLUSTER_WIDTH)
#define LMFF_CLUSTER_WIDTH 8
#endif

#ifdef LMFF_MIXED_PREC
typedef float MY_FLOAT;
#else
typedef double MY_FLOAT;
#endif

#if LMFF_SIMD_SVE
#include "lmff_simd_sve.h"
#elif LMFF_SIMD_AVX512
#include "lmff_simd_avx512.h"
#endif

/* ---------------------------------------------------------------------- */
/* Pair LMFF helpers (mixed / FP64 selected here, not in pair_LMFF)        */
/* ---------------------------------------------------------------------- */

#ifdef LMFF_MIXED_PREC
typedef svsi_t MY_TAGINT;
#else
typedef svsl_t MY_TAGINT;
#endif

__always_inline void lmff_load_cluster_xq(svp_t predi32, svp_t predi64, svsi_t is32,
                                                svsl_t is64, int iicnt, double (*x)[3], double *q,
                                                MY_VEC &xi, MY_VEC &yi, MY_VEC &zi, MY_VEC &qi)
{
#ifdef LMFF_MIXED_PREC
  svp_t predi64lo = svwhilelt_b64(0, iicnt);
  svp_t predi64hi = svwhilelt_b64(8, iicnt);
  svsl_t is64lo = svunpklo(is32);
  svsl_t is64hi = svunpkhi(is32);
  svd_t xilo = svld1_gather_offset(predi64lo, (double *) x, is64lo * 24);
  svd_t yilo = svld1_gather_offset(predi64lo, (double *) x, is64lo * 24 + 8);
  svd_t zilo = svld1_gather_offset(predi64lo, (double *) x, is64lo * 24 + 16);
  svd_t xihi = svld1_gather_offset(predi64hi, (double *) x, is64hi * 24);
  svd_t yihi = svld1_gather_offset(predi64hi, (double *) x, is64hi * 24 + 8);
  svd_t zihi = svld1_gather_offset(predi64hi, (double *) x, is64hi * 24 + 16);
  svd_t qilo = svld1_gather_index(predi64lo, q, is64lo);
  svd_t qihi = svld1_gather_index(predi64hi, q, is64hi);
  xi = svuzp1(svcvt_f32_f64_x(predi64lo, xilo), svcvt_f32_f64_x(predi64hi, xihi));
  yi = svuzp1(svcvt_f32_f64_x(predi64lo, yilo), svcvt_f32_f64_x(predi64hi, yihi));
  zi = svuzp1(svcvt_f32_f64_x(predi64lo, zilo), svcvt_f32_f64_x(predi64hi, zihi));
  qi = svuzp1(svcvt_f32_f64_x(predi64lo, qilo), svcvt_f32_f64_x(predi64hi, qihi));
#else
  xi = svld1_gather_offset(predi64, (double *) x, is64 * 24);
  yi = svld1_gather_offset(predi64, (double *) x, is64 * 24 + 8);
  zi = svld1_gather_offset(predi64, (double *) x, is64 * 24 + 16);
  qi = svld1_gather_index(predi64, q, is64);
#endif
}

__always_inline MY_TAGINT lmff_load_itag0(svp_t predi32, svp_t predi64, const int *tag,
                                                   svsi_t is32, svsl_t is64)
{
#ifdef LMFF_MIXED_PREC
  return svld1_gather_index(predi32, tag, is32);
#else
  return svld1sw_gather_index_s64(predi64, tag, is64);
#endif
}

__always_inline MY_TAGINT lmff_dup_tag_j(int tag_j)
{
#ifdef LMFF_MIXED_PREC
  return svdup_s32(tag_j);
#else
  return svdup_s64(tag_j);
#endif
}

__always_inline svp_t lmff_pred_same_molecule(svp_t predi32, svp_t predi64,
                                                     svsi_t atom_molecule_i, int mol_j)
{
#ifdef LMFF_MIXED_PREC
  return svcmpne(predi32, atom_molecule_i, svdup_s32(mol_j));
#else
  return svcmpne(predi64, svunpklo(atom_molecule_i), svunpklo(svdup_s32(mol_j)));
#endif
}

__always_inline svp_t lmff_check_vdw(svp_t pred, MY_TAGINT itag, MY_TAGINT jtag,
                                            MY_VEC xi, MY_VEC yi, MY_VEC zi, MY_VEC xj, MY_VEC yj,
                                            MY_VEC zj)
{
#ifdef LMFF_MIXED_PREC
  svsi_t tag_xor = sveor_z(pred, itag, jtag);
  svsi_t bit0 = svand_z(pred, tag_xor, svdup_s32(1));
  svp_t is_odd = svcmpne(pred, bit0, svdup_s32(0));
  svp_t gt = svcmpgt(pred, itag, jtag);
  svp_t ne = svcmpne(pred, itag, jtag);
#else
  svsl_t tag_xor = sveor_z(pred, itag, jtag);
  svsl_t bit0 = svand_z(pred, tag_xor, svdup_s64(1));
  svp_t is_odd = svcmpne(pred, bit0, svdup_s64(0));
  svp_t gt = svcmpgt(pred, itag, jtag);
  svp_t ne = svcmpne(pred, itag, jtag);
#endif
  svp_t skip_tail = sveor_z(pred, gt, is_odd);
  svp_t skip_1_2 = svand_z(pred, ne, skip_tail);
  svp_t eq = svcmpeq(pred, itag, jtag);
  svp_t cond_z = svcmplt(pred, zj, zi);
  svp_t same_z = svcmpeq(pred, zj, zi);
  svp_t cond_y = svcmplt(pred, yj, yi);
  svp_t same_y = svcmpeq(pred, yj, yi);
  svp_t cond_x = svcmplt(pred, xj, xi);
  svp_t y_or_x = svorr_z(pred, cond_y, svand_z(pred, same_y, cond_x));
  svp_t skip3 = svorr_z(pred, cond_z, svand_z(pred, same_z, y_or_x));
  skip3 = svand_z(pred, eq, skip3);
  return svorr_z(pred, skip_1_2, skip3);
}

__always_inline void lmff_scatter_i_forces(svp_t predi64, svsi_t is32, svsl_t is64, int iicnt,
                                                  MY_VEC fxi, MY_VEC fyi, MY_VEC fzi, MY_VEC zero,
                                                  double (*f)[3])
{
#ifdef LMFF_MIXED_PREC
  svp_t predi64lo = svwhilelt_b64(0, iicnt);
  svp_t predi64hi = svwhilelt_b64(8, iicnt);
  svsl_t is64lo = svunpklo(is32);
  svsl_t is64hi = svunpkhi(is32);
  svd_t fxilo = svcvt_f64_f32_x(predi64lo, svzip1(fxi, zero));
  svd_t fyilo = svcvt_f64_f32_x(predi64lo, svzip1(fyi, zero));
  svd_t fzilo = svcvt_f64_f32_x(predi64lo, svzip1(fzi, zero));
  svd_t fxihi = svcvt_f64_f32_x(predi64hi, svzip2(fxi, zero));
  svd_t fyihi = svcvt_f64_f32_x(predi64hi, svzip2(fyi, zero));
  svd_t fzihi = svcvt_f64_f32_x(predi64hi, svzip2(fzi, zero));
  svd_t cfxilo = svld1_gather_offset(predi64lo, (double *) f, is64lo * 24);
  svd_t cfyilo = svld1_gather_offset(predi64lo, (double *) f, is64lo * 24 + 8);
  svd_t cfzilo = svld1_gather_offset(predi64lo, (double *) f, is64lo * 24 + 16);
  svd_t cfxihi = svld1_gather_offset(predi64hi, (double *) f, is64hi * 24);
  svd_t cfyihi = svld1_gather_offset(predi64hi, (double *) f, is64hi * 24 + 8);
  svd_t cfzihi = svld1_gather_offset(predi64hi, (double *) f, is64hi * 24 + 16);
  svst1_scatter_offset(predi64lo, (double *) f, is64lo * 24, cfxilo + fxilo);
  svst1_scatter_offset(predi64hi, (double *) f, is64hi * 24, cfxihi + fxihi);
  svst1_scatter_offset(predi64lo, (double *) f, is64lo * 24 + 8, cfyilo + fyilo);
  svst1_scatter_offset(predi64hi, (double *) f, is64hi * 24 + 8, cfyihi + fyihi);
  svst1_scatter_offset(predi64lo, (double *) f, is64lo * 24 + 16, cfzilo + fzilo);
  svst1_scatter_offset(predi64hi, (double *) f, is64hi * 24 + 16, cfzihi + fzihi);
#else
  MY_VEC cfxi = svld1_gather_offset(predi64, (double *) f, is64 * 24);
  MY_VEC cfyi = svld1_gather_offset(predi64, (double *) f, is64 * 24 + 8);
  MY_VEC cfzi = svld1_gather_offset(predi64, (double *) f, is64 * 24 + 16);
  svst1_scatter_offset(predi64, (double *) f, is64 * 24, cfxi + fxi);
  svst1_scatter_offset(predi64, (double *) f, is64 * 24 + 8, cfyi + fyi);
  svst1_scatter_offset(predi64, (double *) f, is64 * 24 + 16, cfzi + fzi);
#endif
}

#endif    // LMFF_SIMD_H
