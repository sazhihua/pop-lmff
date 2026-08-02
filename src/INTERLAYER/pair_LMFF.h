/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(LMFF, PairLMFF);
// clang-format on
#else

#ifndef LMP_LMFF_H
#define LMP_LMFF_H

#include "pair.h"
#include "lmff_simd.h"

namespace LAMMPS_NS {

static constexpr MY_FLOAT Tap_coeff[8] = {1.0, 0.0, 0.0, 0.0, -35.0, 84.0, -70.0, 20.0};

class PairLMFF : public Pair {
 public:
  PairLMFF(class LAMMPS *);
  virtual ~PairLMFF();

  virtual void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  double single(int, int, int, int, double, double, double, double &) override;

  static constexpr int NPARAMS_PER_LINE = 13;

  enum {
    ILP_GrhBN,
    ILP_TMD,
    SAIP_METAL,
    AIP_WATER_2DM
  };    // for telling class variants apart in shared code

 protected:
  int me;
  int variant;
  int maxlocal;            // size of numneigh, firstneigh arrays
  int pgsize;              // size of neighbor page
  int oneatom;             // max # of neighbors for one atom
  MyPage<int> *ipage;      // neighbor list pages
  int *ILP_numneigh;       // # of pair neighbors for each atom
  int **ILP_firstneigh;    // ptr to 1st neighbor of each atom
  int tap_flag;            // flag to turn on/off taper function

  struct Param {
    MY_FLOAT z0, alpha, epsilon, C, delta, d, sR, reff, C6, S;
    MY_FLOAT delta2inv, seff, lambda, rcut;
    int ielement, jelement;
  };

  Param *params;    // parameter set for I-J interactions
  int nmax;         // max # of atoms

  MY_FLOAT cut_global;
  MY_FLOAT cut_normal;
  MY_FLOAT **cutILPsq;
  MY_FLOAT **offset;
  MY_FLOAT **normal;
  MY_FLOAT ***dnormdri;
  MY_FLOAT ****dnormal;

  // adds for ilp/tmd
  int Nnei;
  MY_FLOAT **dnn;
  MY_FLOAT **vect;
  MY_FLOAT **pvet;
  MY_FLOAT ***dpvet1;
  MY_FLOAT ***dpvet2;
  MY_FLOAT ***dNave;

  MY_FLOAT **sigmae_map;
  int **coul_setflag;
  void read_file(char *);
  void allocate();

  void update_internal_list();
  template <int MAX_NNEIGH>
  void calc_atom_normal(int i, int itype, int *ILP_neigh, int nneigh, MY_FLOAT &normalx, MY_FLOAT &normaly,
                        MY_FLOAT &normalz, MY_FLOAT &dnormdri0, MY_FLOAT &dnormdri1, MY_FLOAT &dnormdri2,
                        MY_FLOAT &dnormdri3, MY_FLOAT &dnormdri4, MY_FLOAT &dnormdri5, MY_FLOAT &dnormdri6,
                        MY_FLOAT &dnormdri7, MY_FLOAT &dnormdri8, MY_FLOAT (*dnormdrk)[9]);

  template <int MAX_NNEIGH, int EFLAG, int VFLAG_EITHER, int TAP_FLAG, int VARIANT = ILP_GrhBN>
  void eval();
  int *layered_neigh;
  int **first_layered_neigh;
  int *bundled_neigh;
  int **first_bundled_neigh;
  int *special_type;
  int *num_intra, *num_inter, *num_vdw;
  int inum_max, jnum_max;

  int shift_flag;
  MY_FLOAT shift;

  /* for omp */
  double (*frep)[3];
  double *eng_rep;
  int num_omp_threads;

  void lmff_omp_alloc(int nlocal, int nghost);
  void lmff_omp_free();

  enum special_type_const {
    NOT_SPECIAL = 0,
    TMD_METAL,
    SAIP_BNCH,
    WATER,
  };

  __always_inline static MY_VEC calc_Tap_sve(svp_t pg, MY_VEC r_ij, MY_VEC Rcut)
  {
    MY_VEC r = r_ij / Rcut;
    svp_t inrange = svcmplt(pg, r, svdup(1.0));
    MY_VEC Tap = svdup(Tap_coeff[7]);
    Tap = svmad_x(inrange, Tap, r, svdup(Tap_coeff[6]));
    Tap = svmad_x(inrange, Tap, r, svdup(Tap_coeff[5]));
    Tap = svmad_x(inrange, Tap, r, svdup(Tap_coeff[4]));
    Tap = svmad_x(inrange, Tap, r, svdup(Tap_coeff[3]));
    Tap = svmad_x(inrange, Tap, r, svdup(Tap_coeff[2]));
    Tap = svmad_x(inrange, Tap, r, svdup(Tap_coeff[1]));
    Tap = svmad_x(inrange, Tap, r, svdup(Tap_coeff[0]));
    Tap = svsel(inrange, Tap, svdup(0.0));
    return Tap;
  }

  __always_inline static MY_VEC calc_dTap_sve(svp_t pg, MY_VEC r_ij, MY_VEC Rcut)
  {
    MY_VEC r = r_ij / Rcut;
    svp_t inrange = svcmplt(pg, r, svdup(1.0));
    MY_VEC dTap = svdup(7.0 * Tap_coeff[7]);
    dTap = svmad_x(inrange, dTap, r, svdup(6.0 * Tap_coeff[6]));
    dTap = svmad_x(inrange, dTap, r, svdup(5.0 * Tap_coeff[5]));
    dTap = svmad_x(inrange, dTap, r, svdup(4.0 * Tap_coeff[4]));
    dTap = svmad_x(inrange, dTap, r, svdup(3.0 * Tap_coeff[3]));
    dTap = svmad_x(inrange, dTap, r, svdup(2.0 * Tap_coeff[2]));
    dTap = svmad_x(inrange, dTap, r, svdup(Tap_coeff[1]));
    dTap = dTap / Rcut;
    dTap = svsel(inrange, dTap, svdup(0.0));
    return dTap;
  }

};

}    // namespace LAMMPS_NS

#endif
#endif

