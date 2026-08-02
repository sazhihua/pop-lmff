/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   This is an optimized version of ilp/graphene/hbn based on the contirubtion of:
   author: Wengen Ouyang (Wuhan University, China)
   e-mail: w.g.ouyang at gmail dot com

   Optimizations are done by:
   author1: Ping Gao (National Supercomputing Center in Wuxi, China) implements the base ILP potential.
   e-mail: qdgaoping at gmail dot com

   author2: Xiaohui Duan (Shandong University, China) adjusts the framework to adopt SAIP, TMD, WATER2DM, etc.
   e-mail: sunrise_duan at 126 dot com

   Optimizations are described in:
   Gao, Ping and Duan, Xiaohui, et al:
   LMFF: Efficient and Scalable Layered Materials Force Field on Heterogeneous Many-Core Processors
   DOI: 10.1145/3458817.3476137

   Potential is described by:
   [Ouyang et al., J. Chem. Theory Comput. 16(1), 666-676 (2020)]
*/

#include "pair_LMFF.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "interlayer_taper.h"
#include "math_special.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "my_page.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pointers.h"
#include "potential_file_reader.h"
#include "gptl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <omp.h>

// Diagnostic build: comment out this line to remove all cluster statistics.
// #define LMFF_CLUSTER_STATS

#define CLUSTERSIZE LMFF_CLUSTER_WIDTH

using namespace LAMMPS_NS;
using namespace InterLayer;
using namespace MathConst;
using namespace MathSpecial;
using namespace MathExtra;

static constexpr int DELTA = 4;
static constexpr int PGDELTA = 1;
static constexpr int LMFF_ATOM_BLK = 64;
static constexpr int LMFF_ENG_STRIDE = 6;

static inline size_t lmff_upalign(size_t v, size_t align)
{
  return (v + align - 1) & ~(align - 1);
}

#ifdef LMFF_CLUSTER_STATS
static inline unsigned long long lmff_count_active_lanes(svp_t pred)
{
#if LMFF_SIMD_AVX512
  return static_cast<unsigned long long>(
      __builtin_popcount(static_cast<unsigned int>(pred)));
#elif defined(LMFF_MIXED_PREC)
  return static_cast<unsigned long long>(svcntp_b32(svptrue_b32(), pred));
#else
  return static_cast<unsigned long long>(svcntp_b64(svptrue_b64(), pred));
#endif
}
#endif

static const char cite_ilp_cur[] =
    "ilp/graphene/hbn/opt potential doi:10.1145/3458817.3476137\n"
    "@inproceedings{gao2021lmff\n"
    " author = {Gao, Ping and Duan, Xiaohui and Others},\n"
    " title = {LMFF: Efficient and Scalable Layered Materials Force Field on Heterogeneous "
    "Many-Core Processors},\n"
    " year = {2021},\n"
    " isbn = {9781450384421},\n"
    " publisher = {Association for Computing Machinery},\n"
    " address = {New York, NY, USA},\n"
    " url = {https://doi.org/10.1145/3458817.3476137},\n"
    " doi = {10.1145/3458817.3476137},\n"
    " booktitle = {Proceedings of the International Conference for High Performance Computing, "
    "Networking, Storage and Analysis},\n"
    " articleno = {42},\n"
    " numpages = {14},\n"
    " location = {St. Louis, Missouri},\n"
    " series = {SC'21},\n"
    "}\n\n";


// to indicate which potential style was used in outputs
static std::map<int, std::string> variant_map = {
    {PairLMFF::ILP_GrhBN, "ilp/graphene/hbn"},
    {PairLMFF::ILP_TMD, "ilp/tmd"},
    {PairLMFF::AIP_WATER_2DM, "aip/water/2dm"},
    {PairLMFF::SAIP_METAL, "saip/metal"}};

/* ---------------------------------------------------------------------- */

PairLMFF::PairLMFF(LAMMPS *lmp) : Pair(lmp), variant(ILP_GrhBN),
    layered_neigh(nullptr), first_layered_neigh(nullptr),
    bundled_neigh(nullptr), first_bundled_neigh(nullptr),
    special_type(nullptr), num_intra(nullptr), num_inter(nullptr), num_vdw(nullptr)
{
  restartinfo = 0;
  one_coeff = 0;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);

  if (lmp->citeme) lmp->citeme->add(cite_ilp_cur);

  nextra = 4;
  pvector = new double[nextra];

  // initialize element to parameter maps
  params = nullptr;
  cutILPsq = nullptr;

  nmax = 0;
  maxlocal = 0;
  ILP_numneigh = nullptr;
  ILP_firstneigh = nullptr;
  ipage = nullptr;
  pgsize = oneatom = 0;

  normal = nullptr;
  dnormal = nullptr;
  dnormdri = nullptr;

  // for ilp/tmd
  dnn = nullptr;
  vect = nullptr;
  pvet = nullptr;
  dpvet1 = nullptr;
  dpvet2 = nullptr;
  dNave = nullptr;

  // always compute energy offset
  offset_flag = 1;

  // turn on the taper function by default
  tap_flag = 1;

  single_enable = 0;
  inum_max = 0;
  jnum_max = 0;

  frep = nullptr;
  eng_rep = nullptr;
  num_omp_threads = 0;
}

/* ---------------------------------------------------------------------- */

PairLMFF::~PairLMFF()
{
  lmff_omp_free();
  memory->destroy(layered_neigh);
  memory->sfree(first_layered_neigh);
  memory->destroy(bundled_neigh);
  memory->sfree(first_bundled_neigh);
  memory->destroy(num_intra);
  memory->destroy(num_inter);
  memory->destroy(num_vdw);
  memory->destroy(special_type);
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLMFF::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(offset, n, n, "pair:offset");
  memory->create(sigmae_map, n, n, "pair:sigmae_map");
  memory->create(coul_setflag, n, n, "pair:coul_setflag");
  for (int i = 1; i < n; i++)
    for (int j = 1; j < n; j++) {
      sigmae_map[i][j] = 0.0;
      coul_setflag[i][j] = 0;
    }
  map = new int[n];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLMFF::settings(int narg, char **arg)
{
  if (narg < 1 || narg > 2) error->all(FLERR, "Illegal pair_style command");
  if (!utils::strmatch(force->pair_style, "^hybrid/overlay"))
    error->all(FLERR, "Pair style ilp/graphene/hbn must be used as sub-style with hybrid/overlay");

  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  if (narg == 2) tap_flag = utils::numeric(FLERR, arg[1], false, lmp);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLMFF::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  // Coulomb shielding params for specific type pairs (merged coul/shield)
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0) {
    if (narg < 3 || narg > 4)
      error->all(FLERR, "Incorrect args for pair coefficients" + utils::errorurl(21));

    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

    MY_FLOAT sigmae_one = utils::numeric(FLERR, arg[2], false, lmp);
    MY_FLOAT sigmae3 = sigmae_one * sigmae_one * sigmae_one;

    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo, i); j <= jhi; j++) {
        sigmae_map[i][j] = sigmae3;
        sigmae_map[j][i] = sigmae3;
        coul_setflag[i][j] = 1;
        coul_setflag[j][i] = 1;
        count++;
      }
    }

    if (count == 0)
      error->all(FLERR, "Incorrect args for pair coefficients" + utils::errorurl(21));
    return;
  }

  if (narg < 4)
    error->all(FLERR, "Incorrect args for pair coefficients" + utils::errorurl(21));

  // set elem3param for all element triplet combinations
  // must be a single exact match to lines read from file
  // do not allow for ACB in place of ABC

  shift_flag = 0;
  shift = 0.0;

  map_element2type(narg - 3, arg + 3);
  read_file(arg[2]);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLMFF::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  if (!offset_flag) error->all(FLERR, "Must use 'pair_modify shift yes' with this pair style");

  if (offset_flag && (cut_global > 0.0)) {
    int iparam_ij = elem2param[map[i]][map[j]];
    Param &p = params[iparam_ij];
    offset[i][j] =
        -p.C6 * pow(1.0 / cut_global, 6) / (1.0 + exp(-p.d * (cut_global / p.seff - 1.0)));
  } else
    offset[i][j] = 0.0;
  offset[j][i] = offset[i][j];

  return cut_global;
}

/* ----------------------------------------------------------------------
   read Interlayer potential file
------------------------------------------------------------------------- */

void PairLMFF::read_file(char *filename)
{
  memory->sfree(params);
  params = nullptr;
  nparams = maxparam = 0;

  // open file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, filename, variant_map[variant], unit_convert_flag);
    char *line;

    // transparently convert units for supported conversions

    int unit_convert = reader.get_unit_convert();
    MY_FLOAT conversion_factor = utils::get_conversion_factor(utils::ENERGY, unit_convert);

    while ((line = reader.next_line(NPARAMS_PER_LINE))) {

      try {
        ValueTokenizer values(line);

        std::string iname = values.next_string();
        std::string jname = values.next_string();

        // ielement,jelement = 1st args
        // if both args are in element list, then parse this line
        int ielement, jelement;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;
        if (ielement == nelements) continue;
        for (jelement = 0; jelement < nelements; jelement++)
          if (jname == elements[jelement]) break;
        if (jelement == nelements) continue;

        // expand storage, if needed

        if (nparams == maxparam) {
          maxparam += DELTA;
          params = (Param *) memory->srealloc(params, maxparam * sizeof(Param), "pair:params");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(params + nparams, 0, DELTA * sizeof(Param));
        }

        // load up parameter settings and error check their values

        params[nparams].ielement = ielement;
        params[nparams].jelement = jelement;
        params[nparams].z0 = values.next_double();
        params[nparams].alpha = values.next_double();
        params[nparams].delta = values.next_double();
        params[nparams].epsilon = values.next_double();
        params[nparams].C = values.next_double();
        params[nparams].d = values.next_double();
        params[nparams].sR = values.next_double();
        params[nparams].reff = values.next_double();
        params[nparams].C6 = values.next_double();
        // S provides a convenient scaling of all energies
        params[nparams].S = values.next_double();
        params[nparams].rcut = values.next_double();

      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }

      // energies in meV further scaled by S
      // S = 43.3634 meV = 1 kcal/mol

      MY_FLOAT meV = 1e-3 * params[nparams].S;
      if (unit_convert) meV *= conversion_factor;

      params[nparams].C *= meV;
      params[nparams].C6 *= meV;
      params[nparams].epsilon *= meV;

      // precompute some quantities
      params[nparams].delta2inv = pow(params[nparams].delta, -2.0);
      params[nparams].lambda = params[nparams].alpha / params[nparams].z0;
      params[nparams].seff = params[nparams].sR * params[nparams].reff;

      nparams++;
    }
  }

  MPI_Bcast(&nparams, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam, 1, MPI_INT, 0, world);

  if (comm->me != 0) {
    params = (Param *) memory->srealloc(params, maxparam * sizeof(Param), "pair:params");
  }

  MPI_Bcast(params, maxparam * sizeof(Param), MPI_BYTE, 0, world);

  memory->destroy(elem2param);
  memory->destroy(cutILPsq);
  memory->create(elem2param, nelements, nelements, "pair:elem2param");
  memory->create(cutILPsq, nelements, nelements, "pair:cutILPsq");
  for (int i = 0; i < nelements; i++) {
    for (int j = 0; j < nelements; j++) {
      int n = -1;
      for (int m = 0; m < nparams; m++) {
        if (i == params[m].ielement && j == params[m].jelement) {
          if (n >= 0)
            error->all(FLERR, "{} potential file {} has a duplicate entry for: {} {}",
                       variant_map[variant], filename, elements[i], elements[j]);
          n = m;
        }
      }
      if (n < 0)
        error->all(FLERR, "{} potential file {} is missing an entry for: {} {}",
                   variant_map[variant], filename, elements[i], elements[j]);
      elem2param[i][j] = n;
      cutILPsq[i][j] = params[n].rcut * params[n].rcut;
    }
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLMFF::init_style()
{
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style ilp/graphene/hbn requires newton pair on");
  if (!atom->molecule_flag)
    error->all(FLERR, "Pair style ilp/graphene/hbn requires atom attribute molecule");

  int coul_flag = 0;
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      if (coul_setflag[i][j]) coul_flag = 1;
  if (coul_flag && !atom->q_flag)
    error->all(FLERR, "Pair style LMFF requires atom attribute q when Coulomb shielding is used");

  // It seems that ghost neighbors is required for some historical reason, and it is not needed now

  neighbor->add_request(this, NeighConst::REQ_FULL);

  if (comm->me == 0) {
#ifdef LMFF_MIXED_PREC
    error->warning(FLERR, "LMFF: using FP32 compute precision ...\n");
#else
    error->warning(FLERR, "LMFF: using FP64 compute precision ...\n");
#endif
  }
}

/* ---------------------------------------------------------------------- */

void PairLMFF::compute(int eflag, int vflag)
{
  GPTLstart("LMFF");
  ev_init(eflag, vflag);
  pvector[0] = pvector[1] = pvector[2] = pvector[3] = 0.0;

  if (neighbor->ago == 0) update_internal_list();

  if (variant == ILP_GrhBN) {
    if (eflag_global || eflag_atom) {
      if (vflag_either) {
        if (tap_flag) {
          eval<3, 1, 1, 1>();
        } else {
          eval<3, 1, 1, 0>();
        }
      } else {
        if (tap_flag) {
          eval<3, 1, 0, 1>();
        } else {
          eval<3, 1, 0, 0>();
        }
      }
    } else {
      if (vflag_either) {
        if (tap_flag) {
          eval<3, 0, 1, 1>();
        } else {
          eval<3, 0, 1, 0>();
        }
      } else {
        if (tap_flag) {
          eval<3, 0, 0, 1>();
        } else {
          eval<3, 0, 0, 0>();
        }
      }
    }
  } else if (variant == ILP_TMD) {
    if (eflag_global || eflag_atom) {
      if (vflag_either) {
        if (tap_flag) {
          eval<6, 1, 1, 1, ILP_TMD>();
        } else {
          eval<6, 1, 1, 0, ILP_TMD>();
        }
      } else {
        if (tap_flag) {
          eval<6, 1, 0, 1, ILP_TMD>();
        } else {
          eval<6, 1, 0, 0, ILP_TMD>();
        }
      }
    } else {
      if (vflag_either) {
        if (tap_flag) {
          eval<6, 0, 1, 1, ILP_TMD>();
        } else {
          eval<6, 0, 1, 0, ILP_TMD>();
        }
      } else {
        if (tap_flag) {
          eval<6, 0, 0, 1, ILP_TMD>();
        } else {
          eval<6, 0, 0, 0, ILP_TMD>();
        }
      }
    }
  } else if (variant == AIP_WATER_2DM) {
    if (eflag_global || eflag_atom) {
      if (vflag_either) {
        if (tap_flag) {
          eval<6, 1, 1, 1, AIP_WATER_2DM>();
        } else {
          eval<6, 1, 1, 0, AIP_WATER_2DM>();
        }
      } else {
        if (tap_flag) {
          eval<6, 1, 0, 1, AIP_WATER_2DM>();
        } else {
          eval<6, 1, 0, 0, AIP_WATER_2DM>();
        }
      }
    } else {
      if (vflag_either) {
        if (tap_flag) {
          eval<6, 0, 1, 1, AIP_WATER_2DM>();
        } else {
          eval<6, 0, 1, 0, AIP_WATER_2DM>();
        }
      } else {
        if (tap_flag) {
          eval<6, 0, 0, 1, AIP_WATER_2DM>();
        } else {
          eval<6, 0, 0, 0, AIP_WATER_2DM>();
        }
      }
    }
  } else if (variant == SAIP_METAL) {
    if (eflag_global || eflag_atom) {
      if (vflag_either) {
        if (tap_flag) {
          eval<3, 1, 1, 1, SAIP_METAL>();
        } else {
          eval<3, 1, 1, 0, SAIP_METAL>();
        }
      } else {
        if (tap_flag) {
          eval<3, 1, 0, 1, SAIP_METAL>();
        } else {
          eval<3, 1, 0, 0, SAIP_METAL>();
        }
      }
    } else {
      if (vflag_either) {
        if (tap_flag) {
          eval<3, 0, 1, 1, SAIP_METAL>();
        } else {
          eval<3, 0, 1, 0, SAIP_METAL>();
        }
      } else {
        if (tap_flag) {
          eval<3, 0, 0, 1, SAIP_METAL>();
        } else {
          eval<3, 0, 0, 0, SAIP_METAL>();
        }
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();

  // pvector[0]=vdw, [1]=rep, [2]=ILP total, [3]=Coulomb
  if (eflag_global) pvector[2] = pvector[0] + pvector[1];

  GPTLstop("LMFF");
}

/* ------------------------------------------------------------------------ */

__always_inline static bool check_vdw(int itag, int jtag, MY_FLOAT *xi, MY_FLOAT *xj)
{
  if (itag > jtag) {
    if ((itag + jtag) % 2 == 0) return false;
  } else if (itag < jtag) {
    if ((itag + jtag) % 2 == 1) return false;
  } else {
    if (xj[2] < xi[2]) return false;
    if (xj[2] == xi[2] && xj[1] < xi[1]) return false;
    if (xj[2] == xi[2] && xj[1] == xi[1] && xj[0] < xi[0]) return false;
  }
  return true;
}

/* ------------------------------------------------------------------------ */

void PairLMFF::lmff_omp_alloc(int nlocal, int nghost)
{
  int max_threads = omp_get_max_threads();
  int nall = (nlocal + nghost) * 1.2;
  int nall_upalign = (nall + LMFF_ATOM_BLK - 1) / LMFF_ATOM_BLK * LMFF_ATOM_BLK;

  if (nmax < nall_upalign || num_omp_threads < max_threads) {
    lmff_omp_free();
    size_t frep_bytes = lmff_upalign(nall_upalign * max_threads * 3 * sizeof(double), 256);
    size_t eng_bytes = lmff_upalign(max_threads * LMFF_ENG_STRIDE * sizeof(double), 256);
    frep = (double(*)[3]) aligned_alloc(256, frep_bytes);
    eng_rep = (double*) aligned_alloc(256, eng_bytes);
    if (!frep || !eng_rep) error->one(FLERR, "LMFF: aligned_alloc failed for OpenMP buffers");
    memset(eng_rep, 0, eng_bytes);
    nmax = nall_upalign;
    num_omp_threads = max_threads;
  }
}

void PairLMFF::lmff_omp_free()
{
  if (frep) free((void*) frep);
  if (eng_rep) free((void*) eng_rep);
  frep = nullptr;
  eng_rep = nullptr;
  nmax = 0;
  num_omp_threads = 0;
}

/* ---------------------------------------------------------------------- */

template <int MAX_NNEIGH, int EFLAG, int VFLAG_EITHER, int TAP_FLAG, int VARIANT>
void PairLMFF::eval()
{
  constexpr int EVFLAG = EFLAG || VFLAG_EITHER;

  double evdwl, ecoul;
  evdwl = 0.0;
  ecoul = 0.0;

  double (*x)[3] = (double(*)[3])atom->x[0];
  double *q = atom->q;

  int *type = atom->type;
  int ntypes = atom->ntypes;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;
  int *tag = (int*)atom->tag;

  MY_FLOAT fp1x, fp1y, fp1z;
  double fi[3], fj[3], fk[3] = {0.0, 0.0, 0.0};
  MY_FLOAT cij;

  int inum = list->inum;
  int *ilist = list->ilist;

  int num_clusters = (inum + CLUSTERSIZE - 1) / CLUSTERSIZE;

  MY_VEC zero = svdup(0.0);
  MY_VEC one = svdup(1.0);
  MY_VEC two = svdup(2.0);
  MY_VEC one_third = svdup(-0.33333333333333333);
  MY_VEC vhalf = svdup(0.5);
  MY_VEC six = svdup(6.0);
  MY_VEC qqrd2e0 = svdup(qqrd2e);

  GPTLstart("omp alloc");
  int nghost = atom->nghost;
  lmff_omp_alloc(nlocal, nghost);
  int nall = nlocal + nghost;
  int nall_upalign = (nall + LMFF_ATOM_BLK - 1) / LMFF_ATOM_BLK * LMFF_ATOM_BLK;
  double (*force)[3] = (double(*)[3])atom->f[0];
  GPTLstop("omp alloc");

#ifdef LMFF_CLUSTER_STATS
  unsigned long long stat_compute_slots = 0;
  unsigned long long stat_active_lanes = 0;
#endif

  GPTLstart("omp parallel");
#ifdef LMFF_CLUSTER_STATS
  #pragma omp parallel reduction(+ : stat_compute_slots, stat_active_lanes)
#else
  #pragma omp parallel
#endif
  {
    int i, j, ii, jj, itype, itype_map, jtype, k, kk;
    MY_FLOAT prodnorm1, fkcx, fkcy, fkcz;
    MY_FLOAT xtmp, ytmp, ztmp, delx, dely, delz, fpair, fpair1;
    MY_FLOAT rsq, r, Rcut, rhosq1, exp0, exp1, Tap, dTap, Vilp;
    MY_FLOAT frho1, Erep, fsum, rdsq1;
    int itag, jtag;
    int iparam_ij;
    MY_FLOAT rsq1, rsq2;
    MY_FLOAT r1_hat[3], r2_hat[3];

    MY_FLOAT r3, th, depsdr, epsr, forcecoul, factor_coul, Vc, fvc;
    
    int tid = omp_get_thread_num();
    double (*f)[3] = frep + nall_upalign * tid;
    memset(f, 0, nall_upalign * 3 * sizeof(double));
    double l_pvector[4] = {0.0, 0.0, 0.0, 0.0};
    double l_eng_vdwl = 0.0;
    double l_eng_coul = 0.0;

    #pragma omp for
    for (int ic = 0; ic < num_clusters; ic++) {
      int iis = ic * CLUSTERSIZE;
      int iie = std::min(iis+CLUSTERSIZE, inum);
      int iicnt = iie - iis;

      MY_FLOAT normx[CLUSTERSIZE], normy[CLUSTERSIZE], normz[CLUSTERSIZE];
      int ILP_neigh[CLUSTERSIZE][MAX_NNEIGH];
      int ILP_nneigh[CLUSTERSIZE] = {0};
      MY_FLOAT dnormdxi[CLUSTERSIZE][9];
      MY_FLOAT dnormdxi0[CLUSTERSIZE], dnormdxi1[CLUSTERSIZE], dnormdxi2[CLUSTERSIZE], dnormdxi3[CLUSTERSIZE], dnormdxi4[CLUSTERSIZE];
      MY_FLOAT dnormdxi5[CLUSTERSIZE], dnormdxi6[CLUSTERSIZE], dnormdxi7[CLUSTERSIZE], dnormdxi8[CLUSTERSIZE];
      MY_FLOAT dnormdxk[CLUSTERSIZE][MAX_NNEIGH][9];
      int itypes[CLUSTERSIZE];

      GPTLstart("ILP_neigh and normal");
      for(ii = iis; ii < iie; ii++) {
        i = ilist[ii];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        itype = type[i];
        itypes[ii - iis] = itype;
        itype_map = map[type[i]];
        int *jlist_intra = first_layered_neigh[i];
        int jnum_intra = num_intra[i];
        itag = tag[i];
        for (jj = 0; jj < jnum_intra; jj++) {
          j = jlist_intra[jj];

          jtype = map[type[j]];
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx * delx + dely * dely + delz * delz;

          if (rsq != 0 && rsq < cutILPsq[itype_map][jtype]) {
            if ((VARIANT == ILP_TMD || VARIANT == AIP_WATER_2DM) && special_type[itype] == TMD_METAL && itype != type[j]) continue;
            if (ILP_nneigh[ii - iis] >= MAX_NNEIGH) {
              printf("atom %d has %d neighbors, exceeding MAX_NNEIGH=%d\n", i, ILP_nneigh[ii - iis], MAX_NNEIGH);
              error->one(FLERR, "There are too many neighbors for calculating normals");
            }

            ILP_neigh[ii - iis][ILP_nneigh[ii - iis]++] = j;
          }
        }    // loop over jj

        calc_atom_normal<MAX_NNEIGH>(i, itypes[ii - iis], ILP_neigh[ii - iis], ILP_nneigh[ii - iis],
                                    normx[ii - iis], normy[ii - iis], normz[ii - iis],
                                    dnormdxi0[ii - iis], dnormdxi1[ii - iis], dnormdxi2[ii - iis],
                                    dnormdxi3[ii - iis], dnormdxi4[ii - iis], dnormdxi5[ii - iis],
                                    dnormdxi6[ii - iis], dnormdxi7[ii - iis], dnormdxi8[ii - iis],
                                    dnormdxk[ii - iis]);
      }
      GPTLstop("ILP_neigh and normal");

      GPTLstart("FvdW and some of FRep");

      svp_t predi32 = svwhilelt_b32(0, iicnt);
      svp_t predi64 = svwhilelt(0, iicnt);

      MY_VEC vnormx = svld1(predi64, normx);
      MY_VEC vnormy = svld1(predi64, normy);
      MY_VEC vnormz = svld1(predi64, normz);
      MY_VEC vdnormdxi[9] = {
          svld1(predi64, dnormdxi0), svld1(predi64, dnormdxi1), svld1(predi64, dnormdxi2),
          svld1(predi64, dnormdxi3), svld1(predi64, dnormdxi4), svld1(predi64, dnormdxi5),
          svld1(predi64, dnormdxi6), svld1(predi64, dnormdxi7), svld1(predi64, dnormdxi8)};
      MY_VEC vdproddnix = zero;
      MY_VEC vdproddniy = zero;
      MY_VEC vdproddniz = zero;

      svsi_t is32 = svld1(predi32, ilist + iis);
      svsl_t is64 = svunpklo(is32);
      MY_VEC xi, yi, zi, qi;
      lmff_load_cluster_xq(predi32, predi64, is32, is64, iicnt, x, q, xi, yi, zi, qi);

      int *jlist_inter = first_bundled_neigh[ic];
      int jnum_inter = num_inter[ic];

      svsi_t atomMoleculeI = svld1_gather_index(predi32, atom->molecule, is32);
      MY_FLOAT delta2inv_[ntypes + 1][iicnt];
      MY_FLOAT lambda_[ntypes + 1][iicnt];
      MY_FLOAT C_[ntypes + 1][iicnt];
      MY_FLOAT epsilon_[ntypes + 1][iicnt];
      MY_FLOAT d_[ntypes + 1][iicnt];
      MY_FLOAT C6_[ntypes + 1][iicnt];
      MY_FLOAT z0_[ntypes + 1][iicnt];
      MY_FLOAT seff_[ntypes + 1][iicnt];
      MY_FLOAT cutsq_[ntypes + 1][iicnt];
      MY_FLOAT sigmae_[ntypes + 1][iicnt];
      MY_FLOAT coul_active_[ntypes + 1][iicnt];
      MY_FLOAT Rcut_[ntypes + 1][iicnt];

      MY_VEC vdelta2inv[ntypes + 1];
      MY_VEC vlambda[ntypes + 1];
      MY_VEC vC[ntypes + 1];
      MY_VEC vepsilon[ntypes + 1];
      MY_VEC vd[ntypes + 1];
      MY_VEC vC6[ntypes + 1];
      MY_VEC vseff[ntypes + 1];
      MY_VEC vz0[ntypes + 1];
      MY_VEC vcutsq[ntypes + 1];
      MY_VEC vsigmae[ntypes + 1];
      MY_VEC vcoul_active[ntypes + 1];
      MY_VEC vRcut[ntypes + 1];
      MY_VEC vinv_vsigma[ntypes + 1];

      for(int jtemp = 1; jtemp <= ntypes; jtemp++) {
        for(int itemp = 0; itemp < iicnt; itemp++) {
          int iparam_ij = elem2param[map[itypes[itemp]]][map[jtemp]];
          Param &p = params[iparam_ij];
          delta2inv_[jtemp][itemp] = p.delta2inv;
          lambda_[jtemp][itemp] = p.lambda;
          C_[jtemp][itemp] = p.C;
          epsilon_[jtemp][itemp] = p.epsilon;
          d_[jtemp][itemp] = p.d;
          C6_[jtemp][itemp] = p.C6;
          z0_[jtemp][itemp] = p.z0;
          seff_[jtemp][itemp] = 1.0 / p.seff;
          cutsq_[jtemp][itemp] = cutsq[itypes[itemp]][jtemp];
          sigmae_[jtemp][itemp] = sigmae_map[itypes[itemp]][jtemp];
          coul_active_[jtemp][itemp] = coul_setflag[itypes[itemp]][jtemp] ? 1.0 : 0.0;
          Rcut_[jtemp][itemp] = sqrt(cutsq_[jtemp][itemp]);
        }
        vdelta2inv[jtemp] = svld1(predi64, &delta2inv_[jtemp][0]);
        vlambda[jtemp] = svld1(predi64, &lambda_[jtemp][0]);
        vC[jtemp] = svld1(predi64, &C_[jtemp][0]);
        vepsilon[jtemp] = svld1(predi64, &epsilon_[jtemp][0]);
        vd[jtemp] = svld1(predi64, &d_[jtemp][0]);
        vC6[jtemp] = svld1(predi64, &C6_[jtemp][0]);
        vseff[jtemp] = svld1(predi64, &seff_[jtemp][0]);
        vz0[jtemp] = svld1(predi64, &z0_[jtemp][0]);
        vcutsq[jtemp] = svld1(predi64, &cutsq_[jtemp][0]);
        vsigmae[jtemp] = svld1(predi64, &sigmae_[jtemp][0]);
        vcoul_active[jtemp] = svld1(predi64, &coul_active_[jtemp][0]);
        svp_t has_sigmae = svcmpgt(predi64, vsigmae[jtemp], zero);
        vinv_vsigma[jtemp] = svdiv_x(has_sigmae, one, vsigmae[jtemp]);
        vRcut[jtemp] = svld1(predi64, &Rcut_[jtemp][0]);
      }


      MY_VEC fxi = zero, fyi = zero, fzi = zero;
      MY_TAGINT itag0 = lmff_load_itag0(predi32, predi64, tag, is32, is64);

      int valid_j[4];
      int valid_len = 0;
      MY_VEC delx0_[4], dely0_[4], delz0_[4];
      MY_VEC rsq0_[4];
      svp_t incut0_[4];

      for(jj = 0; jj < jnum_inter; jj++) {
        j = jlist_inter[jj];
#ifdef LMFF_CLUSTER_STATS
        stat_compute_slots += static_cast<unsigned long long>(iicnt);
#endif
        svp_t pred_mask = lmff_pred_same_molecule(predi32, predi64, atomMoleculeI, atom->molecule[j]);
        if (!svptest_any(svptrue(), pred_mask)) continue;

        jtype = type[j];
        MY_VEC xj0 = svdup(x[j][0]);
        MY_VEC yj0 = svdup(x[j][1]);
        MY_VEC zj0 = svdup(x[j][2]);

        MY_VEC delx0 = xi - xj0;
        MY_VEC dely0 = yi - yj0;
        MY_VEC delz0 = zi - zj0;

        MY_VEC rsq0 = delx0 * delx0 + dely0 * dely0 + delz0 * delz0;
        MY_VEC cutsq0 = vcutsq[jtype];
        svp_t incut0 = svcmplt(pred_mask, rsq0, cutsq0);

#ifdef LMFF_CLUSTER_STATS
        stat_active_lanes += lmff_count_active_lanes(incut0);
#endif
        if (!svptest_any(svptrue(), incut0)) continue;

        if(valid_len == 3) {
          delx0_[valid_len] = delx0;
          dely0_[valid_len] = dely0;
          delz0_[valid_len] = delz0;
          rsq0_[valid_len] = rsq0;
          incut0_[valid_len] = incut0;
          valid_j[valid_len++] = jj;
          valid_len = 0;
          int j0 = jlist_inter[valid_j[0]];
          int j1 = jlist_inter[valid_j[1]];
          int j2 = jlist_inter[valid_j[2]];
          int j3 = jlist_inter[valid_j[3]];

          MY_VEC xj0 = svdup(x[j0][0]);
          MY_VEC yj0 = svdup(x[j0][1]);
          MY_VEC zj0 = svdup(x[j0][2]);
          MY_VEC xj1 = svdup(x[j1][0]);
          MY_VEC yj1 = svdup(x[j1][1]);
          MY_VEC zj1 = svdup(x[j1][2]);
          MY_VEC xj2 = svdup(x[j2][0]);
          MY_VEC yj2 = svdup(x[j2][1]);
          MY_VEC zj2 = svdup(x[j2][2]);
          MY_VEC xj3 = svdup(x[j3][0]);
          MY_VEC yj3 = svdup(x[j3][1]);
          MY_VEC zj3 = svdup(x[j3][2]);

          MY_VEC delx0 = delx0_[0];
          MY_VEC dely0 = dely0_[0];
          MY_VEC delz0 = delz0_[0];
          MY_VEC delx1 = delx0_[1];
          MY_VEC dely1 = dely0_[1];
          MY_VEC delz1 = delz0_[1];
          MY_VEC delx2 = delx0_[2];
          MY_VEC dely2 = dely0_[2];
          MY_VEC delz2 = delz0_[2];
          MY_VEC delx3 = delx0_[3];
          MY_VEC dely3 = dely0_[3];
          MY_VEC delz3 = delz0_[3];

          MY_VEC rsq0 = rsq0_[0];
          MY_VEC rsq1 = rsq0_[1];
          MY_VEC rsq2 = rsq0_[2];
          MY_VEC rsq3 = rsq0_[3];

          svp_t incut0 = incut0_[0];
          svp_t incut1 = incut0_[1];
          svp_t incut2 = incut0_[2];
          svp_t incut3 = incut0_[3];

          int jtype0 = type[j0];
          int jtype1 = type[j1];
          int jtype2 = type[j2];
          int jtype3 = type[j3];

          MY_VEC r0 = svsqrt_x(incut0, rsq0);
          MY_VEC r1 = svsqrt_x(incut1, rsq1);
          MY_VEC r2 = svsqrt_x(incut2, rsq2);
          MY_VEC r3 = svsqrt_x(incut3, rsq3);

          MY_VEC r2inv0 = one / rsq0;
          MY_VEC r2inv1 = one / rsq1;
          MY_VEC r2inv2 = one / rsq2;
          MY_VEC r2inv3 = one / rsq3;

          MY_VEC rinv0  = r0 * r2inv0;
          MY_VEC rinv1  = r1 * r2inv1;
          MY_VEC rinv2  = r2 * r2inv2;
          MY_VEC rinv3  = r3 * r2inv3;

          MY_VEC Tap0;
          MY_VEC Tap1;
          MY_VEC Tap2;
          MY_VEC Tap3;

          MY_VEC dTap0;
          MY_VEC dTap1;
          MY_VEC dTap2;
          MY_VEC dTap3;

          if(TAP_FLAG) {
            MY_VEC Rcut0 = vRcut[jtype0];
            MY_VEC Rcut1 = vRcut[jtype1];
            MY_VEC Rcut2 = vRcut[jtype2];
            MY_VEC Rcut3 = vRcut[jtype3];

            Tap0 = calc_Tap_sve(incut0, r0, Rcut0);
            Tap1 = calc_Tap_sve(incut1, r1, Rcut1);
            Tap2 = calc_Tap_sve(incut2, r2, Rcut2);
            Tap3 = calc_Tap_sve(incut3, r3, Rcut3);

            dTap0 = calc_dTap_sve(incut0, r0, Rcut0);
            dTap1 = calc_dTap_sve(incut1, r1, Rcut1);
            dTap2 = calc_dTap_sve(incut2, r2, Rcut2);
            dTap3 = calc_dTap_sve(incut3, r3, Rcut3);
          } else {
            Tap0 = one;
            Tap1 = one;
            Tap2 = one;
            Tap3 = one;

            dTap0 = zero;
            dTap1 = zero;
            dTap2 = zero;
            dTap3 = zero;
          }

          MY_VEC prodnorm10 = vnormx * delx0 + vnormy * dely0 + vnormz * delz0;
          MY_VEC prodnorm11 = vnormx * delx1 + vnormy * dely1 + vnormz * delz1;
          MY_VEC prodnorm12 = vnormx * delx2 + vnormy * dely2 + vnormz * delz2;
          MY_VEC prodnorm13 = vnormx * delx3 + vnormy * dely3 + vnormz * delz3;

          MY_VEC rhosq10 = rsq0 - prodnorm10 * prodnorm10;
          MY_VEC rhosq11 = rsq1 - prodnorm11 * prodnorm11;
          MY_VEC rhosq12 = rsq2 - prodnorm12 * prodnorm12;
          MY_VEC rhosq13 = rsq3 - prodnorm13 * prodnorm13;

          MY_VEC rdsq10 = rhosq10 * vdelta2inv[jtype0];
          MY_VEC rdsq11 = rhosq11 * vdelta2inv[jtype1];
          MY_VEC rdsq12 = rhosq12 * vdelta2inv[jtype2];
          MY_VEC rdsq13 = rhosq13 * vdelta2inv[jtype3];


          MY_VEC r30 = rsq0 * r0;
          MY_VEC r31 = rsq1 * r1;
          MY_VEC r32 = rsq2 * r2;
          MY_VEC r33 = rsq3 * r3;

          MY_VEC th0 = r30 + vinv_vsigma[jtype0];
          MY_VEC th1 = r31 + vinv_vsigma[jtype1];
          MY_VEC th2 = r32 + vinv_vsigma[jtype2];
          MY_VEC th3 = r33 + vinv_vsigma[jtype3];

          MY_VEC vtmp0_ln[4] = {th0, th1, th2, th3};
          MY_VEC vtmp1_ln[4];
          svnxp_log<4, 4>(vtmp1_ln, vtmp0_ln);

          MY_VEC th0_ln = one_third * vtmp1_ln[0];
          MY_VEC th1_ln = one_third * vtmp1_ln[1];
          MY_VEC th2_ln = one_third * vtmp1_ln[2];
          MY_VEC th3_ln = one_third * vtmp1_ln[3];

          MY_VEC vtmp0_vector[8] = {-vlambda[jtype0] * (r0 - vz0[jtype0]), -rdsq10, th0_ln, -vd[jtype0] * (r0 * vseff[jtype0] - 1.0),
                                   -vlambda[jtype1] * (r1 - vz0[jtype1]), -rdsq11, th1_ln, -vd[jtype1] * (r1 * vseff[jtype1] - 1.0)};
          MY_VEC exp00_vector[8];
          svnxp_exp<8, 4>(exp00_vector, vtmp0_vector);
          MY_VEC vtmp1_vector[8] = {-vlambda[jtype2] * (r2 - vz0[jtype2]), -rdsq12, th2_ln, -vd[jtype2] * (r2 * vseff[jtype2] - 1.0),
                                   -vlambda[jtype3] * (r3 - vz0[jtype3]), -rdsq13, th3_ln, -vd[jtype3] * (r3 * vseff[jtype3] - 1.0)};
          MY_VEC exp01_vector[8];
          svnxp_exp<8, 4>(exp01_vector, vtmp1_vector);

          MY_VEC exp00 = exp00_vector[0];
          MY_VEC exp01 = exp00_vector[4];
          MY_VEC exp02 = exp01_vector[0];
          MY_VEC exp03 = exp01_vector[4];

          MY_VEC exp10 = exp00_vector[1];
          MY_VEC exp11 = exp00_vector[5];
          MY_VEC exp12 = exp01_vector[1];
          MY_VEC exp13 = exp01_vector[5];

          MY_VEC epsr0 = exp00_vector[2];
          MY_VEC epsr1 = exp00_vector[6];
          MY_VEC epsr2 = exp01_vector[2];
          MY_VEC epsr3 = exp01_vector[6];

          svp_t is_nan_mask = svcmpne(svptrue(), epsr0, epsr0);
          epsr0 = svsel(is_nan_mask, zero, epsr0);
          is_nan_mask = svcmpne(svptrue(), epsr1, epsr1);
          epsr1 = svsel(is_nan_mask, zero, epsr1);
          is_nan_mask = svcmpne(svptrue(), epsr2, epsr2);
          epsr2 = svsel(is_nan_mask, zero, epsr2);
          is_nan_mask = svcmpne(svptrue(), epsr3, epsr3);
          epsr3 = svsel(is_nan_mask, zero, epsr3);

          MY_VEC TSvdw0 = one + exp00_vector[3];
          MY_VEC TSvdw1 = one + exp00_vector[7];
          MY_VEC TSvdw2 = one + exp01_vector[3];
          MY_VEC TSvdw3 = one + exp01_vector[7];

          MY_VEC frho10 = exp10 * vC[jtype0];
          MY_VEC frho11 = exp11 * vC[jtype1];
          MY_VEC frho12 = exp12 * vC[jtype2];
          MY_VEC frho13 = exp13 * vC[jtype3];

          MY_VEC Erep0 = vhalf * vepsilon[jtype0] + frho10;
          MY_VEC Erep1 = vhalf * vepsilon[jtype1] + frho11;
          MY_VEC Erep2 = vhalf * vepsilon[jtype2] + frho12;
          MY_VEC Erep3 = vhalf * vepsilon[jtype3] + frho13;

          MY_VEC Vilp0 = exp00 * Erep0;
          MY_VEC Vilp1 = exp01 * Erep1;
          MY_VEC Vilp2 = exp02 * Erep2;
          MY_VEC Vilp3 = exp03 * Erep3;

          MY_VEC fpair0 = vlambda[jtype0] * exp00 * rinv0 * Erep0;
          MY_VEC fpair1 = vlambda[jtype1] * exp01 * rinv1 * Erep1;
          MY_VEC fpair2 = vlambda[jtype2] * exp02 * rinv2 * Erep2;
          MY_VEC fpair3 = vlambda[jtype3] * exp03 * rinv3 * Erep3;

          MY_VEC fpair10 = two * exp00 * frho10 * vdelta2inv[jtype0];
          MY_VEC fpair11 = two * exp01 * frho11 * vdelta2inv[jtype1];
          MY_VEC fpair12 = two * exp02 * frho12 * vdelta2inv[jtype2];
          MY_VEC fpair13 = two * exp03 * frho13 * vdelta2inv[jtype3];

          MY_VEC fsum0 = fpair0 + fpair10;
          MY_VEC fsum1 = fpair1 + fpair11;
          MY_VEC fsum2 = fpair2 + fpair12;
          MY_VEC fsum3 = fpair3 + fpair13;

          MY_VEC fp1x0 = prodnorm10 * vnormx * fpair10;
          MY_VEC fp1y0 = prodnorm10 * vnormy * fpair10;
          MY_VEC fp1z0 = prodnorm10 * vnormz * fpair10;
          MY_VEC fp1x1 = prodnorm11 * vnormx * fpair11;
          MY_VEC fp1y1 = prodnorm11 * vnormy * fpair11;
          MY_VEC fp1z1 = prodnorm11 * vnormz * fpair11;
          MY_VEC fp1x2 = prodnorm12 * vnormx * fpair12;
          MY_VEC fp1y2 = prodnorm12 * vnormy * fpair12;
          MY_VEC fp1z2 = prodnorm12 * vnormz * fpair12;
          MY_VEC fp1x3 = prodnorm13 * vnormx * fpair13;
          MY_VEC fp1y3 = prodnorm13 * vnormy * fpair13;
          MY_VEC fp1z3 = prodnorm13 * vnormz * fpair13;


          MY_VEC fkcx0 = (delx0 * fsum0 - fp1x0) * Tap0 - Vilp0 * dTap0 * delx0 * rinv0;
          MY_VEC fkcy0 = (dely0 * fsum0 - fp1y0) * Tap0 - Vilp0 * dTap0 * dely0 * rinv0;
          MY_VEC fkcz0 = (delz0 * fsum0 - fp1z0) * Tap0 - Vilp0 * dTap0 * delz0 * rinv0;
          MY_VEC fkcx1 = (delx1 * fsum1 - fp1x1) * Tap1 - Vilp1 * dTap1 * delx1 * rinv1;
          MY_VEC fkcy1 = (dely1 * fsum1 - fp1y1) * Tap1 - Vilp1 * dTap1 * dely1 * rinv1;
          MY_VEC fkcz1 = (delz1 * fsum1 - fp1z1) * Tap1 - Vilp1 * dTap1 * delz1 * rinv1;
          MY_VEC fkcx2 = (delx2 * fsum2 - fp1x2) * Tap2 - Vilp2 * dTap2 * delx2 * rinv2;
          MY_VEC fkcy2 = (dely2 * fsum2 - fp1y2) * Tap2 - Vilp2 * dTap2 * dely2 * rinv2;
          MY_VEC fkcz2 = (delz2 * fsum2 - fp1z2) * Tap2 - Vilp2 * dTap2 * delz2 * rinv2;
          MY_VEC fkcx3 = (delx3 * fsum3 - fp1x3) * Tap3 - Vilp3 * dTap3 * delx3 * rinv3;
          MY_VEC fkcy3 = (dely3 * fsum3 - fp1y3) * Tap3 - Vilp3 * dTap3 * dely3 * rinv3;
          MY_VEC fkcz3 = (delz3 * fsum3 - fp1z3) * Tap3 - Vilp3 * dTap3 * delz3 * rinv3;

          fxi = svadd_m(incut0, fxi, fkcx0);
          fyi = svadd_m(incut0, fyi, fkcy0);
          fzi = svadd_m(incut0, fzi, fkcz0);
          fxi = svadd_m(incut1, fxi, fkcx1);
          fyi = svadd_m(incut1, fyi, fkcy1);
          fzi = svadd_m(incut1, fzi, fkcz1);
          fxi = svadd_m(incut2, fxi, fkcx2);
          fyi = svadd_m(incut2, fyi, fkcy2);
          fzi = svadd_m(incut2, fzi, fkcz2);
          fxi = svadd_m(incut3, fxi, fkcx3);
          fyi = svadd_m(incut3, fyi, fkcy3);
          fzi = svadd_m(incut3, fzi, fkcz3);

          f[j0][0] -= svaddv(incut0, fkcx0);
          f[j0][1] -= svaddv(incut0, fkcy0);
          f[j0][2] -= svaddv(incut0, fkcz0);
          f[j1][0] -= svaddv(incut1, fkcx1);
          f[j1][1] -= svaddv(incut1, fkcy1);
          f[j1][2] -= svaddv(incut1, fkcz1);
          f[j2][0] -= svaddv(incut2, fkcx2);
          f[j2][1] -= svaddv(incut2, fkcy2);
          f[j2][2] -= svaddv(incut2, fkcz2);
          f[j3][0] -= svaddv(incut3, fkcx3);
          f[j3][1] -= svaddv(incut3, fkcy3);
          f[j3][2] -= svaddv(incut3, fkcz3);

          MY_VEC cij0 = -prodnorm10 * fpair10 * Tap0;
          MY_VEC cij1 = -prodnorm11 * fpair11 * Tap1;
          MY_VEC cij2 = -prodnorm12 * fpair12 * Tap2;
          MY_VEC cij3 = -prodnorm13 * fpair13 * Tap3;

          vdproddnix += cij0 * delx0 + cij1 * delx1 + cij2 * delx2 + cij3 * delx3;
          vdproddniy += cij0 * dely0 + cij1 * dely1 + cij2 * dely2 + cij3 * dely3;
          vdproddniz += cij0 * delz0 + cij1 * delz1 + cij2 * delz2 + cij3 * delz3;

          MY_VEC evdwl0;
          MY_VEC evdwl1;
          MY_VEC evdwl2;
          MY_VEC evdwl3;

          if(EFLAG) {
            evdwl0 = Tap0 * Vilp0;
            evdwl1 = Tap1 * Vilp1;
            evdwl2 = Tap2 * Vilp2;
            evdwl3 = Tap3 * Vilp3;

            l_pvector[1] += svaddv(incut0, evdwl0) + svaddv(incut1, evdwl1) + svaddv(incut2, evdwl2) + svaddv(incut3, evdwl3);
          }
          if(EVFLAG) {
            l_eng_vdwl += svaddv(incut0, evdwl0) + svaddv(incut1, evdwl1) + svaddv(incut2, evdwl2) + svaddv(incut3, evdwl3);
          }

          MY_TAGINT jtag0 = lmff_dup_tag_j(tag[j0]);
          MY_TAGINT jtag1 = lmff_dup_tag_j(tag[j1]);
          MY_TAGINT jtag2 = lmff_dup_tag_j(tag[j2]);
          MY_TAGINT jtag3 = lmff_dup_tag_j(tag[j3]);

          svp_t checkVdw0 = lmff_check_vdw(incut0, itag0, jtag0, xi, yi, zi, xj0, yj0, zj0);
          svp_t checkVdw1 = lmff_check_vdw(incut1, itag0, jtag1, xi, yi, zi, xj1, yj1, zj1);
          svp_t checkVdw2 = lmff_check_vdw(incut2, itag0, jtag2, xi, yi, zi, xj2, yj2, zj2);
          svp_t checkVdw3 = lmff_check_vdw(incut3, itag0, jtag3, xi, yi, zi, xj3, yj3, zj3);

          MY_VEC factor_coul0 = svdup(special_coul[sbmask(j0)]);
          MY_VEC factor_coul1 = svdup(special_coul[sbmask(j1)]);
          MY_VEC factor_coul2 = svdup(special_coul[sbmask(j2)]);
          MY_VEC factor_coul3 = svdup(special_coul[sbmask(j3)]);

          MY_VEC depsdr0 = epsr0 * epsr0;
          MY_VEC depsdr1 = epsr1 * epsr1;
          MY_VEC depsdr2 = epsr2 * epsr2;
          MY_VEC depsdr3 = epsr3 * epsr3;

          depsdr0 = depsdr0 * depsdr0;
          depsdr1 = depsdr1 * depsdr1;
          depsdr2 = depsdr2 * depsdr2;
          depsdr3 = depsdr3 * depsdr3;

          MY_VEC qj0 = svdup(q[j0]);
          MY_VEC qj1 = svdup(q[j1]);
          MY_VEC qj2 = svdup(q[j2]);
          MY_VEC qj3 = svdup(q[j3]);

          MY_VEC Vc0 = qqrd2e0 * qi * qj0 * epsr0 * vcoul_active[jtype0];
          MY_VEC Vc1 = qqrd2e0 * qi * qj1 * epsr1 * vcoul_active[jtype1];
          MY_VEC Vc2 = qqrd2e0 * qi * qj2 * epsr2 * vcoul_active[jtype2];
          MY_VEC Vc3 = qqrd2e0 * qi * qj3 * epsr3 * vcoul_active[jtype3];

          MY_VEC r6inv0 = r2inv0 * r2inv0 * r2inv0;
          MY_VEC r6inv1 = r2inv1 * r2inv1 * r2inv1;
          MY_VEC r6inv2 = r2inv2 * r2inv2 * r2inv2;
          MY_VEC r6inv3 = r2inv3 * r2inv3 * r2inv3;

          MY_VEC r8inv0 = r6inv0 * r2inv0;
          MY_VEC r8inv1 = r6inv1 * r2inv1;
          MY_VEC r8inv2 = r6inv2 * r2inv2;
          MY_VEC r8inv3 = r6inv3 * r2inv3;

          MY_VEC TSvdwinv0 = one / TSvdw0;
          MY_VEC TSvdwinv1 = one / TSvdw1;
          MY_VEC TSvdwinv2 = one / TSvdw2;
          MY_VEC TSvdwinv3 = one / TSvdw3;

          MY_VEC TSvdw2inv0 = TSvdwinv0 * TSvdwinv0;
          MY_VEC TSvdw2inv1 = TSvdwinv1 * TSvdwinv1;
          MY_VEC TSvdw2inv2 = TSvdwinv2 * TSvdwinv2;
          MY_VEC TSvdw2inv3 = TSvdwinv3 * TSvdwinv3;

          Vilp0 = -vC6[jtype0] * r6inv0 * TSvdwinv0;
          Vilp1 = -vC6[jtype1] * r6inv1 * TSvdwinv1;
          Vilp2 = -vC6[jtype2] * r6inv2 * TSvdwinv2;
          Vilp3 = -vC6[jtype3] * r6inv3 * TSvdwinv3;

          fpair0 = -six * vC6[jtype0] * r8inv0 * TSvdwinv0 + vC6[jtype0] * vd[jtype0] * vseff[jtype0] * (TSvdw0 - one) * TSvdw2inv0 * r8inv0 * r0;
          fpair1 = -six * vC6[jtype1] * r8inv1 * TSvdwinv1 + vC6[jtype1] * vd[jtype1] * vseff[jtype1] * (TSvdw1 - one) * TSvdw2inv1 * r8inv1 * r1;
          fpair2 = -six * vC6[jtype2] * r8inv2 * TSvdwinv2 + vC6[jtype2] * vd[jtype2] * vseff[jtype2] * (TSvdw2 - one) * TSvdw2inv2 * r8inv2 * r2;
          fpair3 = -six * vC6[jtype3] * r8inv3 * TSvdwinv3 + vC6[jtype3] * vd[jtype3] * vseff[jtype3] * (TSvdw3 - one) * TSvdw2inv3 * r8inv3 * r3;

          fsum0 = fpair0 * Tap0 - Vilp0 * dTap0 * rinv0;
          fsum1 = fpair1 * Tap1 - Vilp1 * dTap1 * rinv1;
          fsum2 = fpair2 * Tap2 - Vilp2 * dTap2 * rinv2;
          fsum3 = fpair3 * Tap3 - Vilp3 * dTap3 * rinv3;

          MY_VEC fvdwx0 = fsum0 * delx0;
          MY_VEC fvdwy0 = fsum0 * dely0;
          MY_VEC fvdwz0 = fsum0 * delz0;
          MY_VEC fvdwx1 = fsum1 * delx1;
          MY_VEC fvdwy1 = fsum1 * dely1;
          MY_VEC fvdwz1 = fsum1 * delz1;
          MY_VEC fvdwx2 = fsum2 * delx2;
          MY_VEC fvdwy2 = fsum2 * dely2;
          MY_VEC fvdwz2 = fsum2 * delz2;
          MY_VEC fvdwx3 = fsum3 * delx3;
          MY_VEC fvdwy3 = fsum3 * dely3;
          MY_VEC fvdwz3 = fsum3 * delz3;

          MY_VEC forcecoul0 = qqrd2e0 * qi * qj0 * r0 * depsdr0 * vcoul_active[jtype0];
          MY_VEC forcecoul1 = qqrd2e0 * qi * qj1 * r1 * depsdr1 * vcoul_active[jtype1];
          MY_VEC forcecoul2 = qqrd2e0 * qi * qj2 * r2 * depsdr2 * vcoul_active[jtype2];
          MY_VEC forcecoul3 = qqrd2e0 * qi * qj3 * r3 * depsdr3 * vcoul_active[jtype3];

          MY_VEC fvc0 = forcecoul0 * Tap0 - Vc0 * dTap0 / r0;
          MY_VEC fvc1 = forcecoul1 * Tap1 - Vc1 * dTap1 / r1;
          MY_VEC fvc2 = forcecoul2 * Tap2 - Vc2 * dTap2 / r2;
          MY_VEC fvc3 = forcecoul3 * Tap3 - Vc3 * dTap3 / r3;

          fxi = svadd_m(checkVdw0, fxi, fvdwx0 + delx0 * factor_coul0 * fvc0);
          fyi = svadd_m(checkVdw0, fyi, fvdwy0 + dely0 * factor_coul0 * fvc0);
          fzi = svadd_m(checkVdw0, fzi, fvdwz0 + delz0 * factor_coul0 * fvc0);
          fxi = svadd_m(checkVdw1, fxi, fvdwx1 + delx1 * factor_coul1 * fvc1);
          fyi = svadd_m(checkVdw1, fyi, fvdwy1 + dely1 * factor_coul1 * fvc1);
          fzi = svadd_m(checkVdw1, fzi, fvdwz1 + delz1 * factor_coul1 * fvc1);
          fxi = svadd_m(checkVdw2, fxi, fvdwx2 + delx2 * factor_coul2 * fvc2);
          fyi = svadd_m(checkVdw2, fyi, fvdwy2 + dely2 * factor_coul2 * fvc2);
          fzi = svadd_m(checkVdw2, fzi, fvdwz2 + delz2 * factor_coul2 * fvc2);
          fxi = svadd_m(checkVdw3, fxi, fvdwx3 + delx3 * factor_coul3 * fvc3);
          fyi = svadd_m(checkVdw3, fyi, fvdwy3 + dely3 * factor_coul3 * fvc3);
          fzi = svadd_m(checkVdw3, fzi, fvdwz3 + delz3 * factor_coul3 * fvc3);

          f[j0][0] -= svaddv(checkVdw0, fvdwx0 + delx0 * factor_coul0 * fvc0);
          f[j0][1] -= svaddv(checkVdw0, fvdwy0 + dely0 * factor_coul0 * fvc0);
          f[j0][2] -= svaddv(checkVdw0, fvdwz0 + delz0 * factor_coul0 * fvc0);
          f[j1][0] -= svaddv(checkVdw1, fvdwx1 + delx1 * factor_coul1 * fvc1);
          f[j1][1] -= svaddv(checkVdw1, fvdwy1 + dely1 * factor_coul1 * fvc1);
          f[j1][2] -= svaddv(checkVdw1, fvdwz1 + delz1 * factor_coul1 * fvc1);
          f[j2][0] -= svaddv(checkVdw2, fvdwx2 + delx2 * factor_coul2 * fvc2);
          f[j2][1] -= svaddv(checkVdw2, fvdwy2 + dely2 * factor_coul2 * fvc2);
          f[j2][2] -= svaddv(checkVdw2, fvdwz2 + delz2 * factor_coul2 * fvc2);
          f[j3][0] -= svaddv(checkVdw3, fvdwx3 + delx3 * factor_coul3 * fvc3);
          f[j3][1] -= svaddv(checkVdw3, fvdwy3 + dely3 * factor_coul3 * fvc3);
          f[j3][2] -= svaddv(checkVdw3, fvdwz3 + delz3 * factor_coul3 * fvc3);

          MY_VEC ecoul0;
          MY_VEC ecoul1;
          MY_VEC ecoul2;
          MY_VEC ecoul3;

          if(EFLAG) {
            evdwl0 = Tap0 * Vilp0;
            evdwl1 = Tap1 * Vilp1;
            evdwl2 = Tap2 * Vilp2;
            evdwl3 = Tap3 * Vilp3;

            ecoul0 = Vc0 * Tap0 * factor_coul0;
            ecoul1 = Vc1 * Tap1 * factor_coul1;
            ecoul2 = Vc2 * Tap2 * factor_coul2;
            ecoul3 = Vc3 * Tap3 * factor_coul3;

            l_pvector[0] += svaddv(checkVdw0, evdwl0) + svaddv(checkVdw1, evdwl1) + svaddv(checkVdw2, evdwl2) + svaddv(checkVdw3, evdwl3);

            l_pvector[3] += svaddv(checkVdw0, ecoul0) + svaddv(checkVdw1, ecoul1) + svaddv(checkVdw2, ecoul2) + svaddv(checkVdw3, ecoul3);
          }
          if(EVFLAG) {
            l_eng_vdwl += svaddv(checkVdw0, evdwl0) + svaddv(checkVdw1, evdwl1) + svaddv(checkVdw2, evdwl2) + svaddv(checkVdw3, evdwl3);

            l_eng_coul += svaddv(checkVdw0, ecoul0) + svaddv(checkVdw1, ecoul1) + svaddv(checkVdw2, ecoul2) + svaddv(checkVdw3, ecoul3);
          }
        } else {
          delx0_[valid_len] = delx0;
          dely0_[valid_len] = dely0;
          delz0_[valid_len] = delz0;
          rsq0_[valid_len] = rsq0;
          incut0_[valid_len] = incut0;
          valid_j[valid_len++] = jj;
        }
      }

      for(jj = 0; jj < valid_len; jj++) {
        j = jlist_inter[valid_j[jj]];
        MY_VEC xj0 = svdup(x[j][0]);
        MY_VEC yj0 = svdup(x[j][1]);
        MY_VEC zj0 = svdup(x[j][2]);


        MY_VEC delx0 = delx0_[jj];
        MY_VEC dely0 = dely0_[jj];
        MY_VEC delz0 = delz0_[jj];
        MY_VEC rsq0 = rsq0_[jj];
        svp_t incut0 = incut0_[jj];
        jtype = type[j];


        MY_VEC r0 = svsqrt_x(incut0, rsq0);

        MY_VEC r2inv0 = one / rsq0;
        MY_VEC rinv0  = r0 * r2inv0;

        MY_VEC Tap0;
        MY_VEC dTap0;

        if(TAP_FLAG) {
          MY_VEC Rcut0 = vRcut[jtype];
          Tap0 = calc_Tap_sve(incut0, r0, Rcut0);
          dTap0 = calc_dTap_sve(incut0, r0, Rcut0);
        } else {
          Tap0 = one;
          dTap0 = zero;
        }

        MY_VEC prodnorm10 = vnormx * delx0 + vnormy * dely0 + vnormz * delz0;
        MY_VEC rhosq10 = rsq0 - prodnorm10 * prodnorm10;
        MY_VEC rdsq10 = rhosq10 * vdelta2inv[jtype];


        MY_VEC r30 = rsq0 * r0;
        MY_VEC th0 = r30 + vinv_vsigma[jtype];

        MY_VEC vtmp0_ln[1] = {th0};
        MY_VEC vtmp1_ln[1];
        svnxp_log<1, 4>(vtmp1_ln, vtmp0_ln);
        MY_VEC th0_ln = one_third * vtmp1_ln[0];

        MY_VEC vtmp0_vector[4] = {-vlambda[jtype] * (r0 - vz0[jtype]), -rdsq10, th0_ln, -vd[jtype] * (r0 * vseff[jtype] - 1.0)};
        MY_VEC exp00_vector[4];
        svnxp_exp<4, 4>(exp00_vector, vtmp0_vector);
        MY_VEC exp00 = exp00_vector[0];
        MY_VEC exp10 = exp00_vector[1];
        MY_VEC epsr0 = exp00_vector[2];
        MY_VEC TSvdw0 = one + exp00_vector[3];

        svp_t is_nan_mask = svcmpne(svptrue(), epsr0, epsr0);
        epsr0 = svsel(is_nan_mask, zero, epsr0);


        MY_VEC frho10 = exp10 * vC[jtype];
        MY_VEC Erep0 = vhalf * vepsilon[jtype] + frho10;

        MY_VEC Vilp0 = exp00 * Erep0;

        MY_VEC fpair0 = vlambda[jtype] * exp00 * rinv0 * Erep0;
        MY_VEC fpair10 = two * exp00 * frho10 * vdelta2inv[jtype];
        MY_VEC fsum0 = fpair0 + fpair10;

        MY_VEC fp1x0 = prodnorm10 * vnormx * fpair10;
        MY_VEC fp1y0 = prodnorm10 * vnormy * fpair10;
        MY_VEC fp1z0 = prodnorm10 * vnormz * fpair10;

        MY_VEC fkcx0 = (delx0 * fsum0 - fp1x0) * Tap0 - Vilp0 * dTap0 * delx0 * rinv0;
        MY_VEC fkcy0 = (dely0 * fsum0 - fp1y0) * Tap0 - Vilp0 * dTap0 * dely0 * rinv0;
        MY_VEC fkcz0 = (delz0 * fsum0 - fp1z0) * Tap0 - Vilp0 * dTap0 * delz0 * rinv0;

        fxi = svadd_m(incut0, fxi, fkcx0);
        fyi = svadd_m(incut0, fyi, fkcy0);
        fzi = svadd_m(incut0, fzi, fkcz0);
        f[j][0] -= svaddv(incut0, fkcx0);
        f[j][1] -= svaddv(incut0, fkcy0);
        f[j][2] -= svaddv(incut0, fkcz0);

        MY_VEC cij0 = -prodnorm10 * fpair10 * Tap0;

        vdproddnix += cij0 * delx0;
        vdproddniy += cij0 * dely0;
        vdproddniz += cij0 * delz0;

        MY_VEC evdwl0;

        if(EFLAG) {
          evdwl0 = Tap0 * Vilp0;
          l_pvector[1] += svaddv(incut0, evdwl0);
        }
        if(EVFLAG) {
          l_eng_vdwl += svaddv(incut0, evdwl0);
        }

        MY_TAGINT jtag0 = lmff_dup_tag_j(tag[j]);

        svp_t checkVdw0 = lmff_check_vdw(incut0, itag0, jtag0, xi, yi, zi, xj0, yj0, zj0);

        if (!svptest_any(svptrue(), checkVdw0)) continue;

        MY_VEC factor_coul0 = svdup(special_coul[sbmask(j)]);

        MY_VEC depsdr0 = epsr0 * epsr0;
        depsdr0 = depsdr0 * depsdr0;
        MY_VEC qj0 = svdup(q[j]);
        MY_VEC Vc0 = qqrd2e0 * qi * qj0 * epsr0 * vcoul_active[jtype];

        MY_VEC r6inv0 = r2inv0 * r2inv0 * r2inv0;
        MY_VEC r8inv0 = r6inv0 * r2inv0;

        MY_VEC TSvdwinv0 = one / TSvdw0;
        MY_VEC TSvdw2inv0 = TSvdwinv0 * TSvdwinv0;

        Vilp0 = -vC6[jtype] * r6inv0 * TSvdwinv0;

        fpair0 = -six * vC6[jtype] * r8inv0 * TSvdwinv0 + vC6[jtype] * vd[jtype] * vseff[jtype] * (TSvdw0 - one) * TSvdw2inv0 * r8inv0 * r0;
        fsum0 = fpair0 * Tap0 - Vilp0 * dTap0 * rinv0;

        MY_VEC fvdwx0 = fsum0 * delx0;
        MY_VEC fvdwy0 = fsum0 * dely0;
        MY_VEC fvdwz0 = fsum0 * delz0;

        MY_VEC forcecoul0 = qqrd2e0 * qi * qj0 * r0 * depsdr0 * vcoul_active[jtype];
        MY_VEC fvc0 = forcecoul0 * Tap0 - Vc0 * dTap0 / r0;

        fxi = svadd_m(checkVdw0, fxi, fvdwx0 + delx0 * factor_coul0 * fvc0);
        fyi = svadd_m(checkVdw0, fyi, fvdwy0 + dely0 * factor_coul0 * fvc0);
        fzi = svadd_m(checkVdw0, fzi, fvdwz0 + delz0 * factor_coul0 * fvc0);
        f[j][0] -= svaddv(checkVdw0, fvdwx0 + delx0 * factor_coul0 * fvc0);
        f[j][1] -= svaddv(checkVdw0, fvdwy0 + dely0 * factor_coul0 * fvc0);
        f[j][2] -= svaddv(checkVdw0, fvdwz0 + delz0 * factor_coul0 * fvc0);

        MY_VEC ecoul0;

        if(EFLAG) {
          evdwl0 = Tap0 * Vilp0;
          ecoul0 = Vc0 * Tap0 * factor_coul0;
          l_pvector[0] += svaddv(checkVdw0, evdwl0);
          l_pvector[3] += svaddv(checkVdw0, ecoul0);
        }
        if(EVFLAG) {
          l_eng_vdwl += svaddv(checkVdw0, evdwl0);
          l_eng_coul += svaddv(checkVdw0, ecoul0);
        }

      }
      fxi += vdnormdxi[0] * vdproddnix + vdnormdxi[3] * vdproddniy + vdnormdxi[6] * vdproddniz;
      fyi += vdnormdxi[1] * vdproddnix + vdnormdxi[4] * vdproddniy + vdnormdxi[7] * vdproddniz;
      fzi += vdnormdxi[2] * vdproddnix + vdnormdxi[5] * vdproddniy + vdnormdxi[8] * vdproddniz;
      lmff_scatter_i_forces(predi64, is32, is64, iicnt, fxi, fyi, fzi, zero, f);

      MY_FLOAT dproddnix[iicnt], dproddniy[iicnt], dproddniz[iicnt];
      svst1(predi64, dproddnix, vdproddnix);
      svst1(predi64, dproddniy, vdproddniy);
      svst1(predi64, dproddniz, vdproddniz);

      GPTLstop("FvdW and some of FRep");

      GPTLstart("some of FRep");

      MY_FLOAT rsq2_[3];
      MY_FLOAT delr2_[3][3];
      MY_FLOAT r2_hat_[3][3];

      for(ii = iis; ii < iie; ii++) {
        i = ilist[ii];
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        itype = type[i];
        itype_map = map[type[i]];

        for (kk = 0; kk < ILP_nneigh[ii - iis]; kk++) {
          k = ILP_neigh[ii - iis][kk];
          if (k == i) continue;
          // derivatives of the product of rij and ni respect to rk, k=0,1,2, where atom k is the neighbors of atom i
          f[k][0] += dnormdxk[ii - iis][kk][0] * dproddnix[ii - iis] + dnormdxk[ii - iis][kk][3] * dproddniy[ii - iis] + dnormdxk[ii - iis][kk][6] * dproddniz[ii - iis];
          f[k][1] += dnormdxk[ii - iis][kk][1] * dproddnix[ii - iis] + dnormdxk[ii - iis][kk][4] * dproddniy[ii - iis] + dnormdxk[ii - iis][kk][7] * dproddniz[ii - iis];
          f[k][2] += dnormdxk[ii - iis][kk][2] * dproddnix[ii - iis] + dnormdxk[ii - iis][kk][5] * dproddniy[ii - iis] + dnormdxk[ii - iis][kk][8] * dproddniz[ii - iis];

        }
      }
      GPTLstop("some of FRep");
    }    // loop over clusters
    eng_rep[tid * LMFF_ENG_STRIDE + 0] = l_eng_vdwl;
    eng_rep[tid * LMFF_ENG_STRIDE + 1] = l_eng_coul;
    eng_rep[tid * LMFF_ENG_STRIDE + 2] = l_pvector[0];
    eng_rep[tid * LMFF_ENG_STRIDE + 3] = l_pvector[1];
    eng_rep[tid * LMFF_ENG_STRIDE + 4] = l_pvector[2];
    eng_rep[tid * LMFF_ENG_STRIDE + 5] = l_pvector[3];
  }
  GPTLstop("omp parallel");

#ifdef LMFF_CLUSTER_STATS
  unsigned long long local_compute_stats[2] = {
      stat_compute_slots, stat_active_lanes};
  unsigned long long global_compute_stats[2] = {0, 0};
  MPI_Allreduce(local_compute_stats, global_compute_stats, 2,
                MPI_UNSIGNED_LONG_LONG, MPI_SUM, world);
  if (comm->me == 0 && neighbor->ago == 0) {
    const double u_active =
        global_compute_stats[0]
        ? static_cast<double>(global_compute_stats[1]) /
              static_cast<double>(global_compute_stats[0])
        : 0.0;
    if (screen)
      fprintf(screen,
              "LMFF_CLUSTER_COMPUTE slots=%llu active=%llu "
              "U_active=%.6f\n",
              global_compute_stats[0], global_compute_stats[1], u_active);
    if (logfile)
      fprintf(logfile,
              "LMFF_CLUSTER_COMPUTE slots=%llu active=%llu "
              "U_active=%.6f\n",
              global_compute_stats[0], global_compute_stats[1], u_active);
  }
#endif

  GPTLstart("reduce data");
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nall; i++) {
    for (int j = 0; j < num_omp_threads; j++) {
      force[i][0] += frep[nall_upalign * j + i][0];
      force[i][1] += frep[nall_upalign * j + i][1];
      force[i][2] += frep[nall_upalign * j + i][2];
    }
  }
  for (int t = 0; t < num_omp_threads; t++) {
    eng_vdwl += eng_rep[t * LMFF_ENG_STRIDE + 0];
    eng_coul += eng_rep[t * LMFF_ENG_STRIDE + 1];
    pvector[0] += eng_rep[t * LMFF_ENG_STRIDE + 2];
    pvector[1] += eng_rep[t * LMFF_ENG_STRIDE + 3];
    pvector[2] += eng_rep[t * LMFF_ENG_STRIDE + 4];
    pvector[3] += eng_rep[t * LMFF_ENG_STRIDE + 5];
  }
  GPTLstop("reduce data");
}

/* ----------------------------------------------------------------------
   Calculate the normals for one atom
------------------------------------------------------------------------- */
inline void deriv_normal(MY_FLOAT dndr[9], MY_FLOAT *del, MY_FLOAT nx, MY_FLOAT ny, MY_FLOAT nz, MY_FLOAT rnnorm)
{
  dndr[0] = (del[2] * nx * ny - del[1] * nx * nz) * rnnorm;
  dndr[3] = (-del[2] * (nx * nx + nz * nz) - del[1] * ny * nz) * rnnorm;
  dndr[6] = (del[2] * ny * nz + del[1] * (nx * nx + ny * ny)) * rnnorm;

  dndr[1] = (del[2] * (ny * ny + nz * nz) + del[0] * nx * nz) * rnnorm;
  dndr[4] = (-del[2] * nx * ny + del[0] * ny * nz) * rnnorm;
  dndr[7] = (-del[2] * nx * nz - del[0] * (nx * nx + ny * ny)) * rnnorm;

  dndr[2] = (-del[1] * (ny * ny + nz * nz) - del[0] * nx * ny) * rnnorm;
  dndr[5] = (del[1] * nx * ny + del[0] * (nx * nx + nz * nz)) * rnnorm;
  dndr[8] = (del[1] * nx * nz - del[0] * ny * nz) * rnnorm;

}
inline void deriv_hat(MY_FLOAT dnhatdn[3][3], MY_FLOAT *n, MY_FLOAT rnnorm, MY_FLOAT factor){
  MY_FLOAT cfactor = rnnorm * factor;
  dnhatdn[0][0] = (n[1]*n[1]+n[2]*n[2])*cfactor;
  dnhatdn[1][0] = -n[1]*n[0]*cfactor;
  dnhatdn[2][0] = -n[2]*n[0]*cfactor;
  dnhatdn[0][1] = -n[0]*n[1]*cfactor;
  dnhatdn[1][1] = (n[0]*n[0]+n[2]*n[2])*cfactor;
  dnhatdn[2][1] = -n[2]*n[1]*cfactor;
  dnhatdn[0][2] = -n[0]*n[2]*cfactor;
  dnhatdn[1][2] = -n[1]*n[2]*cfactor;
  dnhatdn[2][2] = (n[0]*n[0]+n[1]*n[1])*cfactor;
}
inline MY_FLOAT normalize_factor(MY_FLOAT &nx, MY_FLOAT &ny, MY_FLOAT &nz)
{
  MY_FLOAT nnorm = sqrt(nx * nx + ny * ny + nz * nz);
  MY_FLOAT rnnorm = 1 / nnorm;
  nx *= rnnorm;
  ny *= rnnorm;
  nz *= rnnorm;
  return rnnorm;
}
/*
  Yet another normal calculation method for simpiler code.
 */
template <int MAX_NNEIGH>
void PairLMFF::calc_atom_normal(int i, int itype, int *ILP_neigh, int nneigh, MY_FLOAT &nx, MY_FLOAT &ny, MY_FLOAT &nz,
                                MY_FLOAT &dnormdri0, MY_FLOAT &dnormdri1, MY_FLOAT &dnormdri2, MY_FLOAT &dnormdri3,
                                MY_FLOAT &dnormdri4, MY_FLOAT &dnormdri5, MY_FLOAT &dnormdri6, MY_FLOAT &dnormdri7,
                                MY_FLOAT &dnormdri8, MY_FLOAT (*dnormdrk)[9])
{
  double **x = atom->x;
  MY_FLOAT vet[MAX_NNEIGH][3];
  //Sort neighbors for ilp/tmd, etc
  if (MAX_NNEIGH > 3 && nneigh > 3) {
    double *xlast = x[i];
    for (int kk = 0; kk < nneigh; kk++) {
      int jjmin;
      MY_FLOAT rsqmin;
      for (int jj = kk; jj < nneigh; jj++) {
        int j = ILP_neigh[jj] & NEIGHMASK;
        MY_FLOAT delx = x[j][0] - xlast[0];
        MY_FLOAT dely = x[j][1] - xlast[1];
        MY_FLOAT delz = x[j][2] - xlast[2];
        MY_FLOAT rsq = delx * delx + dely * dely + delz * delz;
        if (jj == kk || rsq < rsqmin) {
          jjmin = jj;
          rsqmin = rsq;
        }
      }
      std::swap(ILP_neigh[jjmin], ILP_neigh[kk]);
      xlast = x[ILP_neigh[kk]];
    }
  }
  for (int jj = 0; jj < nneigh; jj++) {
    int j = ILP_neigh[jj] & NEIGHMASK;

    vet[jj][0] = x[j][0] - x[i][0];
    vet[jj][1] = x[j][1] - x[i][1];
    vet[jj][2] = x[j][2] - x[i][2];
  }

  //specialize for AIP_WATER_2DM for hydrogen has special normal vector rule
  if (nneigh <= 1) {
    nx = 0.0;
    ny = 0.0;
    nz = 1.0;
    dnormdri0 = dnormdri1 = dnormdri2 = dnormdri3 = dnormdri4 = dnormdri5 = dnormdri6 = dnormdri7 = dnormdri8 = 0.0;
  } else if (nneigh == 2) {
    nx = vet[0][1] * vet[1][2] - vet[1][1] * vet[0][2];
    ny = vet[0][2] * vet[1][0] - vet[1][2] * vet[0][0];
    nz = vet[0][0] * vet[1][1] - vet[1][0] * vet[0][1];

    MY_FLOAT rnnorm = normalize_factor(nx, ny, nz);
    deriv_normal(dnormdrk[0], vet[1], nx, ny, nz, rnnorm);
    deriv_normal(dnormdrk[1], vet[0], nx, ny, nz, -rnnorm);

    dnormdri0 = -(dnormdrk[0][0] + dnormdrk[1][0]);
    dnormdri1 = -(dnormdrk[0][1] + dnormdrk[1][1]);
    dnormdri2 = -(dnormdrk[0][2] + dnormdrk[1][2]);
    dnormdri3 = -(dnormdrk[0][3] + dnormdrk[1][3]);
    dnormdri4 = -(dnormdrk[0][4] + dnormdrk[1][4]);
    dnormdri5 = -(dnormdrk[0][5] + dnormdrk[1][5]);
    dnormdri6 = -(dnormdrk[0][6] + dnormdrk[1][6]);
    dnormdri7 = -(dnormdrk[0][7] + dnormdrk[1][7]);
    dnormdri8 = -(dnormdrk[0][8] + dnormdrk[1][8]);

  } else if (nneigh >= 3) {
    nx = ny = nz = 0.0;
    for (int kk = 0; kk < nneigh; kk++) {
      int kp1 = (kk + 1 >= nneigh) ? 0 : kk + 1;
      nx += vet[kk][1] * vet[kp1][2] - vet[kp1][1] * vet[kk][2];
      ny += vet[kk][2] * vet[kp1][0] - vet[kp1][2] * vet[kk][0];
      nz += vet[kk][0] * vet[kp1][1] - vet[kp1][0] * vet[kk][1];
    }

    MY_FLOAT rnnorm = normalize_factor(nx, ny, nz);

    dnormdri0 = dnormdri1 = dnormdri2 = dnormdri3 = dnormdri4 = dnormdri5 = dnormdri6 = dnormdri7 = dnormdri8 = 0.0;
    for (int kk = 0; kk < nneigh; kk++) {
      int km1 = (kk - 1 < 0) ? nneigh - 1 : kk - 1;
      int kp1 = (kk + 1 >= nneigh) ? 0 : kk + 1;
      MY_FLOAT del[3];
      del[0] = vet[kp1][0] - vet[km1][0];
      del[1] = vet[kp1][1] - vet[km1][1];
      del[2] = vet[kp1][2] - vet[km1][2];
      deriv_normal(dnormdrk[kk], del, nx, ny, nz, rnnorm);
    }
  }
}


void PairLMFF::update_internal_list()
{
  GPTLstart("update_list");
  int jnum_sum = 0;
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int *tag = (int*)atom->tag;
  double **x = atom->x;
  for (int ii = 0; ii < inum; ii++) { jnum_sum += numneigh[ilist[ii]]; }
  if (inum > inum_max) {
    memory->destroy(num_intra);
    memory->destroy(num_inter);
    memory->destroy(num_vdw);
    memory->sfree(first_layered_neigh);
    memory->sfree(first_bundled_neigh);
    //golden ratio grow
    inum_max = (int) ceil(inum / 0.618);
    memory->create(num_intra, inum_max, "PairLMFF:intra_layer_count");
    memory->create(num_inter, inum_max, "PairLMFF:inter_layer_count");
    memory->create(num_vdw, inum_max, "PairLMFF:vdw_count");
    first_layered_neigh = (int **) memory->smalloc(inum_max * sizeof(int *), "PairLMFF:first_layered_neigh");
    first_bundled_neigh = (int **) memory->smalloc(inum_max * sizeof(int *), "PairLMFF:first_bundled_neigh");
  }
  if (jnum_sum > jnum_max) {
    memory->destroy(layered_neigh);
    memory->destroy(bundled_neigh);
    jnum_max = (int) ceil(jnum_sum / 0.618);
    memory->create(layered_neigh, jnum_max, "PairLMFF:layered_neigh");
    memory->create(bundled_neigh, jnum_max, "PairLMFF:bundled_neigh");
  }

  MY_FLOAT cut_intra = 0;
  for (int i = 0; i < nparams; i++)
    if (params[i].rcut > cut_intra) { cut_intra = params[i].rcut; }

  MY_FLOAT cut_intra_listsq = (cut_intra + neighbor->skin) * (cut_intra + neighbor->skin);

  int total_neigh_intra = 0;
  int total_neigh_inter = 0;

#ifdef LMFF_CLUSTER_STATS
  unsigned long long stat_occurrences = 0;
  unsigned long long stat_unique = 0;
  unsigned long long stat_lane_slots = 0;
#endif

  int num_clusters = (inum + CLUSTERSIZE - 1) / CLUSTERSIZE;


  for (int ic = 0; ic < num_clusters; ic++) {
    int iis = ic * CLUSTERSIZE;
    int iie = std::min(iis + CLUSTERSIZE, inum);
    std::unordered_set<int> filter;
    filter.clear();
    int ninter = 0;
    int *jlist_bundled = first_bundled_neigh[ic] = bundled_neigh + total_neigh_inter;
#ifdef LMFF_CLUSTER_STATS
    unsigned long long stat_cluster_occurrences = 0;
#endif

    for (int ii = iis; ii < iie; ii++) {
      int i = ilist[ii];
      int itag = tag[i];
      int jnum = numneigh[i];
      int *jlist = firstneigh[i];
      int *jlist_layered = first_layered_neigh[i] = layered_neigh + total_neigh_intra;
      int nintra = 0;

      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj] & NEIGHMASK;
        if (atom->molecule[j] == atom->molecule[i]) {
          MY_FLOAT delx = x[i][0] - x[j][0];
          MY_FLOAT dely = x[i][1] - x[j][1];
          MY_FLOAT delz = x[i][2] - x[j][2];
          MY_FLOAT rsq = delx * delx + dely * dely + delz * delz;
          if (rsq < cut_intra_listsq) jlist_layered[nintra++] = j;
        } else {
#ifdef LMFF_CLUSTER_STATS
          ++stat_cluster_occurrences;
#endif
          if (filter.insert(j).second) {
            jlist_bundled[ninter++] = j;
          }
        }
      }

      num_intra[i] = nintra;
      total_neigh_intra += nintra;
    }
    std::sort(jlist_bundled, jlist_bundled + ninter);
    num_inter[ic] = ninter;
    total_neigh_inter += ninter;
#ifdef LMFF_CLUSTER_STATS
    const unsigned long long stat_cluster_width =
        static_cast<unsigned long long>(iie - iis);
    stat_occurrences += stat_cluster_occurrences;
    stat_unique += static_cast<unsigned long long>(ninter);
    stat_lane_slots +=
        stat_cluster_width * static_cast<unsigned long long>(ninter);
#endif
  }

#ifdef LMFF_CLUSTER_STATS
  unsigned long long local_list_stats[3] = {
      stat_occurrences, stat_unique, stat_lane_slots};
  unsigned long long global_list_stats[3] = {0, 0, 0};
  MPI_Allreduce(local_list_stats, global_list_stats, 3,
                MPI_UNSIGNED_LONG_LONG, MPI_SUM, world);
  if (comm->me == 0) {
    const double reuse =
        global_list_stats[1]
        ? static_cast<double>(global_list_stats[0]) /
              static_cast<double>(global_list_stats[1])
        : 0.0;
    const double u_list =
        global_list_stats[2]
        ? static_cast<double>(global_list_stats[0]) /
              static_cast<double>(global_list_stats[2])
        : 0.0;
    const double relative_storage =
        global_list_stats[0]
        ? static_cast<double>(global_list_stats[1]) /
              static_cast<double>(global_list_stats[0])
        : 0.0;
    if (screen)
      fprintf(screen,
              "LMFF_CLUSTER_LIST S=%llu L=%llu slots=%llu "
              "R_reuse=%.6f U_list=%.6f L_over_S=%.6f\n",
              global_list_stats[0], global_list_stats[1],
              global_list_stats[2], reuse, u_list, relative_storage);
    if (logfile)
      fprintf(logfile,
              "LMFF_CLUSTER_LIST S=%llu L=%llu slots=%llu "
              "R_reuse=%.6f U_list=%.6f L_over_S=%.6f\n",
              global_list_stats[0], global_list_stats[1],
              global_list_stats[2], reuse, u_list, relative_storage);
  }
#endif

  GPTLstop("update_list");
}
double PairLMFF::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                                  double /*factor_coul*/, double factor_lj, double &fforce)
{
  double r, r2inv, r6inv, r8inv, forcelj, philj, fpair;
  double Tap, dTap, Vilp, TSvdw, TSvdw2inv;

  int iparam_ij = elem2param[map[itype]][map[jtype]];
  Param &p = params[iparam_ij];

  r = sqrt(rsq);
  // turn on/off taper function
  if (tap_flag) {
    Tap = calc_Tap(r, sqrt(cutsq[itype][jtype]));
    dTap = calc_dTap(r, sqrt(cutsq[itype][jtype]));
  } else {
    Tap = 1.0;
    dTap = 0.0;
  }

  r2inv = 1.0 / rsq;
  r6inv = r2inv * r2inv * r2inv;
  r8inv = r2inv * r6inv;

  TSvdw = 1.0 + exp(-p.d * (r / p.seff - 1.0));
  TSvdw2inv = pow(TSvdw, -2.0);
  Vilp = -p.C6 * r6inv / TSvdw;
  // derivatives
  fpair = -6.0 * p.C6 * r8inv / TSvdw + p.d / p.seff * p.C6 * (TSvdw - 1.0) * r6inv * TSvdw2inv / r;
  forcelj = fpair;
  fforce = factor_lj * (forcelj * Tap - Vilp * dTap / r);

  philj = Vilp * Tap;
  return factor_lj * philj;
}
