# POP-LMFF: A Portable High-Performance Layered Materials Force Field

POP-LMFF is a high-performance implementation of the Layered Materials Force Field (LMFF) in LAMMPS. It provides portable SIMD acceleration for multi-core processors, including x86 processors with AVX-512 and ARM processors with SVE/SVE2.

Supported execution modes:

- Full FP64 precision
- FP32 mixed-precision execution

Supported architectures:

- x86 CPUs with AVX-512
- ARM CPUs with SVE/SVE2 512-bit vectors


## Required LAMMPS Packages

The following LAMMPS packages are required:

- MANYBODY
- INTERLAYER
- MOLECULE
- EXTRA-FIX


# Dependencies

## Compiler Requirements

### AVX-512 Platforms

- Intel oneAPI compiler (`icpx/icx`)
- Intel MKL


### ARM SVE/SVE2 Platforms

- GCC or Clang compiler supporting SVE/SVE2
- ARMv9 architecture with 512-bit SVE vectors


## GPTL Timing Library

POP-LMFF uses **GPTL (General Purpose Timing Library)** for performance profiling.

GPTL is not included in this repository. Users may need to compile and install GPTL manually before building POP-LMFF.

After installation, please ensure that the compiler and linker can locate the GPTL headers and libraries.

If performance profiling is not required, GPTL-related components can be disabled.


# Compilation

## AVX-512 Compilation

The AVX-512 implementation targets x86 processors using Intel oneAPI compilers.


### FP64 Version

```bash
cmake \
-D CMAKE_CXX_COMPILER=icpx \
-D CMAKE_C_COMPILER=icx \
-D PKG_MANYBODY=ON \
-D PKG_INTERLAYER=ON \
-D PKG_MOLECULE=ON \
-D PKG_EXTRA-FIX=ON \
-D CMAKE_CXX_STANDARD=17 \
-D CMAKE_CXX_FLAGS="-O3 -ffast-math -march=native -qmkl -DLMFF_SVN_MATH_MKL" \
../cmake
````

### FP32 Mixed-Precision Version

The FP32 mixed-precision implementation enables optimized single-precision vector execution.

```bash
cmake \
-D CMAKE_CXX_COMPILER=icpx \
-D CMAKE_C_COMPILER=icx \
-D PKG_MANYBODY=ON \
-D PKG_INTERLAYER=ON \
-D PKG_MOLECULE=ON \
-D PKG_EXTRA-FIX=ON \
-D CMAKE_CXX_STANDARD=17 \
-D CMAKE_CXX_FLAGS="-O3 -ffast-math -march=native -DLMFF_MIXED_PREC -qmkl -DLMFF_SVN_MATH_MKL" \
../cmake
```

## ARM SVE Compilation

The SVE implementation targets ARMv9 processors with 512-bit vector length.

### FP64 Version

```bash
cmake \
-D PKG_MANYBODY=ON \
-D PKG_INTERLAYER=ON \
-D PKG_MOLECULE=ON \
-D PKG_EXTRA-FIX=ON \
-D CMAKE_CXX_STANDARD=17 \
-D CMAKE_CXX_FLAGS="-O3 -funroll-loops -mcpu=native -ffast-math -std=gnu++17 -march=armv9+sve2 -msve-vector-bits=512" \
../cmake
```

### FP32 Mixed-Precision Version

```bash
cmake \
-D PKG_MANYBODY=ON \
-D PKG_INTERLAYER=ON \
-D PKG_MOLECULE=ON \
-D PKG_EXTRA-FIX=ON \
-D CMAKE_CXX_STANDARD=17 \
-D CMAKE_CXX_FLAGS="-O3 -funroll-loops -mcpu=native -ffast-math -std=gnu++17 -march=armv9+sve2 -msve-vector-bits=512 -DLMFF_MIXED_PREC" \
../cmake
```

# Build

After configuration, compile POP-LMFF using:

```bash
make -j
```

The generated LAMMPS executable is:

```bash
./lmp
```

# Running Example

A single-core LMFF simulation can be executed as follows:

```bash
mpirun -np 1 ./lmp -in in.grhBN.lmff
```

For multi-core and multi-node simulations, users can adjust the MPI process number and OpenMP settings according to the target platform.

# License

POP-LMFF follows the license of the original LAMMPS project.
