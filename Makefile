#Crown Copyright 2012 AWE.
#
# This file is part of CloverLeaf.
#
# CloverLeaf is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# CloverLeaf is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# CloverLeaf. If not, see http://www.gnu.org/licenses/.

#  @brief Makefile for CloverLeaf
#  @author Michael Boulton Wayne Gaudin
#  @details Agnostic, platform independent makefile for the Clover Leaf benchmark code.


ifndef COMPILER
  MESSAGE=select a compiler to compile in OpenMP, e.g. make COMPILER=INTEL
endif

ifndef NV_ARCH
  MESSAGE=select an NVIDA device to compile in CUDA, e.g. make NV_ARCH=KEPLER
endif

OMP_INTEL     = -openmp
OMP_SUN       = -xopenmp=parallel -vpara
OMP_GNU       = -fopenmp
OMP_CRAY      =
OMP_PGI       = -mp=nonuma
OMP_PATHSCALE = -mp
OMP_XLF       = -qsmp=omp
OMP=$(OMP_$(COMPILER))

FLAGS_INTEL     = -O3 -ipo
FLAGS_SUN       = -O2
FLAGS_GNU       = -O2
FLAGS_CRAY      = -O2 -em -ra -f free -F
FLAGS_PGI       = -O2 -Mpreprocess
FLAGS_PATHSCALE = -O2
FLAGS_XLF       = -O2
FLAGS_          = -O2
CFLAGS_INTEL     = -O3 -ipo -restrict -fno-alias
CFLAGS_SUN       = -O2
CFLAGS_GNU       = -O2
CFLAGS_CRAY      = -em
CFLAGS_PGI       = -O2
CFLAGS_PATHSCALE = -O2
CFLAGS_XLF       = -O2
CFLAGS_          = -O2

ifdef DEBUG
  FLAGS_INTEL     = -O0 -g -debug all -check all -traceback -check noarg_temp_created
  FLAGS_SUN       = -O0 -xopenmp=noopt -g
  FLAGS_GNU       = -O0 -g
  FLAGS_CRAY      = -O0 -g -em -eD
  FLAGS_PGI       = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk -Mpreprocess
  FLAGS_PATHSCALE = -O0 -g
  FLAGS_XLF       = -O0 -g
  FLAGS_          = -O0 -g
  CFLAGS_INTEL    = -O0 -g -c -debug all -traceback -restrict
  CFLAGS_CRAY     = -O0 -g -em -eD
  NV_FLAGS += -g -G
endif

ifdef IEEE
  I3E_INTEL     = -fp-model strict -fp-model source -prec-div -prec-sqrt
  I3E_SUN       = -fsimple=0 -fns=no
  I3E_GNU       = -ffloat-store
  I3E_CRAY      = -hflex_mp=intolerant
  I3E_PGI       = -Kieee
  I3E_PATHSCALE = -mieee-fp
  I3E_XLF       = -qfloat=nomaf
  I3E=$(I3E_$(COMPILER))
endif

# flags for nvcc
# set NV_ARCH to select the correct one
NV_ARCH=KEPLER
CODE_GEN_FERMI=-gencode arch=compute_20,code=sm_21
CODE_GEN_KEPLER=-gencode arch=compute_35,code=sm_35

LDLIBS+=-lstdc++ -lcudart

FLAGS=$(FLAGS_$(COMPILER)) $(OMP) $(I3E) $(OPTIONS)
CFLAGS=$(CFLAGS_$(COMPILER)) $(OMP) $(I3E) $(C_OPTIONS) -c
MPI_COMPILER=mpif90
C_MPI_COMPILER=mpicc

NV_FLAGS+=-D MANUALLY_CHOOSE_GPU

# requires CUDA_HOME to be set - not the same on all machines
NV_FLAGS=-I$(CUDA_HOME)/include $(CODE_GEN_$(NV_ARCH)) -restrict -Xcompiler "$(CFLAGS_GNU)" -D MPI_HDR $(NV_OPTIONS)
NV_FLAGS+=-D NO_ERR_CHK

ifdef DEBUG
NV_FLAGS+=-O0 -g -G
else
NV_FLAGS+=-O3
endif

C_FILES=\
	accelerate_kernel_c.o           \
	pack_kernel_c.o \
	PdV_kernel_c.o                  \
	timer_c.o                  \
	initialise_chunk_kernel_c.o                  \
	calc_dt_kernel_c.o                  \
	field_summary_kernel_c.o                  \
	update_halo_kernel_c.o                  \
	generate_chunk_kernel_c.o                  \
	flux_calc_kernel_c.o            \
	revert_kernel_c.o               \
	reset_field_kernel_c.o          \
	ideal_gas_kernel_c.o            \
	viscosity_kernel_c.o            \
	advec_cell_kernel_c.o			\
	advec_mom_kernel_c.o

FORTRAN_FILES=\
	clover.o \
	pack_kernel.o \
	data.o			\
	definitions.o			\
	report.o			\
	timer.o			\
	parse.o			\
	read_input.o			\
	initialise_chunk_kernel.o	\
	initialise_chunk.o		\
	build_field.o			\
	update_halo_kernel.o		\
	update_halo.o			\
	ideal_gas_kernel.o		\
	ideal_gas.o			\
	start.o			\
	generate_chunk_kernel.o	\
	generate_chunk.o		\
	initialise.o			\
	field_summary_kernel.o	\
	field_summary.o		\
	viscosity_kernel.o		\
	viscosity.o			\
	calc_dt_kernel.o		\
	calc_dt.o			\
	timestep.o			\
	accelerate_kernel.o		\
	accelerate.o			\
	revert_kernel.o		\
	revert.o			\
	PdV_kernel.o			\
	PdV.o				\
	flux_calc_kernel.o		\
	flux_calc.o			\
	advec_cell_kernel.o		\
	advec_cell_driver.o		\
	advec_mom_kernel.o		\
	advec_mom_driver.o		\
	advection.o			\
	reset_field_kernel.o		\
	reset_field.o			\
	hydro.o			\
	visit.o			\
	clover_leaf.o

CUDA_FILES= \
	accelerate_kernel_cuda.o \
	advec_cell_kernel_cuda.o \
	advec_mom_kernel_cuda.o \
	calc_dt_kernel_cuda.o \
	cuda_errors.o \
	cuda_strings.o \
	field_summary_kernel_cuda.o \
	flux_calc_kernel_cuda.o \
	generate_chunk_kernel_cuda.o \
	ideal_gas_kernel_cuda.o \
	init_cuda.o \
	initialise_chunk_kernel_cuda.o \
	pack_kernel_cuda.o \
	PdV_kernel_cuda.o \
	reset_field_kernel_cuda.o \
	revert_kernel_cuda.o \
	update_halo_kernel_cuda.o \
	viscosity_kernel_cuda.o

clover_leaf: Makefile $(FORTRAN_FILES) $(C_FILES) $(CUDA_FILES)
	$(MPI_COMPILER) $(FLAGS)	\
	$(FORTRAN_FILES)	\
	$(C_FILES)	\
	$(CUDA_FILES) \
	$(LDFLAGS) \
	$(LDLIBS) \
	-o clover_leaf
	@echo $(MESSAGE)

include make.deps

%.o: %.cu Makefile make.deps
	nvcc $(NV_FLAGS) -c $< -o $*.o
%.mod %_module.mod %_leaf_module.mod: %.f90 %.o
	@true
%.o: %.f90 Makefile make.deps
	$(MPI_COMPILER) $(FLAGS) -c $< -o $*.o
%.o: %.c Makefile make.deps
	$(C_MPI_COMPILER) $(CFLAGS) -c $< -o $*.o

clean:
	rm -f *.o *.mod *genmod* *.lst *.cub *.ptx clover_leaf
