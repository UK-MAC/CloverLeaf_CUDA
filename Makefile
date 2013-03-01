ifndef COMPILER
  MESSAGE=select a compiler to compile in OpenMP, e.g. make COMPILER=INTEL
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

# flags for nvcc
# set NV_ARCH to select the correct one
CODE_GEN_FERMI=-gencode arch=compute_20,code=sm_21
CODE_GEN_KEPLER=-gencode arch=compute_30,code=sm_35

# requires CUDA_HOME to be set - not the same on all machines
NV_FLAGS=-O2 -w -c -I $(CUDA_HOME)/include $(CODE_GEN_$(NV_ARCH)) -DNO_ERR_CHK

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

CPPLIBS_PGI=-pgcpplibs
CPPLIBS_GNU=-lstdc++
CPPLIBS=$(CPPLIBS_$(COMPILER))

FLAGS=$(FLAGS_$(COMPILER)) $(OMP) $(I3E) $(OPTIONS) $(RESIDENT_FLAG)
CFLAGS=$(CFLAGS_$(COMPILER)) $(OMP) $(I3E) $(C_OPTIONS) -c
MPI_COMPILER=mpif90
C_MPI_COMPILER=mpicc

CUDA_FILES=\
	mpi_transfers_cuda.o\
	advec_cell_kernel_cuda.o\
	advec_mom_kernel_cuda.o\
	generate_chunk_kernel_cuda.o\
	reset_field_kernel_cuda.o\
	viscosity_kernel_cuda.o\
	initialise_chunk_kernel_cuda.o\
	revert_kernel_cuda.o\
	chunk_cuda.o\
	flux_calc_kernel_cuda.o\
	init_cuda.o\
	accelerate_kernel_cuda.o\
	calc_dt_kernel_cuda.o\
	field_summary_kernel_cuda.o\
	PdV_kernel_cuda.o\
	ideal_gas_kernel_cuda.o\
	update_halo_kernel_cuda.o

all: clover_leaf
	rm -f *.o *.mod *genmod* *.lst

clover_leaf: cuda_clover c_lover *.f90
	$(MPI_COMPILER) $(FLAGS)	\
	data.f90			\
	definitions.f90			\
	cufor_mpi_interop.f90	\
	clover.f90			\
	report.f90			\
	timer.f90			\
	parse.f90			\
	read_input.f90			\
	initialise_chunk_kernel.f90	\
	initialise_chunk.f90		\
	build_field.f90			\
	update_halo_kernel.f90		\
	update_halo.f90			\
	ideal_gas_kernel.f90		\
	ideal_gas.f90			\
	start.f90			\
	generate_chunk_kernel.f90	\
	generate_chunk.f90		\
	initialise.f90			\
	field_summary_kernel.f90	\
	field_summary.f90		\
	viscosity_kernel.f90		\
	viscosity.f90			\
	calc_dt_kernel.f90		\
	calc_dt.f90			\
	timestep.f90			\
	accelerate_kernel.f90		\
	accelerate.f90			\
	revert_kernel.f90		\
	revert.f90			\
	PdV_kernel.f90			\
	PdV.f90				\
	flux_calc_kernel.f90		\
	flux_calc.f90			\
	advec_cell_kernel.f90		\
	advec_cell_driver.f90		\
	advec_mom_kernel.f90		\
	advec_mom_driver.f90		\
	advection.f90			\
	reset_field_kernel.f90		\
	reset_field.f90			\
	hydro.f90			\
	visit.f90			\
	clover_leaf.f90			\
	accelerate_kernel_c.o           \
	revert_kernel_c.o               \
	reset_field_kernel_c.o          \
	advec_mom_kernel_c.o            \
	PdV_kernel_c.o                  \
	flux_calc_kernel_c.o            \
	ideal_gas_kernel_c.o            \
	advec_cell_kernel_c.o           \
	viscosity_kernel_c.o            \
	timer_c.o                       \
	$(CUDA_FILES)	\
	-L $(CUDA_HOME)/lib64 -lcudart $(CPPLIBS) 	\
	-o clover_leaf

c_lover: 
	$(C_MPI_COMPILER) $(CFLAGS)     \
	accelerate_kernel_c.c           \
	PdV_kernel_c.c                  \
	flux_calc_kernel_c.c            \
	revert_kernel_c.c               \
	reset_field_kernel_c.c          \
	ideal_gas_kernel_c.c            \
	viscosity_kernel_c.c            \
	advec_cell_kernel_c.c		\
	timer_c.c            

cuda_clover: $(CUDA_FILES)

%.o: %.cu 
	nvcc $(NV_FLAGS) $<

clean:
	rm -f *.o *.mod *genmod* *.lst

