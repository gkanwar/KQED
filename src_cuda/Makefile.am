TOPDIR=@top_srcdir@
BUILDDIR=@top_builddir@

lib_LIBRARIES = libKQED_cuda.a

## just include the header directory
AM_CFLAGS=-I${TOPDIR}/src_cuda/HEADERS/

INCLLIBFILES=./LIBFILES/all_kernels.cu.h \
	./LIBFILES/cheby.cu.h ./LIBFILES/chnr_dV.cu.h ./LIBFILES/chnr_dS.cu.h \
	./LIBFILES/chnr_dT.cu.h ./LIBFILES/con_kernel.cu.h \
	./LIBFILES/getff-new.cu.h ./LIBFILES/kernels.cu.h ./LIBFILES/pi_pert.cu.h \
	./LIBFILES/QED_kernel.cu.h ./LIBFILES/QED_kernel_xy0.cu.h \
	./LIBFILES/SYMXY.cu.h ./LIBFILES/SYMXY0.cu.h \
	./LIBFILES/Tabd.cu.h \
        ./LIBFILES/sub_kernel.cu.h
LIBCUFILES=./LIBFILES/KQED.cu
LIBCUFILES_OBJS=$(LIBCUFILES:%.cu=%.o)
LIBCFILES=./LIBFILES/init.c \
	./LIBFILES/io.c
LIBFILES=$(LIBCUFILES) $(LIBCFILES)


UTILSFILES=./UTILS/GLU_timer.c ./UTILS/GLU_bswap.c ./UTILS/crc32c.c

## all the source files apart from amu_forlattice.c
libKQED_cuda_a_SOURCES = ${LIBFILES} ${UTILSFILES} ${INCLLIBFILES}
libKQED_cuda_a_AR = $(NVCC) $(NVCCFLAGS) -lib -o
libKQED_cuda_a_LIBADD = $(CUDA_LIBS)
libKQED_cuda_a_CFLAGS = $(CUDA_CFLAGS) $(INCLUDES) $(CFLAGS) $(AM_CFLAGS)

## don't know if AM_*FLAGS need to be explicit here or if there is a better way
$(LIBCUFILES_OBJS): %.o: %.cu $(INCLLIBFILES:./LIBFILES/%=${TOPDIR}/src_cuda/LIBFILES/%)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(DEFS) $(DEFAULT_INCLUDES) \
	--compiler-options="$(CFLAGS) $(CPPFLAGS) $(AM_CFLAGS) $(AM_CPPFLAGS)" \
	-Xcicc -O2 -dc -o $@ $< # cicc O3 takes forever
.cu.o:
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(DEFS) $(DEFAULT_INCLUDES) \
	--compiler-options="$(CFLAGS) $(CPPFLAGS) $(AM_CFLAGS) $(AM_CPPFLAGS)" \
	-Xcicc -O2 -dc -o $@ $< # cicc O3 takes forever
# .c.o:
# 	$(NVCC) -x cu $(NVCCFLAGS) $(INCLUDES) $(DEFS) $(DEFAULT_INCLUDES) \
# 	--compiler-options="$(CFLAGS) $(CPPFLAGS) $(AM_CFLAGS) $(AM_CPPFLAGS)" \
# 	-dc -o $@ $<

## and their respective headers
include_HEADERS = \
	$(INCLLIBFILES:./LIBFILES/%.cu.h=./HEADERS/%.h) \
	$(LIBCFILES:./LIBFILES/%.c=./HEADERS/%.h) \
	$(UTILSFILES:./UTILS/%.c=./HEADERS/%.h) \
	./HEADERS/KQED.h \
	./HEADERS/cu_util.h \
	../config.h
# 

if !PREF

Bindir = "${prefix}"/bin

bin_PROGRAMS = KQED

KQED_SOURCES = amu_forlattice.cu
KQED_CFLAGS = $(CUDA_CFLAGS) $(INCLUDES) $(CFLAGS)
KQED_LDADD = libKQED_cuda.a ${LDFLAGS} -lm $(CUDA_LIBS) $(CUDA_LDFLAGS) -lcudart
KQED_LINK = $(NVCC) $(NVCCFLAGS) --compiler-options="$(KQED_CFLAGS)" $(KQED_LDFLAGS) $(LDFLAGS) -o $@

endif

