SUBDIRS =

vectordir = $(includedir)/dune/istl/vector
vector_HEADERS =			\
  cuda.hh				\
  cuda_allocator.hh \
  cuda_backend.hh \
  cuda_kernels.hh \
  host.hh


include $(top_srcdir)/am/global-rules

EXTRA_DIST = CMakeLists.txt

LINK = $CC -o $@ -lcudart -L/sfw/cuda/5.5/lib64 -lcublas
lib_LIBRARIES = libcuda_backend.a
libcuda_backend_a_SOURCES = cuda_backend.cu cuda_kernels.cu cuda_allocator.cu

NVCC = nvcc
NVCCFLAGS = $(DUNE_CPPFLAGS)-g -v --ptxas-options=-v -O3 -I$(top_srcdir) $(DEFS) -use_fast_math -arch sm_13

.cu.o:
		$(NVCC) $(NVCCFLAGS)  -c -o $@ $<
