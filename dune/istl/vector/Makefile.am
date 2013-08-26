SUBDIRS =

vectordir = $(includedir)/dune/istl/vector
vector_HEADERS =			\
  cuda.hh				\
  host.hh


include $(top_srcdir)/am/global-rules

EXTRA_DIST = CMakeLists.txt

LINK = $CC -o $@ -lcudart -L/sfw/cuda/5.5/lib64
lib_LIBRARIES = libcuda_backend.a
libcuda_backend_a_SOURCES = cuda_backend.cu

NVCC = nvcc
NVCCFLAGS = -g -v --ptxas-options=-v -O3 -I$(top_srcdir) $(DEFS) -use_fast_math

.cu.o:
		$(NVCC) $(NVCCFLAGS)  -c -o $@ $<
