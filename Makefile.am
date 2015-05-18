# we need the module file to be able to build via dunecontrol
EXTRA_DIST= CMakeLists.txt dune.module

# don't follow the full GNU-standard
# we need automake 1.9 or newer
AUTOMAKE_OPTIONS = foreign 1.9

SUBDIRS = cmake dune doc lib m4

# use configured compiler for "make distcheck"
# doxygen is difficult to convince to build in a special directory...
DISTCHECK_CONFIGURE_FLAGS = --with-dune-common="$(DUNE_COMMON_ROOT)" CXX="$(CXX)" CC="$(CC)" --enable-parallel=@ENABLE_PARALLEL@ MPICC="$(MPICC)"

include $(top_srcdir)/am/global-rules
include $(top_srcdir)/am/top-rules

# Distribute and install config.h.cmake
configdir = $(datadir)/dune-istl
dist_config_DATA = config.h.cmake

include $(top_srcdir)/am/cmake-pkg-config
