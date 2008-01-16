# $Id$

if MPI
  MPITESTS = indicestest syncertest vectorcommtest \
	selectiontest
endif

if SUPERLU
  SUPERLUTESTS =  superlutest overlappingschwarztest
endif

# which tests where program to build and run are equal
NORMALTESTS = matrixutilstest matrixtest bvectortest indexsettest \
	bcrsbuildtest matrixiteratortest

# list of tests to run (indicestest is special case)
TESTS = $(NORMALTESTS) $(MPITESTS)

# programs just to build when "make check" is used
check_PROGRAMS = $(NORMALTESTS) $(MPITESTS) $(SUPERLUTESTS)

# define the programs

if SUPERLU
  superlutest_SOURCES = superlutest.cc
  superlutest_LDADD= $(SUPERLU_LIBS)
  superlutest_LDFLAGS= $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
  superlutest_CPPFLAGS=$(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS)

  overlappingschwarztest_SOURCES = overlappingschwarztest.cc
  overlappingschwarztest_LDADD= $(SUPERLU_LIBS)
  overlappingschwarztest_LDFLAGS= $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
  overlappingschwarztest_CPPFLAGS=$(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS)
endif
bcrsbuildtest_SOURCES = bcrsbuild.cc

bvectortest_SOURCES = bvectortest.cc
#bvectortest_DEPENDENCIES = $(LOCAL_LIBS)

matrixutilstest_SOURCES = matrixutilstest.cc laplacian.hh
#matrixutilstest_DEPENDENCIES = $(LOCAL_LIBS)

matrixiteratortest_SOURCES = matrixiteratortest.cc

matrixtest_SOURCES = matrixtest.cc


if MPI
  indicestest_SOURCES = indicestest.cc
  indicestest_CXXFLAGS = $(MPI_CPPFLAGS)
  indicestest_LDFLAGS = $(MPI_LDFLAGS) $(MPI_LIBS)
#indicestest_DEPENDENCIES = $(LOCAL_LIBS)

  selectiontest_SOURCES = selectiontest.cc
  selectiontest_CXXFLAGS = $(MPI_CPPFLAGS)
  selectiontest_LDFLAGS = $(MPI_LDFLAGS) $(MPI_LIBS)
#selectiontest_DEPENDENCIES = $(LOCAL_LIBS)

  indexsettest_SOURCES = indexsettest.cc
#indexsettest_DEPENDENCIES = $(LOCAL_LIBS)

  syncertest_SOURCES = syncertest.cc
  syncertest_CXXFLAGS = $(MPI_CPPFLAGS)
  syncertest_LDFLAGS = $(MPI_LDFLAGS) $(MPI_LIBS)
#syncertest_DEPENDENCIES = $(LOCAL_LIBS)

  vectorcommtest_SOURCES = vectorcommtest.cc
  vectorcommtest_CXXFLAGS = $(MPI_CPPFLAGS)
  vectorcommtest_LDFLAGS = $(MPI_LDFLAGS) $(MPI_LIBS)
#vectorcommtest_DEPENDENCIES = $(LOCAL_LIBS)
endif
include $(top_srcdir)/am/global-rules
