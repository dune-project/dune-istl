# $Id$

if MPI
  MPITESTS = indicestest indexsettest syncertest vectorcommtest \
	selectiontest
endif

if MPI
  PARMETISTESTS= matrixredisttest
endif

if SUPERLU
  SUPERLUTESTS =  superlutest overlappingschwarztest
endif

if PARDISO
  PARDISOTEST = test_pardiso
endif

# which tests where program to build and run are equal
NORMALTESTS = matrixutilstest matrixtest bvectortest \
	bcrsbuildtest matrixiteratortest mv iotest scaledidmatrixtest

# list of tests to run (indicestest is special case)
TESTS = $(NORMALTESTS) $(MPITESTS) $(SUPERLUTESTS) $(PARDISOTEST) $(PARMETISTESTS)

# programs just to build when "make check" is used
check_PROGRAMS = $(NORMALTESTS) $(MPITESTS) $(SUPERLUTESTS) $(PARDISOTEST) $(PARMETISTESTS)

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

if PARDISO
  test_pardiso_SOURCES = test_pardiso.cc
  test_pardiso_LDADD = $(PARDISO_LIBS) $(LAPACK_LIBS) $(BLAS_LIBS) $(LIBS) $(FLIBS)
endif

bcrsbuildtest_SOURCES = bcrsbuild.cc

bvectortest_SOURCES = bvectortest.cc

matrixutilstest_SOURCES = matrixutilstest.cc laplacian.hh

matrixiteratortest_SOURCES = matrixiteratortest.cc

matrixtest_SOURCES = matrixtest.cc

mv_SOURCES = mv.cc

iotest_SOURCES = iotest.cc

scaledidmatrixtest_SOURCES = scaledidmatrixtest.cc

if MPI
  indicestest_SOURCES = indicestest.cc
  indicestest_CXXFLAGS = $(MPI_CPPFLAGS)
  indicestest_LDFLAGS = $(MPI_LDFLAGS) $(MPI_LIBS)

  selectiontest_SOURCES = selectiontest.cc
  selectiontest_CXXFLAGS = $(MPI_CPPFLAGS)
  selectiontest_LDFLAGS = $(MPI_LDFLAGS) $(MPI_LIBS)

  indexsettest_SOURCES = indexsettest.cc

  syncertest_SOURCES = syncertest.cc
  syncertest_CXXFLAGS = $(MPI_CPPFLAGS)
  syncertest_LDFLAGS = $(DUNE_COMMON_LDFLAGS) $(DUNE_COMMON_LIBS) $(MPI_LDFLAGS) $(MPI_LIBS)

  vectorcommtest_SOURCES = vectorcommtest.cc
  vectorcommtest_CXXFLAGS = $(MPI_CPPFLAGS)
  vectorcommtest_LDFLAGS = $(MPI_LDFLAGS) $(MPI_LIBS)
endif

if MPI
  matrixredisttest_SOURCES = matrixredisttest.cc
  matrixredisttest_CXXFLAGS = ${PARMETIS_CPPFLAGS} $(MPI_CPPFLAGS)
  matrixredisttest_LDFLAGS  = $(AM_LDFLAGS) ${PARMETIS_LDFLAGS} ${PARMETIS_LIBS} ${MPI_LDFLAGS} ${MPI_LIBS}
endif
include $(top_srcdir)/am/global-rules
