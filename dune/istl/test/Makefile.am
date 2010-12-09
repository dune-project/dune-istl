# $Id$

if MPI
  MPITESTS = vectorcommtest
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
NORMALTESTS = basearraytest matrixutilstest matrixtest bvectortest vbvectortest \
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

basearraytest_SOURCES = basearraytest.cc

bcrsbuildtest_SOURCES = bcrsbuild.cc

bvectortest_SOURCES = bvectortest.cc

vbvectortest_SOURCES = vbvectortest.cc

matrixutilstest_SOURCES = matrixutilstest.cc laplacian.hh

matrixiteratortest_SOURCES = matrixiteratortest.cc

matrixtest_SOURCES = matrixtest.cc

mv_SOURCES = mv.cc

iotest_SOURCES = iotest.cc

scaledidmatrixtest_SOURCES = scaledidmatrixtest.cc

if MPI
  vectorcommtest_SOURCES = vectorcommtest.cc
  vectorcommtest_CPPFLAGS = $(AM_CPPFLAGS)	\
	$(DUNEMPICPPFLAGS)
  vectorcommtest_LDFLAGS = $(AM_LDFLAGS)	\
	$(DUNEMPILDFLAGS)
  vectorcommtest_LDADD =			\
	$(DUNEMPILIBS)				\
	$(LDADD)
endif

if MPI
  matrixredisttest_SOURCES = matrixredisttest.cc
  matrixredisttest_CPPFLAGS = $(AM_CPPFLAGS)	\
	$(DUNEMPICPPFLAGS)			\
	${PARMETIS_CPPFLAGS}
  matrixredisttest_LDFLAGS = $(AM_LDFLAGS)	\
	$(DUNEMPILDFLAGS)			\
	${PARMETIS_LDFLAGS}
  matrixredisttest_LDADD =			\
	${PARMETIS_LDFLAGS} ${PARMETIS_LIBS}	\
	${DUNEMPILIBS}				\
	$(LDADD)
endif
include $(top_srcdir)/am/global-rules
