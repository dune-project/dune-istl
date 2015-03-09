if MPI
  MPITESTS = vectorcommtest matrixmarkettest
endif

if MPI
  PARMETISTESTS= matrixredisttest
endif

if SUPERLU
  SUPERLUTESTS =  superlutest superluztest superluctest superlustest
endif

if PARDISO
  PARDISOTEST = test_pardiso
endif

if UMFPACK
  UMFPACKTEST = umfpacktest
endif

# which tests where program to build and run are equal
NORMALTESTS = basearraytest \
              bcrsassigntest \
              bcrsbuildtest \
              bcrsimplicitbuildtest \
              bvectortest \
              complexmatrixtest \
              complexrhstest \
              dotproducttest \
              iotest \
              inverseoperator2prectest \
              matrixiteratortest \
              matrixtest \
              matrixutilstest \
              mmtest \
              mv \
              overlappingschwarztest \
              scaledidmatrixtest \
              seqmatrixmarkettest \
              solvertest \
              vbvectortest

# list of tests to run (indicestest is special case)
TESTS = $(NORMALTESTS) $(MPITESTS) $(SUPERLUTESTS) $(PARDISOTEST) $(PARMETISTESTS) $(UMFPACKTEST)

# programs just to build when "make check" is used
check_PROGRAMS = $(NORMALTESTS) $(MPITESTS) $(SUPERLUTESTS) $(PARDISOTEST) $(PARMETISTESTS) $(UMFPACKTEST)

# define the programs

overlappingschwarztest_SOURCES = overlappingschwarztest.cc
overlappingschwarztest_LDADD =
overlappingschwarztest_LDFLAGS= $(AM_LDFLAGS)
overlappingschwarztest_CPPFLAGS= $(AM_CPPFLAGS)

if SUPERLU
  superlutest_SOURCES = superlutest.cc
  superlutest_LDADD= $(SUPERLU_LIBS)
  superlutest_LDFLAGS= $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
  superlutest_CPPFLAGS=$(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS)

  superlustest_SOURCES = superlutest.cc
  superlustest_LDADD= $(SUPERLU_LIBS)
  superlustest_LDFLAGS= $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
  superlustest_CPPFLAGS=$(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS) -DSUPERLU_NTYPE=0


  superluctest_SOURCES = superlutest.cc
  superluctest_LDADD= $(SUPERLU_LIBS)
  superluctest_LDFLAGS= $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
  superluctest_CPPFLAGS=$(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS) -DSUPERLU_NTYPE=2


  superluztest_SOURCES = superlutest.cc
  superluztest_LDADD= $(SUPERLU_LIBS)
  superluztest_LDFLAGS= $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
  superluztest_CPPFLAGS=$(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS) -DSUPERLU_NTYPE=3

  overlappingschwarztest_LDADD += $(SUPERLU_LIBS)
  overlappingschwarztest_LDFLAGS +=  $(SUPERLU_LDFLAGS)
  overlappingschwarztest_CPPFLAGS += $(SUPERLU_CPPFLAGS)
endif

if UMFPACK
  umfpacktest_SOURCES= umfpacktest.cc
  umfpacktest_LDADD= $(UMFPACK_LIBS)
  umfpacktest_LDFLAGS= $(AM_LDFLAGS) $(UMFPACK_LDFLAGS)
  umfpacktest_CPPFLAGS= $(AM_CPPFLAGS) $(UMFPACK_CPPFLAGS)

  overlappingschwarztest_LDADD += $(UMFPACK_LIBS)
  overlappingschwarztest_LDFLAGS += $(UMFPACK_LDFLAGS)
  overlappingschwarztest_CPPFLAGS += $(UMFPACK_CPPFLAGS)
endif

if PARDISO
  test_pardiso_SOURCES = test_pardiso.cc
  test_pardiso_LDADD = $(PARDISO_LIBS) $(LAPACK_LIBS) $(BLAS_LIBS) $(LIBS) $(FLIBS)
endif

basearraytest_SOURCES = basearraytest.cc

bcrsassigntest_SOURCES = bcrsassigntest.cc

bcrsbuildtest_SOURCES = bcrsbuild.cc

bcrsimplicitbuildtest_SOURCES = bcrsimplicitbuild.cc
bcrsimplicitbuildtest_CPPFLAGS = $(AM_CPPFLAGS) -DDUNE_ISTL_WITH_CHECKING=1

bvectortest_SOURCES = bvectortest.cc

complexmatrixtest_SOURCES = complexmatrixtest.cc complexdata.hh

complexrhstest_SOURCES = complexrhstest.cc laplacian.hh
complexrhstest_LDADD= $(SUPERLU_LIBS)
complexrhstest_LDFLAGS= $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
complexrhstest_CPPFLAGS=$(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS) -DSUPERLU_NTYPE=3

dotproducttest_SOURCES = dotproducttest.cc

vbvectortest_SOURCES = vbvectortest.cc

matrixutilstest_SOURCES = matrixutilstest.cc laplacian.hh

matrixiteratortest_SOURCES = matrixiteratortest.cc

matrixtest_SOURCES = matrixtest.cc

mmtest_SOURCES = mmtest.cc

mv_SOURCES = mv.cc

iotest_SOURCES = iotest.cc

inverseoperator2prectest_SOURCES = inverseoperator2prectest.cc

scaledidmatrixtest_SOURCES = scaledidmatrixtest.cc

solvertest_SOURCES = solvertest.cc

if MPI
  vectorcommtest_SOURCES = vectorcommtest.cc
  vectorcommtest_CPPFLAGS = $(AM_CPPFLAGS)	\
	$(DUNEMPICPPFLAGS)
  vectorcommtest_LDFLAGS = $(AM_LDFLAGS)	\
	$(DUNEMPILDFLAGS)
  vectorcommtest_LDADD =			\
	$(DUNEMPILIBS)				\
	$(LDADD)
  matrixmarkettest_SOURCES = matrixmarkettest.cc
  matrixmarkettest_CPPFLAGS = $(AM_CPPFLAGS)	\
	$(DUNEMPICPPFLAGS)
  matrixmarkettest_LDFLAGS = $(AM_LDFLAGS)	\
	$(DUNEMPILDFLAGS)
  matrixmarkettest_LDADD =			\
	$(DUNEMPILIBS)				\
	$(LDADD)
endif

seqmatrixmarkettest_SOURCES = matrixmarkettest.cc

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

headercheck_IGNORE = complexdata.hh

include $(top_srcdir)/am/global-rules

EXTRA_DIST = CMakeLists.txt

CLEANFILES = umfpack_decomp
