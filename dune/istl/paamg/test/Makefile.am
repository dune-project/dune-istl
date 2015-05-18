if MPI
  MPITESTS =
  TESTPROGS = galerkintest hierarchytest pamgtest transfertest pamg_comm_repart_test
endif

if UMFPACK
  UMFPACKAMG = umfpackamgtest umfpackfastamg
endif

if DUNE_PTHREAD
  PTHREADTESTS = pthreadamgtest pthreadfastamgtest pthreadtwoleveltest
endif

NORMALTESTS = amgtest fastamg graphtest kamgtest twolevelmethodschwarztest twolevelmethodtest $(MPITESTS) $(UMFPACKAMG)

# which tests to run
TESTS = $(NORMALTESTS) $(TESTPROGS) $(PTHREADTESTS)

# programs just to build when "make check" is used
check_PROGRAMS = $(TESTPROGS) $(NORMALTESTS) $(PTHREADTESTS)

# define the programs

graphtest_SOURCES = graphtest.cc

transfertest_SOURCES = transfertest.cc
transfertest_CPPFLAGS = $(AM_CPPFLAGS)		\
	$(DUNEMPICPPFLAGS)
transfertest_LDFLAGS = $(AM_LDFLAGS)		\
	$(DUNEMPILDFLAGS)
transfertest_LDADD =				\
	$(DUNEMPILIBS)				\
	$(LDADD)

galerkintest_SOURCES = galerkintest.cc anisotropic.hh
galerkintest_CPPFLAGS = $(AM_CPPFLAGS)		\
	$(DUNEMPICPPFLAGS)
galerkintest_LDFLAGS = $(AM_LDFLAGS)		\
	$(DUNEMPILDFLAGS)
galerkintest_LDADD =				\
	$(DUNEMPILIBS)				\
	$(LDADD)

hierarchytest_SOURCES = hierarchytest.cc anisotropic.hh
hierarchytest_CPPFLAGS = $(AM_CPPFLAGS)		\
	$(DUNEMPICPPFLAGS)
hierarchytest_LDADD =				\
	$(DUNEMPILIBS)				\
	$(LDADD)
hierarchytest_LDFLAGS = $(AM_LDFLAGS)		\
	$(DUNEMPILDFLAGS)

amgtest_SOURCES = amgtest.cc
amgtest_CPPFLAGS = $(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS)
amgtest_LDFLAGS = $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
amgtest_LDADD =					\
	$(SUPERLU_LIBS)				\
	$(LDADD)

umfpackamgtest_SOURCES = amgtest.cc
umfpackamgtest_CPPFLAGS = $(AM_CPPFLAGS) $(UMFPACK_CPPFLAGS)
umfpackamgtest_LDFLAGS = $(AM_LDFLAGS) $(UMFPACK_LDFLAGS)
umfpackamgtest_LDADD = $(UMFPACK_LIBS)

fastamg_SOURCES = fastamg.cc
fastamg_CPPFLAGS = $(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS)
fastamg_LDFLAGS = $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
fastamg_LDADD =					\
	$(SUPERLU_LIBS)				\
	$(LDADD)

umfpackfastamg_SOURCES = fastamg.cc
umfpackfastamg_CPPFLAGS = $(AM_CPPFLAGS) $(UMFPACK_CPPFLAGS)
umfpackfastamg_LDFLAGS = $(AM_LDFLAGS) $(UMFPACK_LDFLAGS)
umfpackfastamg_LDADD = $(UMFPACK_LIBS)

kamgtest_SOURCES = kamgtest.cc
kamgtest_CPPFLAGS = $(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS)
kamgtest_LDFLAGS = $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
kamgtest_LDADD =					\
	$(SUPERLU_LIBS)				\
	$(LDADD)

pamgtest_SOURCES = parallelamgtest.cc
pamgtest_CPPFLAGS = $(AM_CPPFLAGS)		\
	$(DUNEMPICPPFLAGS)			\
	$(SUPERLU_CPPFLAGS)			\
	$(PARMETIS_CPPFLAGS)
pamgtest_LDFLAGS = $(AM_LDFLAGS)		\
	$(DUNEMPILDFLAGS)			\
	$(PARMETIS_LDFLAGS)
pamgtest_LDADD =				\
	$(PARMETIS_LIBS)			\
	$(SUPERLU_LIBS)				\
	$(DUNEMPILIBS)				\
	$(LDADD)

pamg_comm_repart_test_SOURCES = parallelamgtest.cc
pamg_comm_repart_test_CPPFLAGS = $(AM_CPPFLAGS)		\
	$(DUNEMPICPPFLAGS)			\
	$(SUPERLU_CPPFLAGS)			\
	$(PARMETIS_CPPFLAGS) -DAMG_REPART_ON_COMM_GRAPH
pamg_comm_repart_test_LDFLAGS = $(AM_LDFLAGS)		\
	$(DUNEMPILDFLAGS)			\
	$(PARMETIS_LDFLAGS)
pamg_comm_repart_test_LDADD =				\
	$(PARMETIS_LIBS)			\
	$(SUPERLU_LIBS)				\
	$(DUNEMPILIBS)				\
	$(LDADD)

twolevelmethodschwarztest_SOURCES = twolevelmethodtest.cc
twolevelmethodschwarztest_CPPFLAGS = $(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS) -DUSE_OVERLAPPINGSCHWARZ
twolevelmethodschwarztest_LDFLAGS = $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
twolevelmethodschwarztest_LDADD =					\
	$(SUPERLU_LIBS)				\
	$(LDADD)

twolevelmethodtest_SOURCES = twolevelmethodtest.cc
twolevelmethodtest_CPPFLAGS = $(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS)
twolevelmethodtest_LDFLAGS = $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
twolevelmethodtest_LDADD =					\
	$(SUPERLU_LIBS)				\
	$(LDADD)

pthreadamgtest_SOURCES = pthreadamgtest.cc
pthreadamgtest_CPPFLAGS = $(AM_CPPFLAGS) $(PTHREAD_CFLAGS) -DMYAMG="Dune::Amg::AMG<Operator,Vector,Smoother>"
pthreadamgtest_LDFLAGS = $(AM_LDFLAGS) $(PTHREAD_CFLAGS)
pthreadamgtest_LDADD = $(PTHREAD_LIBS) $(LDADD)

pthreadfastamgtest_SOURCES = pthreadamgtest.cc
pthreadfastamgtest_CPPFLAGS = $(AM_CPPFLAGS) $(PTHREAD_CFLAGS) -DMYAMG="Dune::Amg::FastAMG<Operator,Vector>"
pthreadfastamgtest_LDFLAGS = $(AM_LDFLAGS) $(PTHREAD_CFLAGS)
pthreadfastamgtest_LDADD = $(PTHREAD_LIBS) $(LDADD)

pthreadtwoleveltest_SOURCES = pthreadtwoleveltest.cc
pthreadtwoleveltest_CPPFLAGS = $(AM_CPPFLAGS) $(PTHREAD_CFLAGS)
pthreadtwoleveltest_LDFLAGS = $(AM_LDFLAGS) $(PTHREAD_CFLAGS)
pthreadtwoleveltest_LDADD = $(PTHREAD_LIBS) $(LDADD)

include $(top_srcdir)/am/global-rules

EXTRA_DIST = CMakeLists.txt
