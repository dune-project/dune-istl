# $Id$

if MPI
  MPITESTS =
  TESTPROGS = galerkintest hierarchytest pamgtest transfertest pamg_comm_repart_test
endif

NORMALTESTS = amgtest fastamg graphtest kamgtest twolevelmethodtest $(MPITESTS)

# which tests to run
TESTS = $(NORMALTESTS) $(TESTPROGS)

# programs just to build when "make check" is used
check_PROGRAMS = $(TESTPROGS) $(NORMALTESTS)

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

fastamg_SOURCES = fastamg.cc
fastamg_CPPFLAGS = $(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS)
fastamg_LDFLAGS = $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
fastamg_LDADD =					\
	$(SUPERLU_LIBS)				\
	$(LDADD)


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


twolevelmethodtest_SOURCES = twolevelmethodtest.cc
twolevelmethodtest_CPPFLAGS = $(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS)
twolevelmethodtest_LDFLAGS = $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
twolevelmethodtest_LDADD =					\
	$(SUPERLU_LIBS)				\
	$(LDADD)


include $(top_srcdir)/am/global-rules

EXTRA_DIST = CMakeLists.txt
