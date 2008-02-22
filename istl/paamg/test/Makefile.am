# $Id$

if MPI
  MPITESTS =
  TESTPROGS = galerkintest hierarchytest pamgtest transfertest
endif

NORMALTESTS = amgtest graphtest $(MPITESTS)

# which tests to run
TESTS = $(NORMALTESTS) $(TESTPROGS)

# programs just to build when "make check" is used
check_PROGRAMS = $(TESTPROGS) $(NORMALTESTS)

# define the programs

graphtest_SOURCES = graphtest.cc

transfertest_SOURCES = transfertest.cc
transfertest_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CPPFLAGS)
transfertest_LDFLAGS = $(AM_LDFLAGS) $(MPI_LDFLAGS)

galerkintest_SOURCES = galerkintest.cc anisotropic.hh
galerkintest_CXXFLAGS = $(AM_CPPFLAGS)  $(MPI_CPPFLAGS)
galerkintest_LDFLAGS = $(AM_LDFLAGS) $(MPI_LDFLAGS)

hierarchytest_SOURCES = hierarchytest.cc anisotropic.hh
hierarchytest_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CPPFLAGS)
hierarchytest_LDADD =  $(AM_LIBS) $(MPI_LIBS)
hierarchytest_LDFLAGS = $(AM_LDFLAGS) $(MPI_LDFLAGS)

amgtest_SOURCES = amgtest.cc
amgtest_CPPFLAGS = $(AM_CPPFLAGS) $(SUPERLU_CPPFLAGS)
amgtest_LDFLAGS = $(AM_LDFLAGS) $(SUPERLU_LDFLAGS)
amgtest_LDADD = $(SUPERLU_LIBS)

pamgtest_SOURCES = parallelamgtest.cc
pamgtest_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CPPFLAGS) $(SUPERLU_CPPFLAGS)
pamgtest_LDFLAGS = $(AM_LDFLAGS) $(MPI_LDFLAGS)
pamgtest_LDADD = $(MPI_LIBS) $(SUPERLU_LIBS)

include $(top_srcdir)/am/global-rules
