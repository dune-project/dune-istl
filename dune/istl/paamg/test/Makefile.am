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

include $(top_srcdir)/am/global-rules
