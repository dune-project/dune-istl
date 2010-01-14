# $Id$

SUBDIRS = figures

# only build these programs if an MPI-implementation was found
if MPI
  MPIPROGRAMS = indexset
endif
# programs just to build when "make check" is used
check_PROGRAMS = $(MPIPROGRAMS)
# list of tests to run (indicestest is special case)
TESTS = $(MPIPROGRAMS)

if BUILD_DOCS
  DOCFILES = communication.pdf
  EXTRA_DIST = communication.pdf
endif

# setting like in dune-web
CURDIR=doc/istl/comm
BASEDIR=../../..

# install the docs
docdir=$(datadir)/doc/dune-istl/comm

include $(top_srcdir)/am/latex
include $(top_srcdir)/am/webstuff
include $(top_srcdir)/am/global-rules

indexset_SOURCES = indexset.cc buildindexset.hh reverse.hh
indexset_CPPFLAGS = $(AM_CPPFLAGS)		\
	$(DUNEMPICPPFLAGS)
indexset_LDFLAGS = $(AM_LDFLAGS)		\
	$(DUNEMPILDFLAGS)
indexset_LDADD =				\
	$(DUNEMPILIBS)				\
	$(LDADD)
