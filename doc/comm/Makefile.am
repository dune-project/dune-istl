# $Id$

SUBDIRS = figures

# only build these programs if an MPI-implementation was found
if MPI
  MPIPROGRAMS = indexset
endif

SUFFIXES = .dvi .tex .pdf

if BUILD_DOCS
  DOCS = communication.pdf communication.ps
endif


# setting like in dune-web
RELDIR=istl/comm
CURDIR=doc/$(RELDIR)
BASEDIR=../../..
commdir=$(datadir)/doc/dune-istl/comm

# list of tests to run (indicestest is special case)
TESTS = $(MPIPROGRAMS)

# programs just to build when "make check" is used
check_PROGRAMS = $(MPIPROGRAMS)

indexset_SOURCES = indexset.cc buildindexset.hh reverse.hh
indexset_CXXFLAGS = $(MPI_CPPFLAGS)
indexset_LDADD = $(MPI_LDFLAGS)

comm_DATA = $(DOCS) #$(indexset_SOURCES)

EXTRA_DIST = communication.tex
communication.dvi : communication.tex buildindexset.hh reverse.hh


CLEANFILES = *.aux *.bbl *.blg *.log *.out *.toc *.dvi *.pdf *.ps

include $(top_srcdir)/am/latex
include $(top_srcdir)/am/webstuff
include $(top_srcdir)/am/global-rules
