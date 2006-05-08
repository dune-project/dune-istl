# $Id$

SUBDIRS = doxygen layout comm

if BUILD_DOCS
  DOCS = istl.pdf istl.ps
endif


# setting like in dune-web
BASEDIR=../..
CURDIR=doc/istl
docdir=$(datadir)/doc/dune/istl

doc_DATA = $(DOCS)

CLEANFILES = *.aux *.bbl *.blg *.log *.out *.toc *.dvi *.pdf *.ps


include $(top_srcdir)/am/latex
include $(top_srcdir)/am/webstuff
include $(top_srcdir)/am/global-rules
