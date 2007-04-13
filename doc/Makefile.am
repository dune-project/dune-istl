# $Id$

SUBDIRS = doxygen layout comm

TEXSOURCES = istl.tex istl.bib
EPSFILES = blockstructure.eps
EXTRA_DIST = istl.pdf Makefile.dist.in Makefile.dist.am
if BUILD_DOCS
  DOCS = istl.pdf
endif


# setting like in dune-web
BASEDIR=../..
CURDIR=doc/istl
docdir=$(datadir)/doc/dune-istl

doc_DATA = $(DOCS)

CLEANFILES = *.aux *.bbl *.blg *.log *.out *.toc *.dvi *.pdf *.ps

dist-hook:
	cp $(srcdir)/Makefile.dist.am $(distdir)/Makefile.am
	cp $(srcdir)/Makefile.dist.in $(distdir)/Makefile.in

include $(top_srcdir)/am/latex
include $(top_srcdir)/am/webstuff
include $(top_srcdir)/am/global-rules
