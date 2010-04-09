# $Id$

SUBDIRS = doxygen comm

TEXSOURCES = istl.tex istl.bib
EPSFILES = blockstructure.eps
if BUILD_DOCS
  DOCFILES = istl.pdf
  EXTRA_DIST = istl.pdf
  EXTRAINSTALL = istl.pdf
endif

# setting like in dune-web
BASEDIR=../..
CURDIR=doc/istl

# where to install the docs
docdir=$(datadir)/doc/dune-istl

include $(top_srcdir)/am/latex
include $(top_srcdir)/am/webstuff
include $(top_srcdir)/am/global-rules
