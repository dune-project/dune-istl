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

CLEANFILES = *.aux *.bbl *.blg *.log *.out *.toc *.dvi *.ps

# files to clean only in svn tree but not in tarball
SVNCLEANFILES =  *.pdf
clean-local:
	if test -e $(top_srcdir)/doc/doxygen/Doxydep; then rm -rf $(SVNCLEANFILES); fi

if BUILD_DOCS
dist-hook:
	sed $(srcdir)/Makefile.dist.am -e 's/Makefile\.dist/Makefile/g' > $(distdir)/Makefile.am
	sed $(srcdir)/Makefile.dist.in -e 's/Makefile\.dist/Makefile/g' > $(distdir)/Makefile.in
else
dist-hook:
	echo "# No documentation included in distribution! " > $(distdir)/$(DOCUMENTATION_TAG_FILE)
	sed $(srcdir)/Makefile.dist.am -e 's/Makefile\.dist/Makefile/g' > $(distdir)/Makefile.am
	sed $(srcdir)/Makefile.dist.in -e 's/Makefile\.dist/Makefile/g' > $(distdir)/Makefile.in
endif

include $(top_srcdir)/am/latex
include $(top_srcdir)/am/webstuff
include $(top_srcdir)/am/global-rules
