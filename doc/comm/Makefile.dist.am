# $Id: Makefile.am 665 2006-10-30 13:15:50Z christi $

SUBDIRS = figures

# only build these programs if an MPI-implementation was found
if MPI
  MPIPROGRAMS = indexset
endif

commdir=$(datadir)/doc/dune-istl/comm

comm_DATA = communication.pdf

CONFIG_CLEAN_FILES = *.dist

include $(top_srcdir)/am/latex
include $(top_srcdir)/am/webstuff
include $(top_srcdir)/am/global-rules
