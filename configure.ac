#! /bin/bash
# $Id$
# Process this file with autoconf to produce a configure script.
DUNE_AC_INIT # gets module version from dune.module file
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([istl/allocator.hh])
AM_CONFIG_HEADER([config.h])

# make dune -> . symlink
DUNE_SYMLINK

# check all dune-module stuff
DUNE_CHECK_ALL

# set up flags for the automated test system
DUNE_AUTOBUILD_FLAGS

# preset variable to path such that #include <dune/...> works
AC_SUBST([AM_CPPFLAGS], '$(DUNE_CPPFLAGS) -I$(top_srcdir)')
AC_SUBST([AM_LDFLAGS], '$(DUNE_LDFLAGS) $(DUNE_LIBS)')

# write output
AC_CONFIG_FILES([Makefile
    doc/comm/Makefile
    doc/comm/Makefile.dist
    doc/comm/figures/Makefile
    doc/doxygen/Makefile
    doc/doxygen/Doxyfile
    doc/Makefile
    doc/Makefile.dist
    istl/paamg/test/Makefile
    istl/paamg/Makefile
    istl/tutorial/Makefile
    istl/test/Makefile
    istl/Makefile
    m4/Makefile
    dune-istl.pc])
AC_OUTPUT

DUNE_SUMMARY_ALL
