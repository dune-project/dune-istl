#! /bin/bash
# $Id$
# Process this file with autoconf to produce a configure script.
AC_INIT(dune-istl, 0.0, dune@hal.iwr.uni-heidelberg.de)
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([istl/allocator.hh])
AM_CONFIG_HEADER([config.h])

# check all dune-module stuff
DUNE_CHECK_ALL_M([dunecommon])

if test x$HAVE_DUNECOMMON != x1 ; then
  AC_MSG_ERROR([Can't work without the dune-common module. Maybe you have to supply your dune-common directory as --with-dunecommon=dir])
fi

# preset variable to path such that #include <dune/...> works
AC_SUBST([AM_CPPFLAGS], '$(DUNE_CPPFLAGS) -I$(top_srcdir)')
AC_SUBST([AM_LDFLAGS], '$(DUNE_LDFLAGS) $(DUNE_LIBS)')

DUNE_SUMMARY_ALL

echo
echo Note: Most of the libraries checked for above are only used for the self-test
echo of Dune.  The library itself will build and the headers will work even if
echo Albert, MPI, etc. cannot be found.  An exception to this are UG and AmiraMesh
echo which need to be found right now if you want to use them later.
echo

# write output
AC_CONFIG_FILES([Makefile
    doc/istl/comm/Makefile
    doc/istl/Makefile
    doc/layout/Makefile
    doc/doxygen/Makefile
    doc/Makefile
    istl/paamg/test/Makefile
    istl/paamg/Makefile
    istl/tutorial/Makefile
    istl/test/Makefile
    istl/Makefile
    dune.pc])
AC_OUTPUT
