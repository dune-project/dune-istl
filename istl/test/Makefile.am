# $Id$

EXTRA_DIST = perhapsstartmpi perhapsstopmpi

if MPI
  INDEXPROG = indicestest syncertest vectorcommtest
  SELECTPROG = selectiontest
  INDEXRUN = $(INDEXPROG)
endif

# which tests where program to build and run are equal
NORMALTESTS = matrixutilstest bvectortest indexsettest $(SELECTPROG)

# list of tests to run (indicestest is special case)
TESTS = $(NORMALTESTS) perhapsstartmpi $(INDEXRUN) perhapsstopmpi

# programs just to build when "make check" is used
check_PROGRAMS = $(NORMALTESTS) $(INDEXPROG)

# define the programs

bvectortest_SOURCES = bvectortest.cc
#bvectortest_DEPENDENCIES = $(LOCAL_LIBS)

matrixutilstest_SOURCES = matrixutilstest.cc laplacian.hh
#matrixutilstest_DEPENDENCIES = $(LOCAL_LIBS)

indicestest_SOURCES = indicestest.cc
indicestest_CXXFLAGS = $(MPI_CPPFLAGS)
indicestest_LDFLAGS = $(MPI_LDFLAGS) $(MPI_LIBS)
#indicestest_DEPENDENCIES = $(LOCAL_LIBS)

selectiontest_SOURCES = selectiontest.cc
selectiontest_CXXFLAGS = $(MPI_CPPFLAGS)
selectiontest_LDFLAGS = $(MPI_LDFLAGS) $(MPI_LIBS)
#selectiontest_DEPENDENCIES = $(LOCAL_LIBS)

indexsettest_SOURCES = indexsettest.cc
#indexsettest_DEPENDENCIES = $(LOCAL_LIBS)

syncertest_SOURCES = syncertest.cc
syncertest_CXXFLAGS = $(MPI_CPPFLAGS)
syncertest_LDFLAGS = $(MPI_LDFLAGS) $(MPI_LIBS)
#syncertest_DEPENDENCIES = $(LOCAL_LIBS)

vectorcommtest_SOURCES = vectorcommtest.cc
vectorcommtest_CXXFLAGS = $(MPI_CPPFLAGS)
vectorcommtest_LDFLAGS = $(MPI_LDFLAGS) $(MPI_LIBS)
#vectorcommtest_DEPENDENCIES = $(LOCAL_LIBS)

include $(top_srcdir)/am/global-rules
