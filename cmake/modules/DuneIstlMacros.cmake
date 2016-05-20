# .. cmake_module::
#
#    This modules content is executed whenever a module required or suggests dune-istl!
#

find_package(METIS)
find_package(ParMETIS)
include(AddParMETISFlags)
find_package(SuperLU)
include(AddSuperLUFlags)
find_package(ARPACKPP)
include(AddARPACKPPFlags)
find_package(SuiteSparse OPTIONAL_COMPONENTS LDL SPQR UMFPACK)
include(AddSuiteSparseFlags)
