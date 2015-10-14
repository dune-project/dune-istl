# .. cmake_module::
#
#    This modules content is executed whenever a module required or suggests dune-istl!
#

find_package(METIS)
find_package(ParMETIS)
include(AddParMETISFlags)
find_package(SuperLU)
include(AddSuperLUFlags)
find_package(UMFPack)
include(AddUMFPackFlags)
find_package(ARPACKPP)
include(AddARPACKPPFlags)
