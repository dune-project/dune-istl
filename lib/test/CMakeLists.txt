
foreach(BLOCK ${DUNE_ISTL_PRECOMPILE_BLOCKS})
  string(REGEX REPLACE "[^a-zA-Z0-9]" "" BLOCK_CLEAN "${BLOCK}")
  configure_file(solverfactorytest.cc.in solverfactorytest_${BLOCK_CLEAN}.cc)
  dune_add_test(SOURCES ${CMAKE_CURRENT_BINARY_DIR}/solverfactorytest_${BLOCK_CLEAN}.cc
    LINK_LIBRARIES "-Wl,-whole-archive" duneistl "-Wl,-no-whole-archive")
endforeach()

dune_symlink_to_source_files(FILES solverfactorytest.ini)
