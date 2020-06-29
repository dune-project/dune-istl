add_executable(poweriterationtest cond2test.cc)
dune_target_link_libraries(poweriterationtest ${dune-common_LIBRARIES})
dune_add_test(
  TARGET poweriterationtest
  CMD_ARGS 40)

if(SUPERLU_FOUND)
  add_executable(poweriterationsuperlutest cond2test.cc)
  add_dune_superlu_flags(poweriterationsuperlutest)
  dune_add_test(
    TARGET poweriterationsuperlutest
    CMD_ARGS 40)
  dune_target_link_libraries(poweriterationsuperlutest ${dune-common_LIBRARIES})
endif()

if(ARPACKPP_FOUND)
  add_executable(arpackpptest cond2test.cc)
  add_dune_arpackpp_flags(arpackpptest)
  dune_target_link_libraries(arpackpptest ${dune-common_LIBRARIES})
  dune_add_test(TARGET arpackpptest)
endif()

if(ARPACKPP_FOUND AND SUPERLU_FOUND)
  dune_add_test(NAME arpackppsuperlutest
                SOURCES cond2test.cc)
endif()
