add_executable(poweriterationtest cond2test.cc)
target_link_libraries(poweriterationtest dunecommon)
dune_add_test(TARGET poweriterationtest)

if(SUPERLU_FOUND)
  add_executable(poweriterationsuperlutest cond2test.cc)
  add_dune_superlu_flags(poweriterationsuperlutest)
  dune_add_test(TARGET poweriterationsuperlutest)
  target_link_libraries(poweriterationsuperlutest dunecommon)
endif(SUPERLU_FOUND)

if(ARPACKPP_FOUND)
  add_executable(arpackpptest cond2test.cc)
  add_dune_arpackpp_flags(arpackpptest)
  target_link_libraries(arpackpptest dunecommon)
  dune_add_test(TARGET arpackpptest)
endif(ARPACKPP_FOUND)

if(ARPACKPP_FOUND AND SUPERLU_FOUND)
  dune_add_test(NAME arpackppsuperlutest
                SOURCES cond2test.cc)
endif()
