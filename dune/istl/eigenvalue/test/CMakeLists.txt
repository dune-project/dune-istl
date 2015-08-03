set(NORMALTESTS
  poweriterationtest
)

if(SUPERLU_FOUND)
  set(SUPERLUTESTS
    poweriterationsuperlutest
  )
endif(SUPERLU_FOUND)

if(ARPACKPP_FOUND)
  set(ARPACKPPTESTS
    arpackpptest
  )
endif(ARPACKPP_FOUND)

if((ARPACKPP_FOUND) AND (SUPERLU_FOUND))
  set(ARPACKPPSUPERLUTESTS
    arpackppsuperlutest
  )
endif((ARPACKPP_FOUND) AND (SUPERLU_FOUND))

set(ALLTESTS ${NORMALTESTS} ${SUPERLUTESTS} ${ARPACKPPTESTS} ${ARPACKPPSUPERLUTESTS})


# We do not want want to build the tests during make all,
# but just build them on demand
add_directory_test_target(_test_target)
add_dependencies(${_test_target} ${ALLTESTS})


# Provide source files
add_executable(poweriterationtest "cond2test.cc")

if(SUPERLU_FOUND)
  add_executable(poweriterationsuperlutest "cond2test.cc")
  add_dune_superlu_flags("${SUPERLUTESTS}")
endif(SUPERLU_FOUND)

if(ARPACKPP_FOUND)
  add_executable(arpackpptest "cond2test.cc")
  add_dune_arpackpp_flags("${ARPACKPPTESTS}")
endif(ARPACKPP_FOUND)

if((ARPACKPP_FOUND) AND (SUPERLU_FOUND))
  add_executable(arpackppsuperlutest "cond2test.cc")
  add_dune_arpackpp_flags("${ARPACKPPSUPERLUTESTS}")
  add_dune_superlu_flags("${ARPACKPPSUPERLUTESTS}")
endif((ARPACKPP_FOUND) AND (SUPERLU_FOUND))

foreach(_exe ${ALLTESTS})
  target_link_libraries(${_exe} "dunecommon")
  add_test(${_exe} ${_exe})
endforeach(_exe ${ALLTESTS})
