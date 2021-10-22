add_python_targets(istl
  __init__
)
dune_add_pybind11_module(NAME _istl)
set_property(TARGET _istl PROPERTY LINK_LIBRARIES dunecommon APPEND)
if(SKBUILD)
  install(TARGETS _istl LIBRARY DESTINATION python/dune/istl)
endif()
