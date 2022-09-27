add_subdirectory(dune)

dune_python_configure_bindings(
  PATH "."
  CMAKE_METADATA_FLAGS DUNE_OPTS_FILE
)
if(POLICY CMP0087)
  cmake_policy(SET CMP0087 NEW)
endif()
