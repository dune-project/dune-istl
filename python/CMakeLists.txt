add_subdirectory(dune)

dune_python_install_package(
  PATH "."
  CMAKE_METADATA_FILE dune/istl/metadata.cmake
  DEPENDS _istl
)
