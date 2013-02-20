add_executable(poosc08 "poosc08.cc")
add_executable(poosc08_test "poosc08_test.cc")
add_executable(indexset "indexset.cc")
add_dune_mpi_flags(poosc08;poosc08test;indexset)
include(UseLATEX)
ADD_LATEX_DOCUMENT(communication.tex FATHER_TARGET doc
  BIBFILES communcation.bib DEFAULT_PDF INPUTS poosc08_test.cc
  IMAGE_DIRS figures)
create_doc_install(${CMAKE_CURRENT_BINARY_DIR}/communication.pdf share/doc/dune-istl/comm)
