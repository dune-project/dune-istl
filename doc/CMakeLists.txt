add_subdirectory(doxygen)

add_latex_document(
  SOURCE istl.tex
  FATHER_TARGET doc
  INSTALL ${CMAKE_INSTALL_DOCDIR})
