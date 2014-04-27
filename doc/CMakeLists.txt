add_subdirectory(doxygen)

dune_add_latex_document(istl.tex
  FATHER_TARGET doc
  DEFAULT_SAFEPDF
  BIBFILES istl.bib
  IMAGES blockstructure.eps)
create_doc_install(istl.pdf
  ${CMAKE_INSTALL_DOCDIR} istl_safepdf)
