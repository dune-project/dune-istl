add_subdirectory("doxygen")
dune_add_latex_document(istl.tex FATHER_TARGET doc
  BIB_FILES istl.bib  DEFAULT_SAFEPDF IMAGES blockstructure.eps)
create_doc_install(istl.pdf
  ${CMAKE_INSTALL_DOCDIR} istl_safepdf)
