add_subdirectory("doxygen")
add_subdirectory("comm")
if(LATEX_USABLE)
  ADD_LATEX_DOCUMENT(istl.tex FATHER_TARGET doc
    BIB_FILES istl.bib  DEFAULT_SAFEPDF IMAGES blockstructure.eps)
  create_doc_install(istl.pdf
    share/doc/dune-istl/ istl_safepdf)
endif(LATEX_USABLE)
