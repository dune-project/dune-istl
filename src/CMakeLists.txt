add_executable(istl-playground EXCLUDE_FROM_ALL istl-playground.cc)
add_dune_all_flags(istl-playground)
dune_symlink_to_source_files(FILES playground.ini)
dune_add_test(TARGET istl-playground)
