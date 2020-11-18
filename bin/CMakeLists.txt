add_executable(solve_mm EXCLUDE_FROM_ALL solve_mm.cc)
add_dune_all_flags(solve_mm)
dune_symlink_to_source_files(FILES solve_mm.ini)
dune_add_test(TARGET solve_mm)
