dune_python_add_test(NAME pybcrsmatrix
                     COMMAND ${PYTHON_EXECUTABLE} bcrsmatrix.py
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                     LABELS quick)
