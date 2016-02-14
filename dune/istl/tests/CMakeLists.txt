project(dune-cg-unit-tests)

cmake_minimum_required(VERSION 2.8)

# Set path to cmake scripts
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/../../../cmake/modules
    )

# Do not allow in-source builds
include(ConfigSafeGuards)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -coverage -Wall -Wwrite-strings -Wpedantic -std=c++11")

# set up include-directories
include_directories( ${PROJECT_SOURCE_DIR}/../../.. )
include_directories( ${PROJECT_SOURCE_DIR}/../../../../dune-common )
link_directories( ${PROJECT_SOURCE_DIR}/../../../../dune-common/build-cmake/lib )

aux_source_directory( . SRC_LIST )
aux_source_directory( mock SRC_LIST )
aux_source_directory( fglue/tmp SRC_LIST )
aux_source_directory( fglue/fusion SRC_LIST )
aux_source_directory( functional_tests SRC_LIST )

add_executable(unit_tests ${SRC_LIST})

target_link_libraries(unit_tests LINK_PUBLIC gtest pthread dunecommon)

include(CTest)
enable_testing()
add_test(test ${PROJECT_BINARY_DIR}/unit_tests)