cmake_minimum_required(VERSION 3.20)
project(driftless_edge_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Dependencies ----------------------------------------------------------
# Vendored under third_party/ (gitignored) when present, fetched otherwise, so
# a working tree that already has them does not re-download on every configure.
include(FetchContent)

set(DRIFTLESS_EIGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
if(EXISTS ${DRIFTLESS_EIGEN_DIR}/Eigen/Dense)
    message(STATUS "driftless: using vendored Eigen at ${DRIFTLESS_EIGEN_DIR}")
    add_library(driftless_eigen INTERFACE)
    target_include_directories(driftless_eigen SYSTEM INTERFACE ${DRIFTLESS_EIGEN_DIR})
else()
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(eigen)
    add_library(driftless_eigen INTERFACE)
    target_include_directories(driftless_eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})
endif()

# --- The engine ------------------------------------------------------------
add_library(driftless_edge
    src/so3.cpp
    src/cholesky_update.cpp
    src/sqrt_kalman.cpp
    src/road_graph.cpp
    src/osm_reader.cpp
    src/ukf_fusion_engine.cpp
    src/hmm_map_matcher.cpp
    src/velocity_model.cpp
)
target_include_directories(driftless_edge PUBLIC include)
target_link_libraries(driftless_edge PUBLIC driftless_eigen)
target_compile_options(driftless_edge PRIVATE -Wall -Wextra)

# TODO: link ONNX Runtime (C++ API) once training/export/ produces a model,
# and a map-matching / OSM-graph dependency shared in spirit with
# android/.../mapmatch/HmmMapMatcher.kt.

enable_testing()
add_subdirectory(tests)
