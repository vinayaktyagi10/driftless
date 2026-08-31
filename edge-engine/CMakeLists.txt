cmake_minimum_required(VERSION 3.20)
project(driftless_edge_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(driftless_edge
    src/ukf_fusion_engine.cpp
    src/hmm_map_matcher.cpp
    src/velocity_model.cpp
)
target_include_directories(driftless_edge PUBLIC include)

# TODO: link ONNX Runtime (C++ API) once training/export/ produces a model,
# and a map-matching / OSM-graph dependency shared in spirit with
# android/.../mapmatch/HmmMapMatcher.kt.

enable_testing()
add_subdirectory(tests)
