cmake_minimum_required(VERSION 3.20)
project(driftless_edge_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Dependencies ----------------------------------------------------------
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
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR do-not-configure)
    FetchContent_MakeAvailable(eigen)
    add_library(driftless_eigen INTERFACE)
    target_include_directories(driftless_eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})
endif()

# --- ONNX Runtime (C++ API) -------------------------------------------------
# Vendored as a prebuilt binary distribution, not built from source: ONNX
# Runtime's own build is a heavyweight, multi-hour affair with its own build
# tooling that has no place in this project's configure step. Same
# vendored-or-fetched convention as Eigen above: drop a copy under
# third_party/onnxruntime/{include,lib} yourself (gitignored) to skip the
# download, or let this pull the official Linux x64 release tarball.
set(DRIFTLESS_ORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime)
set(DRIFTLESS_ORT_VERSION 1.19.2)

if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(DRIFTLESS_ORT_PLATFORM "osx-arm64")
    else()
        set(DRIFTLESS_ORT_PLATFORM "osx-x86_64")
    endif()
    set(DRIFTLESS_ORT_LIB_NAME "libonnxruntime.dylib")
elseif(UNIX)
    set(DRIFTLESS_ORT_PLATFORM "linux-x64")
    set(DRIFTLESS_ORT_LIB_NAME "libonnxruntime.so")
else()
    message(FATAL_ERROR "driftless: no ONNX Runtime download configured for this platform -- "
                         "vendor it manually under third_party/onnxruntime")
endif()

if(EXISTS ${DRIFTLESS_ORT_DIR}/include/onnxruntime_cxx_api.h)
    message(STATUS "driftless: using vendored ONNX Runtime at ${DRIFTLESS_ORT_DIR}")
else()
    set(DRIFTLESS_ORT_URL
        "https://github.com/microsoft/onnxruntime/releases/download/v${DRIFTLESS_ORT_VERSION}/onnxruntime-${DRIFTLESS_ORT_PLATFORM}-${DRIFTLESS_ORT_VERSION}.tgz")
    set(DRIFTLESS_ORT_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.tgz)
    message(STATUS "driftless: downloading ONNX Runtime ${DRIFTLESS_ORT_VERSION} for ${DRIFTLESS_ORT_PLATFORM}")
    file(DOWNLOAD ${DRIFTLESS_ORT_URL} ${DRIFTLESS_ORT_ARCHIVE} SHOW_PROGRESS)
    file(MAKE_DIRECTORY ${DRIFTLESS_ORT_DIR})
    file(ARCHIVE_EXTRACT INPUT ${DRIFTLESS_ORT_ARCHIVE}
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_extracted)
    file(GLOB DRIFTLESS_ORT_EXTRACTED
         ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_extracted/onnxruntime-${DRIFTLESS_ORT_PLATFORM}-*)
    file(COPY ${DRIFTLESS_ORT_EXTRACTED}/include DESTINATION ${DRIFTLESS_ORT_DIR})
    file(COPY ${DRIFTLESS_ORT_EXTRACTED}/lib DESTINATION ${DRIFTLESS_ORT_DIR})
endif()

# macOS quarantines anything downloaded via file(DOWNLOAD ...); the first
# process to load a quarantined dylib blocks on a Gatekeeper check long
# enough to blow gtest_discover_tests' timeout. Clear it every configure.
if(APPLE)
    execute_process(COMMAND xattr -cr ${DRIFTLESS_ORT_DIR} ERROR_QUIET)
endif()

# Unset before searching: find_file() caches its result, so without this a
# lib/ directory that changes contents between configures (e.g. swapping in a
# different ORT build) would keep resolving to the old, possibly-missing path.
unset(DRIFTLESS_ORT_LIB_PATH CACHE)
find_file(DRIFTLESS_ORT_LIB_PATH ${DRIFTLESS_ORT_LIB_NAME}
    PATHS ${DRIFTLESS_ORT_DIR}/lib NO_DEFAULT_PATH)
if(NOT DRIFTLESS_ORT_LIB_PATH)
    message(FATAL_ERROR "driftless: expected ${DRIFTLESS_ORT_LIB_NAME} under "
                         "${DRIFTLESS_ORT_DIR}/lib but didn't find it -- run "
                         "`ls ${DRIFTLESS_ORT_DIR}/lib` and check what actually landed there.")
endif()

add_library(driftless_onnxruntime SHARED IMPORTED)
set_target_properties(driftless_onnxruntime PROPERTIES
    IMPORTED_LOCATION ${DRIFTLESS_ORT_LIB_PATH}
    INTERFACE_INCLUDE_DIRECTORIES ${DRIFTLESS_ORT_DIR}/include)

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
    src/imu_derived.cpp
)
target_include_directories(driftless_edge PUBLIC include)
target_link_libraries(driftless_edge PUBLIC driftless_eigen driftless_onnxruntime)
target_compile_options(driftless_edge PRIVATE -Wall -Wextra)

# So `./build/tests/test_velocity_model` finds libonnxruntime.so without
# needing LD_LIBRARY_PATH set by hand.
set_target_properties(driftless_edge PROPERTIES
    BUILD_RPATH ${DRIFTLESS_ORT_DIR}/lib)

enable_testing()
add_subdirectory(tests)

option(DRIFTLESS_BUILD_APPS "Build example/runner executables" ON)
if(DRIFTLESS_BUILD_APPS)
    add_subdirectory(apps)
endif()