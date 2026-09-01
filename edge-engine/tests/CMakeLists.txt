set(DRIFTLESS_GTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/googletest)
if(EXISTS ${DRIFTLESS_GTEST_DIR}/CMakeLists.txt)
    message(STATUS "driftless: using vendored GoogleTest at ${DRIFTLESS_GTEST_DIR}")
    add_subdirectory(${DRIFTLESS_GTEST_DIR} ${CMAKE_CURRENT_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)
else()
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.16.0
        GIT_SHALLOW TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

# Trajectory generation shared by the dynamics tests. Deliberately independent
# of the engine's own mechanization -- see trajectory_sim.h for why.
add_library(driftless_test_support support/trajectory_sim.cpp)
target_include_directories(driftless_test_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(driftless_test_support PUBLIC driftless_edge)

function(driftless_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE driftless_edge driftless_test_support GTest::gtest_main)
    target_compile_options(${name} PRIVATE -Wall -Wextra)
    target_compile_definitions(${name} PRIVATE
        DRIFTLESS_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
    gtest_discover_tests(${name})
endfunction()

driftless_add_test(test_cholesky_update)
driftless_add_test(test_so3)
driftless_add_test(test_sqrt_kalman)
driftless_add_test(test_sigma_points)
driftless_add_test(test_predict)
driftless_add_test(test_updates)
driftless_add_test(test_map_matching)
driftless_add_test(test_osm_reader)
driftless_add_test(test_drift_characterization)
driftless_add_test(test_blackout_scenario)
