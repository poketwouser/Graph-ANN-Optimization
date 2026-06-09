cmake_minimum_required(VERSION 3.14)
project(graphann LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags. -s strips symbols: smaller binaries and, on Windows, avoids
# a Smart App Control / WDAC heuristic that intermittently blocks fat unsigned exes.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -s")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    add_link_options(-static)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Library with shared sources
add_library(graphann_lib STATIC
    src/distance.cpp
    src/io_utils.cpp
    src/vamana_index.cpp
    src/kmeans.cpp
)
target_include_directories(graphann_lib PUBLIC include)
target_link_libraries(graphann_lib PUBLIC OpenMP::OpenMP_CXX)

# Build index executable
add_executable(build_index src/build_index.cpp)
target_link_libraries(build_index PRIVATE graphann_lib)

# Search index executable
add_executable(search_index src/search_index.cpp)
target_link_libraries(search_index PRIVATE graphann_lib)

# Cluster builder (Feature 1: K-Means query routing)
add_executable(build_clusters src/build_clusters.cpp)
target_link_libraries(build_clusters PRIVATE graphann_lib)

# Edge-usage refiner (Feature 4: workload-aware graph refinement)
add_executable(refine_index src/refine_index.cpp)
target_link_libraries(refine_index PRIVATE graphann_lib)
