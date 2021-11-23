cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0028 NEW)

# set the project name
project(Kokkos_ifelse)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
)
FetchContent_MakeAvailable(kokkos)

# add_subdirectory(kokkos)

# add the executable
add_executable(kokkos_ifelse main.cpp)

target_link_libraries(kokkos_ifelse Kokkos::kokkos)
