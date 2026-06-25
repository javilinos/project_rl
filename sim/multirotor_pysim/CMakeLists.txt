cmake_minimum_required(VERSION 3.16)
project(multirotor_pysim LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate pybind11 via the installed Python package (apt/pip), robust when no
# system CMake config is present.
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _pybind11_dir)
  execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
list(APPEND CMAKE_PREFIX_PATH "${_pybind11_dir}")
find_package(pybind11 CONFIG REQUIRED)

# Eigen3: prefer the CMake config, else fall back to the standard include dir.
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
  set(EIGEN3_INCLUDE_DIRS /usr/include/eigen3)
endif()

find_package(OpenMP QUIET)

pybind11_add_module(multirotor_pysim src/bindings.cpp)
target_include_directories(multirotor_pysim PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/extern
  ${EIGEN3_INCLUDE_DIRS})
target_compile_options(multirotor_pysim PRIVATE -O3 -funroll-loops)
if(OpenMP_CXX_FOUND)
  target_link_libraries(multirotor_pysim PRIVATE OpenMP::OpenMP_CXX)
endif()

# Drop the .so next to the package so `import multirotor_pysim` works once this
# dir is on PYTHONPATH/sys.path.
set_target_properties(multirotor_pysim PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
