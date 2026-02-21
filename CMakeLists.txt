cmake_minimum_required(VERSION 3.10)
project(GurugramRealEstatePredictor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Armadillo (the high-performance math engine behind mlpack)
find_package(Armadillo REQUIRED)

add_executable(predictor src/main.cpp)
target_link_libraries(predictor PRIVATE ${ARMADILLO_LIBRARIES})
target_include_directories(predictor PRIVATE ${ARMADILLO_INCLUDE_DIRS})