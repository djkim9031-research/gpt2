cmake_minimum_required(VERSION 3.10)
project(gpt2)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA paths
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# CUDA includes and link libraries
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

# Find the libtorch package
find_package(Torch REQUIRED)

# Find the re2 package
find_package(re2 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/tokenizer/src)

# Add source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Add executable
add_executable(gpt2 ${SOURCES})

# Link libraries
target_link_libraries(gpt2 ${CUDA_LIBRARIES} "${TORCH_LIBRARIES}" re2::re2)