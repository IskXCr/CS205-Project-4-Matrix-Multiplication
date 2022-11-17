cmake_minimum_required(VERSION 3.22)

# set the project name
project(Project_4_Matrix_Multiplication)

aux_source_directory(./ DIR_SRCS)

list(REMOVE_ITEM DIR_SRCS "matrix_gen.c")

find_package(OpenMP REQUIRED)

add_compile_options(-O3)

# add the executable
add_executable(test ${DIR_SRCS})
target_link_libraries(test PRIVATE OpenMP::OpenMP_C)