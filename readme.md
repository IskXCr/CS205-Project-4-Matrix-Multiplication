# CS205 C/C++ Programming - Lab Project 4 - Optimization of Matrix Multiplication

**Name**: FANG Jiawei

**SID**: 12110804

## Part 0 - Link
[IskXCr/CS205-Project-4-Matrix-Multiplication](https://github.com/IskXCr/CS205-Project-4-Matrix-Multiplication)


## Part 1 - Bug Fixes and Minor Improvements on Project 3
1. Naive for-loop for initializing elements of the float array to zero are kept, in order to let the compiler optimize this (Which is more efficient than manually implement an algorithm by myself).
2. Removed global linked list for thread safety. Added atomic reference count.
```c
    atomic_size_t refs; /* Reference to this matrix */
```
3. Fixed loop order in matrix multiplication to avoid tranposing the result. Now the program only tranposes op2 to optimize access speed.
4. Use ``omp_get_wtime()`` as the appropriate timer.
5. Improved ``matrix_errno`` for reporting various kind of errors occurred in matrix operations.
6. Optimized language: use *sequential access* instead of *linear access*

## Part 2 - Major Improvements on Matrix Multiplication
1. Used Cache-oblivious algorithm for tranposing a matrix (divide-and-conquer approach).

## Part 3 - Automated Tests
```shell
Matrix Multiplication Plain: 52.308680 s
Matrix Multiplication Ver 1: 0.556091 s
Tranposing took: 0.008345 s
Matrix Multiplication Ver 2: 0.202117 s
```