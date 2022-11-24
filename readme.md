# CS205 C/C++ Programming - Lab Project 4 - Optimization of Matrix Multiplication

**Name**: FANG Jiawei

**SID**: 12110804

## Part 0 - Link
[IskXCr/CS205-Project-4-Matrix-Multiplication](https://github.com/IskXCr/CS205-Project-4-Matrix-Multiplication)


## Part 1 - Bug Fixes and Improvements on Project 3
1. Naive for-loop for initializing elements of the float array to zero are kept, in order to let the compiler optimize this (Which is more efficient than manually implement an algorithm by myself).
2. Removed global linked list for thread safety. Atomic reference count.
```c
    atomic_size_t refs; /* Reference to this matrix */
```
3. Fixed loop order in matrix multiplication to avoid tranposing the result. Now the program only tranposes op2 to optimize access speed.
4. Use ``omp_get_wtime()`` as the appropriate timer.

## Part 2 - Matrix Multiplication