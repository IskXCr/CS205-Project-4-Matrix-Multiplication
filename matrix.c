/* Source file for matrix operations */

#include "matrix.h"
#include "matrix_utils.h"

#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <immintrin.h>
#include <float.h>
#include <math.h>
#include <ctype.h>
#include <omp.h>

/* Definitions for ease */

/* enum declarations */

/* Defines common operations that can be performed on an element-by-element basis */
typedef enum op_code
{
    ADD,      /* Add op1 to op2 */
    SUBTRACT, /* Subtract op2 from op1 */
    MULTIPLY, /* Multiply op1 by op2 */
    DIVIDE    /* Divide op1 by op2 */
} op_code;

/* Global Variables */

static const size_t SZT_MAX = (size_t)-1; /* The maximum size of size_t */

/* Functions*/

/* Test if a set of rows and cols exceeds the maximum float number size_t can represent. */
static inline int
_is_param_valid(const size_t rows, const size_t cols)
{
    return !(SZT_MAX / sizeof(float) < rows || SZT_MAX / sizeof(float) < cols || SZT_MAX / sizeof(float) / rows < cols);
}

/* Create a matrix based on given number rows and cols.

   If rows * cols exceeds the upper limit of size_t, or if rows and cols are completely invalid,
   or if no extra memory is available, return NULL. */
matrix create_matrix(const size_t rows, const size_t cols)
{
    /* Check parameters */
    if (rows == 0 || cols == 0 || !_is_param_valid(rows, cols))
        return NULL;

    matrix result; /* Store the result matrix */
    float *arr;    /* Store the result float array */

    result = (matrix)malloc(sizeof(matrix_struct));
    if (result == NULL)
    {
        out_of_memory();
        return NULL;
    }
    arr = (float *)aligned_alloc(32, rows * cols * sizeof(float));
    if (arr == NULL)
    {
        out_of_memory();
        free(result);
        return NULL;
    }

    result->rows = rows;
    result->cols = cols;
    result->arr = arr;
    result->refs = 1;

    for (size_t i = 0; i < rows * cols; ++i)
        result->arr[i] = 0.0f;

    return result;
}

/* Free a matrix, assuming it is completely legal, or simply points to NULL, or is a null pointer.
   Otherwise, nothing is done. */
void delete_matrix(matrix *m)
{
    if (m == NULL || *m == NULL)
        return;
    if ((*m)->refs != 0)
        --((*m)->refs);

    /* If no more references exist */
    if ((*m)->refs == 0)
    {
        free((*m)->arr);
        free(*m);
    }

    *m = NULL;
}

/* Copy src matrix to dest. If the size of dest matrix doesn't fit, reallocate space and modify dest.
   If source matrix is NULL, do nothing; If the destination matrix is NULL, try to allocate a new one.
   It is assumed that both the source and the destination matrix is valid (i.e., created by
   function create_matrix() or NULL), and thus rows * cols won't overflow, and an invalid pointer won't be dereferenced.
   The caller must check the destination to ensure that all the routines that possess a reference
   to this matrix will allow such changes.

   If failed to reallocate space, first call out_of_memory, then nothing is done on dest. */
matrix_errno copy_matrix(matrix *dest, const matrix src)
{
    if (src == NULL)
        return OP_NULL_PTR;

    size_t target_size = src->rows * src->cols; /* Target size*/

    /* If destionation is not NULL and size of arr space doesn't match */
    if (*dest != NULL && ((*dest)->rows * (*dest)->cols) != target_size)
    {
        float *newarr;
        if ((newarr = (float *)aligned_alloc(32, target_size * sizeof(float))) == NULL)
        {
            out_of_memory();
            return OUT_OF_MEMORY;
        }
        free((*dest)->arr);
        (*dest)->arr = newarr;
    }
    /* Allocate space for destination matrix if it is NULL. If allocation failed, return immediately. */
    else if (*dest == NULL && (*dest = create_matrix(src->rows, src->cols)) == NULL)
        return OUT_OF_MEMORY;

    /* Start copying */
    (*dest)->rows = src->rows;
    (*dest)->cols = src->cols;
    memcpy((*dest)->arr, src->arr, target_size * sizeof(float));

    return COMPLETED;
}

/* Return a reference to the target matrix. If refs count exceeds the upper bound, return NULL.

   If a NULL pointer is received, do nothing and return NULL. */
matrix ref_matrix(matrix m)
{
    if (m != NULL)
    {
        if (m->refs < SZT_MAX)
            m->refs++;
        else
            return NULL;
    }
    return m;
}

/* Return 1 if two matrix are equal, or both NULL, with error less than or equals to ERR. ERR must be positive. */
int test_equality(const matrix op1, const matrix op2, float ERR)
{
    if (op1 == op2)
        return 1;

    if (op1 == NULL || op2 == NULL || op1->rows != op2->rows || op1->cols != op2->cols)
        return 0;

    for (size_t i = 0; i < op1->rows * op1->cols; ++i)
        if (fabsf(op1->arr[i] - op2->arr[i]) > ERR)
            return 0;

    return 1;
}

// Threshold for primitive tranposing.
#define _MAT_TRANS_THRESHOLD 8U

/* Cache-oblivious function for recursively transposing a matrix */
static void
_transpose_mat(float *src, float *dest, size_t r_offset, size_t c_offset,
               size_t src_rows, size_t src_cols, size_t rows, size_t cols)
{
    if (rows <= _MAT_TRANS_THRESHOLD && cols <= _MAT_TRANS_THRESHOLD)
    {
        for (size_t i = r_offset; i < r_offset + rows; ++i)
            for (size_t j = c_offset; j < c_offset + cols; ++j)
                dest[j * src_rows + i] = src[i * src_cols + j];
        return;
    }

    /* Cut the larger dimension in half. Cut rows first. */
    if (rows >= cols)
    {
        _transpose_mat(src, dest, r_offset, c_offset, src_rows, src_cols, rows / 2, cols);
        _transpose_mat(src, dest, r_offset + rows / 2, c_offset, src_rows, src_cols, rows - rows / 2, cols);
    }
    else
    {
        _transpose_mat(src, dest, r_offset, c_offset, src_rows, src_cols, rows, cols / 2);
        _transpose_mat(src, dest, r_offset, c_offset + cols / 2, src_rows, src_cols, rows, cols - cols / 2);
    }
}

/* Transpose a matrix and store the result to a second matrix.
   The result matrix can refer to the src or point to other valid matrix, or simply NULL.
   The pointer to the result matrix cannot be NULL.
   If not, undefined behaviour will occur. If the result matrix isn't ablt to store the result, i.e.,
   either it is NULL or the size doesn't match, the result matrix would be modified to match the need.

   If errors occurred during the operation (for example, memory allocation failure), do nothing on the result matrix.
   Returns the corresonding errno code upon failure. */
matrix_errno transpose_matrix(const matrix src, matrix *result)
{
    if (src == NULL)
        return OP_NULL_PTR;

    size_t s_rows = src->rows; /* src row size */
    size_t s_cols = src->cols; /* src col size */
    size_t r_rows = s_cols;    /* result row size */
    size_t r_cols = s_rows;    /* result col size */

    float *newarr;

    if (*result == NULL)
    {
        if ((*result = create_matrix(r_rows, r_cols)) == NULL) /* If failed to create a new matrix */
        {
            out_of_memory();
            return OUT_OF_MEMORY;
        }
        else
            newarr = (*result)->arr;
    }
    /* Else prepare the result array */
    else if ((newarr = (float *)aligned_alloc(32, (r_rows * r_cols) * sizeof(float))) == NULL)
    {
        out_of_memory();
        return OUT_OF_MEMORY;
    }

    /* Tranpose*/
    // for (size_t i = 0; i < s_rows; ++i)
    //     for (size_t j = 0; j < s_cols; ++j)
    //         newarr[j * r_cols + i] = src->arr[i * s_cols + j];
    _transpose_mat(src->arr, newarr, 0, 0, src->rows, src->cols, src->rows, src->cols);

    /* Clean up */
    (*result)->rows = r_rows;
    (*result)->cols = r_cols;
    if ((*result)->arr != newarr)
    {
        free((*result)->arr);
        (*result)->arr = newarr;
    }

    return COMPLETED;
}

/* Perform EleMent-by-eleMent Arithmetic (EMMA) operation on two matrices and store the result in a third matrix.
   The result matrix can refer to either op1 or op2 at the same time, but the pointer to the result matrix cannot be NULL.
   All operands, including the result, are supposed to be NULL or valid (i.e. created using function create_matrix()),
   so that an invalid pointer won't be dereferenced.
   If not, undefined behaviour will occur. If the result matrix isn't able to store the sum, i.e.,
   either it is NULL or the size doesn't match, the result matrix would be modified to match the need.

   If errors occurred during the operation (for example, operand size unmatches), do nothing on the result matrix.
   However, the user must handle 0 divisor error themselves.
   Returns the corresonding errno code upon failure. */
static matrix_errno
_do_emma_on_matrices(const matrix op1, const matrix op2, matrix *result, op_code code)
{
    if (op1 == NULL || op2 == NULL)
        return OP_NULL_PTR;

    size_t rows, cols, size;
    rows = op1->rows;
    cols = op1->cols;
    size = rows * cols;

    if (rows != op2->rows || cols != op2->cols)
        return OP_UNMATCHED_SIZE;

    float *newarr; /* Used to store the intermediate result */

    /* If result is NULL */
    if (*result == NULL)
    {
        if ((*result = create_matrix(rows, cols)) == NULL) /* If failed to create a new matrix */
        {
            out_of_memory();
            return OUT_OF_MEMORY;
        }
        else
            newarr = (*result)->arr;
    }
    /* Else prepare the result array */
    else if ((newarr = (float *)aligned_alloc(32, size * sizeof(float))) == NULL)
    {
        out_of_memory();
        return OUT_OF_MEMORY;
    }

    /* Start addition */

    // todo: parallel optimization
    switch (code)
    {
    case ADD:
#pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
            newarr[i] = op1->arr[i] + op2->arr[i];
        break;
    case SUBTRACT:
#pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
            newarr[i] = op1->arr[i] - op2->arr[i];
        break;
    case MULTIPLY:
#pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
            newarr[i] = op1->arr[i] * op2->arr[i];
        break;
    case DIVIDE:
#pragma omp parallel for
        for (size_t i = 0; i < size; ++i)
            newarr[i] = op1->arr[i] / op2->arr[i];
        break;
    }

    /* Clean up */
    (*result)->rows = rows;
    (*result)->cols = cols;
    if ((*result)->arr != newarr)
    {
        free((*result)->arr);
        (*result)->arr = newarr;
    }

    return COMPLETED;
}

/* Add two matrices and store the result in a third matrix.
   The result matrix can refer to either op1 or op2 at the same time, but the pointer to the result matrix cannot be NULL.
   All operands, including the result, are supposed to be NULL or valid (i.e. created using function create_matrix()),
   so that an invalid pointer won't be dereferenced.
   If not, undefined behaviour will occur. If the result matrix isn't able to store the sum, i.e.,
   either it is NULL or the size doesn't match, the result matrix would be modified to match the need.

   If errors occurred during the operation (for example, operand size unmatches), do nothing on the result matrix.
   Returns the corresonding errno code upon failure. */
matrix_errno add_matrix(const matrix addend1, const matrix addend2, matrix *result)
{
    return _do_emma_on_matrices(addend1, addend2, result, ADD);
}

/* Subtract subtractor matrix from subtractend matrix and store the result in a third matrix.
   The result matrix can refer to either op1 or op2 at the same time, but the pointer to the result matrix cannot be NULL.
   All operands, including the result, are supposed to be NULL or valid (i.e. created using function create_matrix()),
   so that an invalid pointer won't be dereferenced.
   If not, undefined behaviour will occur. If the result matrix isn't able to store the sum, i.e.,
   either it is NULL or the size doesn't match, the result matrix would be modified to match the need.

   If errors occurred during the operation (for example, operand size unmatches), do nothing on the result matrix.
   Returns the corresonding errno code upon failure. */
matrix_errno subtract_matrix(const matrix subtrahend, const matrix subtractor, matrix *result)
{
    return _do_emma_on_matrices(subtrahend, subtractor, result, SUBTRACT);
}

/* This function multiplies two matrices plainly.

   Multiply two matrices and store the result in a third matrix.
   The result matrix can refer to either op1 or op2 at the same time, but the pointer to the result matrix cannot be NULL.
   All operands, including the result, are supposed to be NULL or valid (i.e. created using function create_matrix()), so that an invalid pointer won't be dereferenced.
   If not, undefined behaviour will occur. If the result matrix isn't able to store the sum, i.e., either it is NULL or the size doesn't match, the result matrix would be modified to match the need.

   If errors occurred during the operation (for example, operand size unmatches), do nothing on the result matrix.

   Returns the corresonding errno code upon failure. */
matrix_errno multiply_matrix_plain(const matrix op1, const matrix op2, matrix *result)
{
    if (op1 == NULL || op2 == NULL)
        return OP_NULL_PTR;

    size_t r_rows = op1->rows; /* number of rows in the result matrix */
    size_t r_cols = op2->cols; /* number of columns in the result matrix */

    if (op1->cols != op2->rows)
        return OP_UNMATCHED_SIZE;

    if (!_is_param_valid(r_rows, r_cols))
        return OP_EXCEEDED_SIZE;

    float *newarr; /* Used to store the intermediate result */

    /* If result is NULL */
    if (*result == NULL)
    {
        if ((*result = create_matrix(r_cols, r_rows)) == NULL) /* If failed to create a new matrix */
        {
            out_of_memory();
            return OUT_OF_MEMORY;
        }
        else
            newarr = (*result)->arr;
    }
    /* Else prepare the result array */
    else if ((newarr = (float *)aligned_alloc(32, (r_rows * r_cols) * sizeof(float))) == NULL)
    {
        out_of_memory();
        return OUT_OF_MEMORY;
    }

    /* Start multiplication */

    size_t c_cnt = op1->cols; /* Cycle count */

    // plain matrix multiplication

    for (size_t i = 0; i < op1->rows; ++i)
    {
        for (size_t j = 0; j < op2->cols; ++j)
        {
            float result = 0;
            for (size_t k = 0; k < op1->cols; ++k)
            {
                result += op1->arr[i * c_cnt + k] * op2->arr[k * op2->cols + j];
            }
            newarr[i * r_cols + j] = result;
        }
    }

    /* Clean up */
    (*result)->rows = r_cols;
    (*result)->cols = r_rows;
    if ((*result)->arr != newarr)
    {
        free((*result)->arr);
        (*result)->arr = newarr;
    }

    return COMPLETED;
}

/* This function multiplies two matrices by first transposing the righthand matrix, using OpenMP to parallelize computation, and then transposing the result.
   This function transposes the matrices to gain sequential access to elements.

   Multiply two matrices and store the result in a third matrix.
   The result matrix can refer to either op1 or op2 at the same time, but the pointer to the result matrix cannot be NULL.
   All operands, including the result, are supposed to be NULL or valid (i.e. created using function create_matrix()), so that an invalid pointer won't be dereferenced.
   If not, undefined behaviour will occur. If the result matrix isn't able to store the sum, i.e., either it is NULL or the size doesn't match, the result matrix would be modified to match the need.

   If errors occurred during the operation (for example, operand size unmatches), do nothing on the result matrix.

   Returns the corresonding errno code upon failure. */
matrix_errno multiply_matrix_ver_1(const matrix op1, const matrix op2, matrix *result)
{
    if (op1 == NULL || op2 == NULL)
        return OP_NULL_PTR;

    size_t r_rows = op1->rows; /* number of rows in the result matrix */
    size_t r_cols = op2->cols; /* number of columns in the result matrix */

    if (op1->cols != op2->rows)
        return OP_UNMATCHED_SIZE;

    if (!_is_param_valid(r_rows, r_cols))
        return OP_EXCEEDED_SIZE;

    float *newarr; /* Used to store the intermediate result */

    /* If result is NULL */
    if (*result == NULL)
    {
        if ((*result = create_matrix(r_cols, r_rows)) == NULL) /* If failed to create a new matrix */
        {
            out_of_memory();
            return OUT_OF_MEMORY;
        }
        else
            newarr = (*result)->arr;
    }
    /* Else prepare the result array */
    else if ((newarr = (float *)aligned_alloc(32, (r_rows * r_cols) * sizeof(float))) == NULL)
    {
        out_of_memory();
        return OUT_OF_MEMORY;
    }

    /* Start multiplication */

    matrix trans2 = NULL; /* Store the transposed op2 */
    matrix_errno tr_res;  /* Store the transpose result */

    if ((tr_res = transpose_matrix(op2, &trans2)) != COMPLETED)
    {
        if (newarr == (*result)->arr) /* Else the result matrix is newly created, delete the result. */
            delete_matrix(result);
        else /* If the original result matrix is not empty, only free newarr */
            free(newarr);
        return tr_res;
    }

    // implemented: transposed matrices (trans2, result) for faster access speed, loop designed to access elements sequentially
    const size_t blk_size = 64 / sizeof(float); // 64 == common cache line size
    const size_t ub1 = op1->rows;               // M
    const size_t ub2 = op2->cols;               // K
    const size_t ub3 = op1->cols;               // N
#pragma omp parallel for simd
    for (size_t m = 0; m < ub1; ++m)
    {
        size_t idx0 = m * r_cols;
        for (size_t k = 0; k < ub2; ++k)
        {
            size_t idx1 = m * ub3;
            size_t idx2 = k * ub3;
            float result0 = 0;
            for (size_t n = 0; n < ub3; ++n)
                result0 += op1->arr[idx1++] * trans2->arr[idx2++];
            newarr[idx0++] = result0;
        }
    }

    /* Clean up */
    delete_matrix(&trans2);
    (*result)->rows = r_cols;
    (*result)->cols = r_rows;
    if ((*result)->arr != newarr)
    {
        free((*result)->arr);
        (*result)->arr = newarr;
    }

    return COMPLETED;
}

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* Horizontally sum a __m256 which stores 8 packed 32-bit precision fp */
static inline float _mm256_hsum(__m256 a)
{
    __m256 t1 = _mm256_hadd_ps(a, a);
    __m256 t2 = _mm256_hadd_ps(t1, t1);
    __m128 t3 = _mm256_extractf128_ps(t2, 1);
    __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);
    return _mm_cvtss_f32(t4);
}

/* Multiplying two matrices A (m * n) and B (n * k) where B is **tranposed**, and store the result in C.
   A, B, C **must** be aligned on a 32-byte boundary. */
static void
_matrix_mul(float *A, float *B, float *C, int m, int n, int k)
{
    // implemented: transposed matrices (trans2, result) for faster access speed, loop designed to access elements sequentially
    const size_t blk_size = 64 / sizeof(float); // 64 == common cache line size

#pragma omp parallel for simd
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t u = 0; u < k; ++u)
        {
            float result = 0.0f;
//==============
// #ifndef __AVX2__
// #define __AVX2__
// #endif
//==============
#ifdef __AVX2__
            float rst[8] = {};
            __m256 r0 = _mm256_loadu_ps(rst);
            size_t j;
            for (j = 0; j < n - 7; j += 8)
            {
                __m256 u1 = _mm256_loadu_ps(A + i * n + j);
                __m256 u2 = _mm256_loadu_ps(B + u * n + j);
                __m256 r1 = _mm256_mul_ps(u1, u2);
                r0 = _mm256_add_ps(r0, r1);
            }
            if (j != n)
            {
                if (n >= 8)
                {
                    result += _mm256_hsum(r0);
                }
                float *tmp1 = (float *)aligned_alloc(32, 8 * sizeof(float));
                float *tmp2 = (float *)aligned_alloc(32, 8 * sizeof(float));
                memcpy(tmp1, A + i * n + j, (n - j) * sizeof(float));
                memcpy(tmp2, B + u * n + j, (n - j) * sizeof(float));
                __m256 u1 = _mm256_load_ps(tmp1);
                __m256 u2 = _mm256_load_ps(tmp2);
                __m256 res = _mm256_mul_ps(u1, u2);
                _mm256_store_ps(tmp1, res);
                for (int i = 0; i < n - j; ++i)
                    result += tmp1[i];
                free(tmp1);
                free(tmp2);
            }
#else
#pragma omp simd reduction(+ \
                           : result)
            for (size_t j = 0; j < n; ++j)
            {
                result += A[i * n + j] * B[u * n + j];
            }
#endif
            C[i * k + u] = result;
        }
    }
}

/* This function multiplies two matrices by first transposing the righthand matrix, using OpenMP and SIMD to parallelize computation, and then transposing the result.
   This function transposes the matrices to gain sequential access to elements.

   Multiply two matrices and store the result in a third matrix.
   The result matrix can refer to either op1 or op2 at the same time, but the pointer to the result matrix cannot be NULL.
   All operands, including the result, are supposed to be NULL or valid (i.e. created using function create_matrix()), so that an invalid pointer won't be dereferenced.
   If not, undefined behaviour will occur. If the result matrix isn't able to store the sum, i.e., either it is NULL or the size doesn't match, the result matrix would be modified to match the need.

   If errors occurred during the operation (for example, operand size unmatches), do nothing on the result matrix.

   Returns the corresonding errno code upon failure. */
matrix_errno
multiply_matrix_ver_2(const matrix op1, const matrix op2, matrix *result)
{
    if (op1 == NULL || op2 == NULL)
        return OP_NULL_PTR;

    size_t r_rows = op1->rows; /* number of rows in the result matrix */
    size_t r_cols = op2->cols; /* number of columns in the result matrix */

    if (op1->cols != op2->rows)
        return OP_UNMATCHED_SIZE;

    if (!_is_param_valid(r_rows, r_cols))
        return OP_EXCEEDED_SIZE;

    float *newarr; /* Used to store the intermediate result */

    /* If result is NULL */
    if (*result == NULL)
    {
        if ((*result = create_matrix(r_cols, r_rows)) == NULL) /* If failed to create a new matrix */
        {
            out_of_memory();
            return OUT_OF_MEMORY;
        }
        else
            newarr = (*result)->arr;
    }
    /* Else prepare the result array */
    else if ((newarr = (float *)aligned_alloc(32, (r_rows * r_cols) * sizeof(float))) == NULL)
    {
        out_of_memory();
        return OUT_OF_MEMORY;
    }

    /* Start multiplication */

    matrix trans2 = NULL; /* Store the transposed op2 */
    matrix_errno tr_res;  /* Store the transpose result */

    if ((tr_res = transpose_matrix(op2, &trans2)) != COMPLETED)
    {
        if (newarr == (*result)->arr) /* Else the result matrix is newly created, delete the result. */
            delete_matrix(result);
        else /* If the original result matrix is not empty, only free newarr */
            free(newarr);
        return tr_res;
    }

    // implemented: transposed matrices (trans2, result) for faster access speed, loop designed to access elements sequentially
    const size_t blk_size = 64 / sizeof(float); // 64 == common cache line size
    const size_t ub1 = op1->rows;               // M
    const size_t ub2 = op2->cols;               // K
    const size_t ub3 = op1->cols;               // N

    _matrix_mul(op1->arr, trans2->arr, newarr, op1->rows, op1->cols, op2->cols);

    /* Clean up */
    delete_matrix(&trans2);
    (*result)->rows = r_cols;
    (*result)->cols = r_rows;
    if ((*result)->arr != newarr)
    {
        free((*result)->arr);
        (*result)->arr = newarr;
    }

    return COMPLETED;
}

/* Perform EleMent-by-eleMent Arithmetic (EMMA) operation on a matrix and a scalar, then store the result in a second matrix.
   The result matrix can refer to src, but the pointer to the result matrix cannot be NULL.
   src and result are supposed to be NULL or valid (i.e. created using function create_matrix()).
   If not, undefined behaviour will occur. If the result matrix isn't able to store the sum, i.e.,
   either it is NULL or the size doesn't match, the result matrix would be modified to match the need.

   If errors occurred during the operation, do nothing on the result matrix.
   However, the user must handle the 0 divisor problem themselves.
   Returns the corresonding errno code upon failure. */
static matrix_errno
_do_emma_on_matrix_and_scalar(const matrix src, matrix *result, float val, op_code code)
{
    if (src == NULL)
        return OP_NULL_PTR;

    /* If the result matrix isn't the same as src matrix, copy the src. */
    if (*result != src)
    {
        matrix_errno tmp0;
        if ((tmp0 = copy_matrix(result, src)) != COMPLETED)
            return tmp0;
    }

    /* Now we operate on the result matrix. */
    switch (code)
    {
    case ADD:
#pragma omp parallel for
        for (size_t i = 0; i < (*result)->rows * (*result)->cols; ++i)
            (*result)->arr[i] += val;
        break;
    case SUBTRACT:
#pragma omp parallel for
        for (size_t i = 0; i < (*result)->rows * (*result)->cols; ++i)
            (*result)->arr[i] -= val;
        break;
    case MULTIPLY:
#pragma omp parallel for
        for (size_t i = 0; i < (*result)->rows * (*result)->cols; ++i)
            (*result)->arr[i] *= val;
        break;
    case DIVIDE:
#pragma omp parallel for
        for (size_t i = 0; i < (*result)->rows * (*result)->cols; ++i)
            (*result)->arr[i] /= val;
        break;
    }

    return COMPLETED;
}

/* Add a scalar to all elements in this matrix and store the result in a third matrix.
   The result matrix can refer to src, but the pointer to the result matrix cannot be NULL.
   src and result are supposed to be NULL or valid (i.e. created using function create_matrix()).
   If not, undefined behaviour will occur.

   If errors occurred during the operation, do nothing on the result matrix.
   Returns the corresonding errno code upon failure.
    */
matrix_errno add_scalar(const matrix src, matrix *result, float val)
{
    return _do_emma_on_matrix_and_scalar(src, result, val, ADD);
}

/* Subtract a scalar from all elements in this matrix and store the result in a third matrix.
   The result matrix can refer to src, but the pointer to the result matrix cannot be NULL.
   src and result are supposed to be NULL or valid (i.e. created using function create_matrix()).
   If not, undefined behaviour will occur.

   If errors occurred during the operation, do nothing on the result matrix.
   Returns the corresonding errno code upon failure.
    */
matrix_errno subtract_scalar(const matrix src, matrix *result, float val)
{
    return _do_emma_on_matrix_and_scalar(src, result, val, SUBTRACT);
}

/* Multiply every elements in this matrix with a scalar and store the result in a third matrix.
   The result matrix can refer to src, but the pointer to the result matrix cannot be NULL.
   src and result are supposed to be NULL or valid (i.e. created using function create_matrix()).
   If not, undefined behaviour will occur.

   If errors occurred during the operation, do nothing on the result matrix.
   Returns the corresonding errno code upon failure.
    */
matrix_errno multiply_scalar(const matrix src, matrix *result, float val)
{
    return _do_emma_on_matrix_and_scalar(src, result, val, MULTIPLY);
}

/* Find the maximum element in a valid matrix. src should never be NULL. */
float matrix_max(const matrix src)
{
    float maximum = FLT_MIN;
    for (size_t i = 0; i < src->rows * src->cols; ++i)
        if (src->arr[i] > maximum)
            maximum = src->arr[i];
    return maximum;
}

/* Find the minimum element in a valid matrix. src should never be NULL. */
float matrix_min(const matrix src)
{
    float minimum = FLT_MAX;
    for (size_t i = 0; i < src->rows * src->cols; ++i)
        if (src->arr[i] < minimum)
            minimum = src->arr[i];
    return minimum;
}

/* Read a matrix determined from a char array that can be either NULL
   or in a standard matrix form.
   If the result matrix cannot hold such, modify the result matrix to matche the need.

   Standard matrix form:
   First line, two positive integers, m, n, denoting the number of rows and columns, respectively.
   For the following m lines, each line contains n floating-point numbers, denoting the elements of each rows
   in that matrix. The line separator is not necessarily needed.

   If failed to reallocate space, first call out_of_memory, then nothing is done on result. */
matrix_errno read_matrix_from_string(char *s, matrix *result)
{
    if (s == NULL)
        return OP_NULL_PTR;

    size_t m, n;
    matrix tmp; /* Store the result matrix. */
    int bytes_read;

    sscanf(s, "%zu %zu%n", &m, &n, &bytes_read);
    s += bytes_read;

    if ((tmp = create_matrix(m, n)) == NULL)
    {
        out_of_memory();
        return OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
        {
            sscanf(s, "%f%n", tmp->arr + i * n + j, &bytes_read);
            s += bytes_read;
        }

    if (*result != NULL)
        delete_matrix(result);
    *result = tmp;

    return COMPLETED;
}

/* Read a matrix determined from a char array that can be either NULL
   or in a standard matrix form.
   If the result matrix cannot hold such, modify the result matrix to matche the need.

   Standard matrix form:
   First line, two positive integers, m, n, denoting the number of rows and columns, respectively.
   For the following m lines, each line contains n floating-point numbers, denoting the elements of each rows
   in that matrix. The line separator is not necessarily needed.

   If failed to reallocate space, first call out_of_memory, then nothing is done on result. */
matrix_errno read_matrix_from_stream(FILE *p, matrix *result)
{
    if (p == NULL)
        return OP_NULL_PTR;

    size_t m, n;
    matrix tmp; /* Store the result matrix. */

    if (fscanf(p, "%zu %zu", &m, &n) != 2)
        return OP_INVALID;

    if ((tmp = create_matrix(m, n)) == NULL)
    {
        out_of_memory();
        return OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            if (fscanf(p, "%f", tmp->arr + i * n + j) <= 0)
            {
                delete_matrix(&tmp);
                return OP_INVALID;
            }

    if (*result != NULL)
        delete_matrix(result);
    *result = tmp;

    return COMPLETED;
}

/* Print to the FILE stream the src matrix ended with a newline character.
   src must be either NULL or a valid matrix, i.e., created by create_matrix() function. */
matrix_errno print_matrix(FILE *p, const matrix src)
{
    if (src == NULL)
    {
        fprintf(p, "(NIL)\n");
        return OP_NULL_PTR;
    }

    fprintf(p, "%zu %zu\n", src->rows, src->cols);

    for (size_t i = 0; i < src->rows; ++i)
    {
        for (size_t j = 0; j < src->cols; ++j)
            fprintf(p, "%8.4g ", src->arr[i * (src->cols) + j]);
        fprintf(p, "\n");
    }
    fflush(p);

    return COMPLETED;
}
