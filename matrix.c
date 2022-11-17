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

static matrix _recycled_list = NULL; /* Linked list of recycled matrices. Those recycled matrices have garbage values and should NEVER be used. */

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

    if (_recycled_list != NULL) /* Reuse existing one first. */
    {
        result = _recycled_list;
        _recycled_list = _recycled_list->next;
    }
    else
    {
        result = (matrix)malloc(sizeof(matrix_struct));
        if (result == NULL)
        {
            out_of_memory();
            return NULL;
        }
    }
    arr = (float *)malloc(rows * cols * sizeof(float));
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
    result->next = NULL;

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
        (*m)->next = _recycled_list;
        _recycled_list = *m;
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
        if ((newarr = (float *)malloc(target_size * sizeof(float))) == NULL)
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
matrix ref_matrix(const matrix m)
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
    else if ((newarr = (float *)malloc((r_rows * r_cols) * sizeof(float))) == NULL)
    {
        out_of_memory();
        return OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < s_rows; ++i)
        for (size_t j = 0; j < s_cols; ++j)
            newarr[j * r_cols + i] = src->arr[i * s_cols + j];

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
    else if ((newarr = (float *)malloc(size * sizeof(float))) == NULL)
    {
        out_of_memory();
        return OUT_OF_MEMORY;
    }

    /* Start addition */

    // todo: parallel optimization
    switch (code)
    {
    case ADD:
        for (size_t i = 0; i < size; ++i)
            newarr[i] = op1->arr[i] + op2->arr[i];
        break;
    case SUBTRACT:
        for (size_t i = 0; i < size; ++i)
            newarr[i] = op1->arr[i] - op2->arr[i];
        break;
    case MULTIPLY:
        for (size_t i = 0; i < size; ++i)
            newarr[i] = op1->arr[i] * op2->arr[i];
        break;
    case DIVIDE:
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

/* Multiply two matrices and store the result in a third matrix.
   The result matrix can refer to either op1 or op2 at the same time, but the pointer to the result matrix cannot be NULL.
   All operands, including the result, are supposed to be NULL or valid (i.e. created using function create_matrix()),
   so that an invalid pointer won't be dereferenced.
   If not, undefined behaviour will occur. If the result matrix isn't able to store the sum, i.e.,
   either it is NULL or the size doesn't match, the result matrix would be modified to match the need.

   If errors occurred during the operation (for example, operand size unmatches), do nothing on the result matrix.
   Returns the corresonding errno code upon failure. */
matrix_errno multiply_matrix(const matrix op1, const matrix op2, matrix *result)
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
    else if ((newarr = (float *)malloc((r_rows * r_cols) * sizeof(float))) == NULL)
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

    size_t c_cnt = op1->cols; /* Cycle count */

    // todo: parallel optimization
    // implemented: transposed matrices (trans2, result) for faster access speed, loop designed to access elements linearly
#pragma omp parallel for
    for (size_t k = 0; k < op2->cols; ++k)
    {
#pragma omp parallel for
        for (size_t m = 0; m < op1->rows; ++m)
        {
            float result0 = 0;
            for (size_t n = 0; n < c_cnt; ++n)
            {
                result0 += op1->arr[m * c_cnt + n] * trans2->arr[k * c_cnt + n];
            }
            newarr[k * op1->rows + m] = result0;
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
    transpose_matrix(*result, result); /* Transpose the matrix since it's calculated in transposed form */

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
        for (size_t i = 0; i < (*result)->rows * (*result)->cols; ++i)
            (*result)->arr[i] += val;
        break;
    case SUBTRACT:
        for (size_t i = 0; i < (*result)->rows * (*result)->cols; ++i)
            (*result)->arr[i] -= val;
        break;
    case MULTIPLY:
        for (size_t i = 0; i < (*result)->rows * (*result)->cols; ++i)
            (*result)->arr[i] *= val;
        break;
    case DIVIDE:
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
