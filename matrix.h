/* Header file for matrices */

#ifndef _MATRIX_H
#define _MATRIX_H

/* Header files */

#include <stddef.h>
#include <stdio.h>

/* Struct and enum declarations */

/* Pointer to a matrix_struct */
typedef struct matrix_struct *matrix;

/* Matrix struct for storing elements of a matrix */
typedef struct matrix_struct
{
    /* Properties of a matrix */
    size_t rows; /* Number of rows */
    size_t cols; /* Number of columns */
    size_t refs;    /* Reference to this matrix */

    /* Linear array for storing actual elements. Access offset to the actual element
       is (r * cols + c), where r is row index and c is column index, all starting from 0. */
    float *arr;  

    /* For reuse of structs */
    struct matrix_struct *next; /* When this matrix is recycled, points to the next recycled element in a linked list. If not, next has garbage value */

} matrix_struct;

typedef enum matrix_errno
{
    COMPLETED = 1,     /* The whole operation completed successfully. */
    OUT_OF_MEMORY,     /* Out of memory upon operations */
    OP_NULL_PTR,       /* Null pointer in required operands */
    OP_INVALID,        /* Empty operand */
    OP_UNMATCHED_SIZE, /* Unmatched size of operands */
    OP_EXCEEDED_SIZE,  /* Size exceeded on requirement */
    UNIMPLEMENTED      /* Unimplemented operation */
} matrix_errno;

/* Function prototypes */

matrix create_matrix(const size_t rows, const size_t cols);

void delete_matrix(matrix *m);

matrix_errno copy_matrix(matrix *dest, const matrix src);

matrix ref_matrix(const matrix m);

int test_equality(const matrix op1, const matrix op2, float ERR);

matrix_errno transpose_matrix(const matrix src, matrix *result);

matrix_errno add_matrix(const matrix addend1, const matrix addend2, matrix *result);

matrix_errno subtract_matrix(const matrix subtrahend, const matrix subtractor, matrix *result);

matrix_errno multiply_matrix(const matrix op1, const matrix op2, matrix *result);

matrix_errno add_scalar(const matrix src, matrix *result, float val);

matrix_errno subtract_scalar(const matrix src, matrix *result, float val);

matrix_errno multiply_scalar(const matrix src, matrix *result, float val);

float matrix_max(const matrix src);

float matrix_min(const matrix src);

matrix_errno read_matrix_from_string(char *s, matrix *result);

matrix_errno read_matrix_from_stream(FILE *p, matrix *result);

matrix_errno print_matrix(FILE *p, const matrix src);

#endif