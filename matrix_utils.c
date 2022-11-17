/* Source file for some matrix utils */

#include "matrix_utils.h"

#include <stdio.h>
#include <stdlib.h>

/* Global variables */
static void (*_handler)(void) = NULL;

/* Functions */

/* Initialize this matrix library using the corresponding handler
   to handle critical errors. */
void init_matrix_utils(void (*handler)(void))
{
    if (handler != NULL)
        _handler = handler;
}

/* When out_of_memory, handles the exception and possibly end the whole program. */
void out_of_memory(void)
{
    fprintf(stderr, "matrix library critical error: out of memory\n");
    if (_handler != NULL)
        (*_handler)();
}