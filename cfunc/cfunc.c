#include <Python.h>
#include <numpy/arrayobject.h>

// Define the function name and docstring
static char func_docstring[] = "Return the game status of a connect-5 board.";
static char func_name[] = "cgetGameEnded";

// Declare the function prototype
static PyObject *cgetGameEnded(PyObject *self, PyObject *args);

// Define the methods table
static PyMethodDef module_methods[] = {
    {func_name, cgetGameEnded, METH_VARARGS, func_docstring},
    {NULL, NULL, 0, NULL}
};

// Define the module name and docstring
static char module_docstring[] = "A module that provides a C function for alpha zero programm.";
static char module_name[] = "cfunc";

// Initialize the module
PyMODINIT_FUNC PyInit_cfunc(void) {
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        module_name,
        module_docstring,
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };
    module = PyModule_Create(&moduledef);
    if (module == NULL)
        return NULL;

    // Import numpy and initialize its C API
    import_array();

    return module;
}

static PyObject *cgetGameEnded(PyObject *self, PyObject *args) { 
    // Parse the input arguments 
    PyArrayObject *board; 
    if (!PyArg_ParseTuple(args, "O", &board)) 
        return NULL;

    // Check the input array type and dimensions
    if (PyArray_TYPE(board) != NPY_INT)
    {
        PyErr_SetString(PyExc_TypeError, "board must be an integer array");
        return NULL;
    }
    if (PyArray_NDIM(board) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "board must be a 2D array");
        return NULL;
    }
    int n = PyArray_DIM(board, 0);
    if (n != PyArray_DIM(board, 1))
    {
        PyErr_SetString(PyExc_ValueError, "board must be a square array");
        return NULL;
    }

    // Get the pointer to the array data
    int *data = (int *)PyArray_DATA(board);

    // Define the number of stones in a row to win
    int n_in_row = 5;

    // Loop over the board and check for winning conditions
    for (int w = 0; w < n; w++)
    {
        for (int h = 0; h < n; h++)
        {
            int index = w * n + h; // Convert 2D index to 1D index
            int stone = data[index]; // Get the value of the current position
            if (stone == 0)
                continue; // Skip empty positions
            if (w <= n - n_in_row && stone == data[index + n] && stone == data[index + 2 * n] &&
                stone == data[index + 3 * n] && stone == data[index + 4 * n])
                return PyLong_FromLong(stone); // Return the winner if there is a vertical line
            if (h <= n - n_in_row && stone == data[index + 1] && stone == data[index + 2] &&
                stone == data[index + 3] && stone == data[index + 4])
                return PyLong_FromLong(stone); // Return the winner if there is a horizontal line
            if (w <= n - n_in_row && h <= n - n_in_row && stone == data[index + n + 1] && stone == data[index + 2 * (n + 1)] &&
                stone == data[index + 3 * (n + 1)] && stone == data[index + 4 * (n + 1)])
                return PyLong_FromLong(stone); // Return the winner if there is a diagonal line (\)
            if (w <= n - n_in_row && h >= n_in_row - 1 && stone == data[index + n - 1] && stone == data[index + 2 * (n - 1)] &&
                stone == data[index + 3 * (n - 1)] && stone == data[index + 4 * (n - 1)])
                return PyLong_FromLong(stone); // Return the winner if there is a diagonal line (/)
        }
    }

    // Check if there are any legal moves left
    for (int i = 0; i < n * n; i++)
    {
        if (data[i] == 0)
            return PyLong_FromLong(0); // Return zero if the game is not ended
    }

    // Return a small positive value if the game is a draw
    return PyFloat_FromDouble(1e-4);
}