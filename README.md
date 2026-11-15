# 20 Deep Learning & 20 Neural Network Practice Questions

## Progress Overview

- **Beginner Level:** Question 1 completed (of 20).
- **Intermediate Level:** Not started.
- **Completed Exercises:** 1/40.

## Question 1 — Inspect and Manipulate Tensors (DL)

**Objective:** Understand tensor structures, dimensions, shapes, data types, and basic tensor operations using PyTorch, with a comparison of equivalent NumPy operations.

### Implementation

The solution is implemented in `01_inspect__manipulate_tensors.py` using **PyTorch** and **NumPy**. It covers:

1. **Tensor creation:** Creates 1D, 2D, and 3D PyTorch tensors, including a 3 × 4 matrix and a batch of two 3 × 3 matrices.
2. **Tensor inspection:** Displays each tensor's values, shape, number of dimensions, data type, and total number of elements using `shape`, `ndim`, `dtype`, and `numel()`.
3. **Reshaping:** Reshapes the 3 × 4 tensor into a 4 × 3 tensor while preserving its 12 elements.
4. **Indexing and slicing:** Retrieves an individual element, the first row, the second column, and a selected submatrix from the 2D tensor.
5. **Data type conversion:** Converts an integer tensor to `torch.float32` using `.to()`.
6. **Element-wise operations:** Performs addition and multiplication on compatible PyTorch tensors.
7. **Matrix multiplication:** Uses `torch.matmul()` to multiply two 2 × 2 matrices.
8. **Tensor aggregation:** Calculates the sum, mean, maximum, and minimum of a tensor.
9. **NumPy vs. PyTorch:** Performs equivalent element-wise addition and multiplication with NumPy arrays and PyTorch tensors and prints the results for comparison.
10. **Automatic differentiation:** Introduces PyTorch Autograd by calculating the gradient of `y = x²` at `x = 3` using `requires_grad=True` and `backward()`.

### Key Concepts Practiced

- Tensor rank, shape, data types, and element counts.
- Reshaping, indexing, and slicing.
- Numeric type conversion.
- Element-wise arithmetic versus matrix multiplication.
- Basic aggregation operations.
- Similarities in NumPy and PyTorch array/tensor arithmetic.
- Automatic differentiation and gradient calculation with PyTorch Autograd.

### Technologies Used

- Python
- PyTorch
- NumPy

### Run the Script

```bash
python 01_inspect__manipulate_tensors.py
```

Ensure PyTorch and NumPy are installed in your Python environment before running the script.
