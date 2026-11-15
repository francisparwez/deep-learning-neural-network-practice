
import torch
import numpy as np

# ============================================================
# QUESTION 1 — INSPECT AND MANIPULATE TENSORS (DL)
# Objective: Understand tensor structures, dimensions, shapes,
# data types, and basic tensor operations.
# ============================================================


# 1. CREATE 1D, 2D, AND 3D TENSORS

tensor1D = torch.tensor([10, 20, 30, 40, 50])

tensor2D = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

tensor3D = torch.tensor([
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
    [
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]
    ]
])


# 2. INSPECT TENSOR PROPERTIES

print("\n--- TENSOR INSPECTION ---")

tensors = {
    "1D Tensor": tensor1D,
    "2D Tensor": tensor2D,
    "3D Tensor": tensor3D
}

for name, tensor in tensors.items():
    print(f"\n{name}")
    print("Tensor:\n", tensor)
    print("Shape:", tensor.shape)
    print("Dimensions:", tensor.ndim)
    print("Data Type:", tensor.dtype)
    print("Total Elements:", tensor.numel())


# 3. RESHAPING TENSORS

print("\n--- RESHAPING ---")

reshaped_2D = tensor2D.reshape(4, 3)

print("Original Shape:", tensor2D.shape)
print("Reshaped Tensor:\n", reshaped_2D)
print("New Shape:", reshaped_2D.shape)
print("Total Elements:", reshaped_2D.numel())


# 4. INDEXING AND SLICING

print("\n--- INDEXING AND SLICING ---")

print("Element at row 1, column 2:", tensor2D[1, 2])

print("First Row:", tensor2D[0, :])

print("Second Column:", tensor2D[:, 1])

print("Submatrix (first two rows, columns 2-4):\n",
      tensor2D[0:2, 1:4])


# 5. DATA TYPE CONVERSION

print("\n--- DATA TYPE CONVERSION ---")

tensor_float = tensor1D.to(torch.float32)

print("Original Tensor:", tensor1D)
print("Original Data Type:", tensor1D.dtype)

print("Converted Tensor:", tensor_float)
print("Converted Data Type:", tensor_float.dtype)


# 6. ELEMENT-WISE ADDITION AND MULTIPLICATION

print("\n--- ELEMENT-WISE OPERATIONS ---")

tensorA = torch.tensor([1, 2, 3])
tensorB = torch.tensor([4, 5, 6])

addition_result = tensorA + tensorB
multiplication_result = tensorA * tensorB

print("Tensor A:", tensorA)
print("Tensor B:", tensorB)

print("Addition:", addition_result)
print("Multiplication:", multiplication_result)


# 7. MATRIX MULTIPLICATION

print("\n--- MATRIX MULTIPLICATION ---")

matrixA = torch.tensor([
    [1, 2],
    [3, 4]
])

matrixB = torch.tensor([
    [5, 6],
    [7, 8]
])

matrix_result = torch.matmul(matrixA, matrixB)

print("Matrix A:\n", matrixA)
print("Matrix B:\n", matrixB)
print("Matrix Multiplication Result:\n", matrix_result)


# 8. TENSOR AGGREGATION

print("\n--- TENSOR AGGREGATION ---")

aggregation_tensor = torch.tensor([10, 20, 30, 40, 50])

print("Tensor:", aggregation_tensor)
print("Sum:", torch.sum(aggregation_tensor))
print("Mean:", torch.mean(aggregation_tensor.float()))
print("Maximum:", torch.max(aggregation_tensor))
print("Minimum:", torch.min(aggregation_tensor))


# 9. NUMPY VS PYTORCH

print("\n--- NUMPY VS PYTORCH ---")

# NumPy arrays
numpy_tensorA = np.array([1, 2, 3])
numpy_tensorB = np.array([4, 5, 6])

numpy_addition = numpy_tensorA + numpy_tensorB
numpy_multiplication = numpy_tensorA * numpy_tensorB

# PyTorch tensors
torch_tensorA = torch.tensor([1, 2, 3])
torch_tensorB = torch.tensor([4, 5, 6])

torch_addition = torch_tensorA + torch_tensorB
torch_multiplication = torch_tensorA * torch_tensorB

print("\nNumPy Addition:", numpy_addition)
print("PyTorch Addition:", torch_addition)

print("\nNumPy Multiplication:", numpy_multiplication)
print("PyTorch Multiplication:", torch_multiplication)


# 10. PYTORCH AUTOGRAD — AUTOMATIC DIFFERENTIATION

print("\n--- PYTORCH AUTOGRAD ---")

x = torch.tensor(3.0, requires_grad=True)

y = x ** 2

y.backward()

print("Input x:", x)
print("Function y = x²:", y)
print("Gradient dy/dx:", x.grad)