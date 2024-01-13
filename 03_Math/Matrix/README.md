# **Math**

## **2. Matrix**

### **2.1. Matrix definition**

- Matrix is a rectangular array of numbers. It is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{m1} & a_{m2} & ... & a_{mn} \end{bmatrix}
```

- m and n is the number of rows and columns of matrix A.
- $`a_{ij}`$ is the element of matrix A at row i and column j.
- $`a_{i}`$ is the $`i^{th}`$ row of matrix A.
- $`a_{j}`$ is the $`j^{th}`$ column of matrix A.

### **2.2. Matrix addition and subtraction**

- Matrix addition and subtraction is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{m1} & a_{m2} & ... & a_{mn} \end{bmatrix}, B = \begin{bmatrix} b_{11} & b_{12} & ... & b_{1n}\\ b_{21} & b_{22} & ... & b_{2n}\\ ... & ... & ... & ...\\ b_{m1} & b_{m2} & ... & b_{mn} \end{bmatrix} \Rightarrow A \pm B = \begin{bmatrix} a_{11} \pm b_{11} & a_{12} \pm b_{12} & ... & a_{1n} \pm b_{1n}\\ a_{21} \pm b_{21} & a_{22} \pm b_{22} & ... & a_{2n} \pm b_{2n}\\ ... & ... & ... & ...\\ a_{m1} \pm b_{m1} & a_{m2} \pm b_{m2} & ... & a_{mn} \pm b_{mn} \end{bmatrix}
```

- Note that two matrices can be added or subtracted if and only if they have the same size.

### **2.3. Matrix multiplication**

- Matrix multiplication is the product of two matrices. It is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{m1} & a_{m2} & ... & a_{mn} \end{bmatrix}, B = \begin{bmatrix} b_{11} & b_{12} & ... & b_{1p}\\ b_{21} & b_{22} & ... & b_{2p}\\ ... & ... & ... & ...\\ b_{n1} & b_{n2} & ... & b_{np} \end{bmatrix}
```

```math
\Rightarrow AB = \begin{bmatrix} a_{11}b_{11} + a_{12}b_{21} + ... + a_{1n}b_{n1} & a_{11}b_{12} + a_{12}b_{22} + ... + a_{1n}b_{n2} & ... & a_{11}b_{1p} + a_{12}b_{2p} + ... + a_{1n}b_{np}\\ a_{21}b_{11} + a_{22}b_{21} + ... + a_{2n}b_{n1} & a_{21}b_{12} + a_{22}b_{22} + ... + a_{2n}b_{n2} & ... & a_{21}b_{1p} + a_{22}b_{2p} + ... + a_{2n}b_{np}\\ ... & ... & ... & ...\\ a_{m1}b_{11} + a_{m2}b_{21} + ... + a_{mn}b_{n1} & a_{m1}b_{12} + a_{m2}b_{22} + ... + a_{mn}b_{n2} & ... & a_{m1}b_{1p} + a_{m2}b_{2p} + ... + a_{mn}b_{np} \end{bmatrix}
```

- Note that two matrices can be multiplied if and only if the number of columns of the first matrix is equal to the number of rows of the second matrix.
- Matrix A is a $`m \times n`$ matrix and matrix B is a $`n \times p`$ matrix. So that the result of matrix multiplication is a $`m \times p`$ matrix.
- Note that matrix multiplication is not commutative. That is, $`AB \neq BA`$.

### **2.4. Matrix transpose**

- Matrix transpose is the operation that flips a matrix over its diagonal. It is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{m1} & a_{m2} & ... & a_{mn} \end{bmatrix} \Rightarrow A^{T} = \begin{bmatrix} a_{11} & a_{21} & ... & a_{m1}\\ a_{12} & a_{22} & ... & a_{m2}\\ ... & ... & ... & ...\\ a_{1n} & a_{2n} & ... & a_{mn} \end{bmatrix}
```

- For simpler, the transpose of a matrix is a matrix whose rows are the columns of the original matrix and whose columns are the rows of the original matrix.
- Matrix A is a $`m \times n`$ matrix so matrix $`A^{T}`$ is a $`n \times m`$ matrix.

### **2.5. Identity matrix**

- Identity matrix is a square matrix with 1s on the main diagonal and 0s elsewhere. It is represented by the following formula:

```math
I_{n} = \begin{bmatrix} 1 & 0 & ... & 0\\ 0 & 1 & ... & 0\\ ... & ... & ... & ...\\ 0 & 0 & ... & 1 \end{bmatrix}
```

- $`I_{n}`$ is a $`n \times n`$ identity matrix.

### **2.6. Triangular matrix**

- Triangular matrix is a square matrix in which all the elements above or below the main diagonal are zero. It is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ 0 & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ 0 & 0 & ... & a_{nn} \end{bmatrix}
```

- If all the elements above the main diagonal are zero, it is called lower triangular matrix.
- If all the elements below the main diagonal are zero, it is called upper triangular matrix.

### **2.7. Diagonal matrix**

- Diagonal matrix is a square matrix in which all the elements except the main diagonal are zero. It is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & 0 & ... & 0\\ 0 & a_{22} & ... & 0\\ ... & ... & ... & ...\\ 0 & 0 & ... & a_{nn} \end{bmatrix}
```

### **2.8. Symmetric matrix**

- Symmetric matrix is a square matrix that is equal to its transpose. It is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{12} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{1n} & a_{2n} & ... & a_{nn} \end{bmatrix} \Rightarrow A^{T} = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{12} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{1n} & a_{2n} & ... & a_{nn} \end{bmatrix}
```

### **2.9. Elementary method**

- Elementary method is a method that is used to transform a matrix into a simpler form. It is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{m1} & a_{m2} & ... & a_{mn} \end{bmatrix} \Rightarrow \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ 0 & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ 0 & 0 & ... & a_{mn} \end{bmatrix}
```

- Some elementary methods are:
  - Interchange two rows.
  - Multiply a row by a nonzero constant.
  - Add a multiple of one row to another row.

- Elementary method does not change the determinant of a matrix.

### **2.10. Rank of a matrix**

- Rank of a matrix is the maximum number of linearly independent rows or columns of a matrix. It is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{m1} & a_{m2} & ... & a_{mn} \end{bmatrix} \Rightarrow rank(A) = r
```

- r is the rank of matrix A. $`r \leq min(m, n)`$.
- To find the rank of a matrix, use elementary method to transform the matrix into a simpler form. The rank of the simpler form is the rank of the original matrix. The simpler form is a matrix that has all zero elements below the main diagonal. The number of nonzero rows is the rank of the matrix.

- For example:

```math
A = \begin{bmatrix} 1 & 2 & 3\\ 2 & 4 & 6\\ 3 & 6 & 9 \end{bmatrix} \Rightarrow \begin{bmatrix} 1 & 2 & 3\\ 0 & 0 & 0\\ 0 & 0 & 0 \end{bmatrix} \Rightarrow rank(A) = 1
```

### **2.11. Determinant of a matrix**

- Determinant of a matrix is a scalar value that is a function of the entries of a square matrix. It is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{n1} & a_{n2} & ... & a_{nn} \end{bmatrix} \Rightarrow det(A) = |A| = \begin{vmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{n1} & a_{n2} & ... & a_{nn} \end{vmatrix}
```

- $`|A|`$ is the determinant of matrix A.

#### **2.11.1. Determinant of a 2x2 matrix**

- Determinant of a 2x2 matrix is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12}\\ a_{21} & a_{22} \end{bmatrix} \Rightarrow det(A) = |A| = \begin{vmatrix} a_{11} & a_{12}\\ a_{21} & a_{22} \end{vmatrix} = a_{11}a_{22} - a_{12}a_{21}
```

#### **2.11.2. Determinant of a 3x3 matrix**

- Determinant of a 3x3 matrix is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & a_{13}\\ a_{21} & a_{22} & a_{23}\\ a_{31} & a_{32} & a_{33} \end{bmatrix} \Rightarrow det(A) = |A| = \begin{vmatrix} a_{11} & a_{12} & a_{13}\\ a_{21} & a_{22} & a_{23}\\ a_{31} & a_{32} & a_{33} \end{vmatrix} = a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} - a_{13}a_{22}a_{31} - a_{12}a_{21}a_{33} - a_{11}a_{23}a_{32}
```

- Sarrus rule is a method to find the determinant of a 3x3 matrix. It is represented by the following formula:

![Sarrus rule](https://d223we85878hn.cloudfront.net/ce889b0b-ae84-4d90-a07d-744772a6651a_640w.jpeg)

#### **2.11.3. Determinant of a general matrix**

- Determinant of a general matrix is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{n1} & a_{n2} & ... & a_{nn} \end{bmatrix} \Rightarrow det(A) = |A| = \begin{vmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{n1} & a_{n2} & ... & a_{nn} \end{vmatrix} = \sum_{j=1}^{n}(-1)^{i+j}a_{ij}M_{ij}
```

- $`M_{ij}`$ is the minor of element $`a_{ij}`$.
- $`M_{ij}`$ is the determinant of the matrix that is obtained by deleting the $`i^{th}`$ row and $`j^{th}`$ column of matrix A.

- The other way to find the determinant of a general matrix is to use elementary method to transform the matrix into a simpler form. The determinant of the simpler form is the determinant of the original matrix. The simpler form is a matrix that has all zero elements below the main diagonal. The determinant of the simpler form is the product of the diagonal elements.

- For example of upper triangular matrix:

```math
A = \begin{bmatrix} a_{11} & a_{12} & a_{13}\\ 0 & a_{22} & a_{23}\\ 0 & 0 & a_{33} \end{bmatrix} \Rightarrow det(A) = |A| = \begin{vmatrix} a_{11} & a_{12} & a_{13}\\ 0 & a_{22} & a_{23}\\ 0 & 0 & a_{33} \end{vmatrix} = a_{11}a_{22}a_{33}
```

- For example of diagonal matrix:

```math
A = \begin{bmatrix} a_{11} & 0 & 0\\ 0 & a_{22} & 0\\ 0 & 0 & a_{33} \end{bmatrix} \Rightarrow det(A) = |A| = \begin{vmatrix} a_{11} & 0 & 0\\ 0 & a_{22} & 0\\ 0 & 0 & a_{33} \end{vmatrix} = a_{11}a_{22}a_{33}
```

#### **2.11.4. Properties of determinant**

- $`|A| = |A^T|`$
- $`|AB| = |A||B|`$
- $`|A^{-1}| = \frac{1}{|A|}`$
- $`|A| = 0`$ if and only if A is singular
- $`|A| \neq 0`$ if and only if A is nonsingular

### **2.12. Inverse of a matrix**

- Inverse of a matrix is a matrix that when multiplied by the original matrix results in the identity matrix. It is represented by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{n1} & a_{n2} & ... & a_{nn} \end{bmatrix} \Rightarrow A^{-1} = \begin{bmatrix} b_{11} & b_{12} & ... & b_{1n}\\ b_{21} & b_{22} & ... & b_{2n}\\ ... & ... & ... & ...\\ b_{n1} & b_{n2} & ... & b_{nn} \end{bmatrix} \Rightarrow AA^{-1} = A^{-1}A = I = \begin{bmatrix} 1 & 0 & ... & 0\\ 0 & 1 & ... & 0\\ ... & ... & ... & ...\\ 0 & 0 & ... & 1 \end{bmatrix}
```

- The inverse of a matrix can be found by the following formula:

```math
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n}\\ a_{21} & a_{22} & ... & a_{2n}\\ ... & ... & ... & ...\\ a_{n1} & a_{n2} & ... & a_{nn} \end{bmatrix} \Rightarrow A^{-1} = \frac{1}{|A|} \begin{bmatrix} A_{11} & A_{21} & ... & A_{n1}\\ A_{12} & A_{22} & ... & A_{n2}\\ ... & ... & ... & ...\\ A_{1n} & A_{2n} & ... & A_{nn} \end{bmatrix}^T
```

- $`A_{ij}`$ is the cofactor of element $`a_{ij}`$.
- $`A_{ij} = (-1)^{i+j}M_{ij}`$.
- $`M_{ij}`$ is the minor of element $`a_{ij}`$. $`M_{ij}`$ is the determinant of the matrix that is obtained by deleting the $`i^{th}`$ row and $`j^{th}`$ column of matrix A.

- For 2x2 matrix:

```math
A = \begin{bmatrix} a_{11} & a_{12}\\ a_{21} & a_{22} \end{bmatrix} \Rightarrow A^{-1} = \frac{1}{|A|} \begin{bmatrix} A_{11} & A_{21}\\ A_{12} & A_{22} \end{bmatrix}^T = \frac{1}{a_{11}a_{22} - a_{12}a_{21}} \begin{bmatrix} a_{22} & -a_{12}\\ -a_{21} & a_{11} \end{bmatrix}
```

- **Properties of inverse of a matrix**
  - $`(A^{-1})^{-1} = A`$
  - $`(AB)^{-1} = B^{-1}A^{-1}`$
  - $`(A^T)^{-1} = (A^{-1})^T`$

### **2.13. Matrix transform vector**

- A matrix can be used to transform a vector.
- It can help to change from original basis to new basis. The columns of the matrix are the new basis vectors in the original coordinate system. So:
  
```math
Br' = r \Rightarrow B^{-1}Br' = B^{-1}r \Rightarrow r' = B^{-1}r
```

- If a matrix A is orthonormal, then $`A^{-1} = A^T`$.
- Orthonormal matrix is a matrix whose columns are of unit length and orthogonal to each other.

#### **2.13.1. Example**

- A matrix can be used to change the basis of a vector. For example, the following matrix: $` A = \begin{bmatrix} 1 & 2\\ 3 & 4 \end{bmatrix} `$ can be used to change the basis of the vector $`\begin{bmatrix} 1\\ 0 \end{bmatrix}`$ from the standard basis to the basis $`\begin{bmatrix} 1\\ 1 \end{bmatrix}, \begin{bmatrix} 1\\ -1 \end{bmatrix}`$:

```math
A\begin{bmatrix} 1\\ 0 \end{bmatrix} = \begin{bmatrix} 1 & 2\\ 3 & 4 \end{bmatrix}\begin{bmatrix} 1\\ 0 \end{bmatrix} = \begin{bmatrix} 1\times1 + 2\times0\\ 3\times1 + 4\times0 \end{bmatrix} = \begin{bmatrix} 1\\ 3 \end{bmatrix}
```

#### **2.13.2. Matrix Gram-Schmidt process**

- The Gram-Schmidt process is a method for orthonormalizing a set of vectors.
- Start with n linearly independent vectors $` v = {v_1, v_2, ..., v_n} `$. Then the following vectors are orthonormalized:

```math
e_1 = \frac{v_1}{|v_1|}\\
u_2 = v_2 - \frac{v_2 \cdot e_1}{e_1 \cdot e_1}e_1 \Rightarrow e_2 = \frac{u_2}{|u_2|}\\
u_3 = v_3 - \frac{v_3 \cdot e_1}{e_1 \cdot e_1}e_1 - \frac{v_3 \cdot e_2}{e_2 \cdot e_2}e_2 \Rightarrow e_3 = \frac{u_3}{|u_3|}\\
```

- ... and so on for $`u_4, u_5..., u_n`$.

### **2.14. Eigenvalues and eigenvectors**

- The eigenvalues and eigenvectors of a matrix A are the scalars $`\lambda`$ and vectors $`v`$ that satisfy the following equation:

```math
Av = \lambda v
```

- The eigenvalues of a matrix A can be found by solving the following equation:

```math
\det(A - \lambda I) = 0
```

- The eigenvectors of a matrix A can be found by solving the following equation:

```math
(A - \lambda I)v = 0
```

- From the eigenvalues and eigenvectors of a matrix A, the following matrix can be constructed:

```math
A = V\Lambda V^{-1}
```

- Where V is a matrix whose columns are the eigenvectors of A and $`\lambda`$ is a diagonal matrix whose diagonal elements are the eigenvalues of A.

#### **2.14.1. Chaging basis using eigen**

- We can use the eigenvalues and eigenvectors of a matrix A to change the basis of a vector x from the standard basis to the eigenbasis of A:

```math
x' = V\Lambda V^{-1}x
```

#### **2.14.2. Matrix diagonalization**

- Matrix A is diagonalizable if it can be written in the following form:

```math
A = V\Lambda V^{-1}
```

- We can use the eigenvalues and eigenvectors of a matrix to calculate the $`n^{th}`$ power of the matrix:

```math
A^n = V\Lambda^n V^{-1}
```

- We can use the above equation to transform a vector x from the standard basis to the eigenbasis n times.
