# **Math**

## **1. Vector**

### **1.1. Vector definition**

- Vector is a quantity that has both magnitude and direction.
- Vector is represented by an arrow. The length of the arrow represents the magnitude of the vector and the direction of the arrow represents the direction of the vector.

### **1.2. Vector addition/subtraction**

- Vector addition is the operation of adding two or more vectors together into a vector sum.
- Vector addition is commutative and associative.
- Vector addition is represented by the following formula:

```math
v_{1} = \begin{bmatrix}
x_{1}\\
y_{1}
\end{bmatrix},
v_{2} = \begin{bmatrix}
x_{2}\\
y_{2}
\end{bmatrix}
\Rightarrow
v_{1} \pm v_{2}= \begin{bmatrix}
x_{1} \pm x_{2}\\
y_{1} \pm y_{2}
\end{bmatrix}
```

### **1.3. Vector multiplication**

- Vector multiplication is the operation of multiplying a vector by a scalar. The result is a vector.
- Vector multiplication is represented by the following formula:

```math
v = \begin{bmatrix}
x\\
y
\end{bmatrix},
k \in \mathbb{R}
\Rightarrow
kv = \begin{bmatrix}
kx\\
ky
\end{bmatrix}
```

### **1.4. Modulus of a vector**

- Modulus of a vector is the magnitude of the vector. It is represented by the length of the arrow representing the vector.
- Modulus of a vector is represented by the following formula:

```math
v = \begin{bmatrix}
v_{1}\\
v_{2}\\
...\\
v_{n}
\end{bmatrix}
\Rightarrow
\lVert v \rVert = \sqrt{v_{1}^{2} + v_{2}^{2} + ... + v_{n}^{2}} = num \in \mathbb{R}
```

### **1.5. Vector dot product**

- Vector dot product is the operation of multiplying the corresponding entries of the two vectors and then summing the products. The result is a scalar.
- Vector dot product is commutative and distributive.
- Vector dot product is represented by the following formula:

```math
v = \begin{bmatrix}
v_{1}\\
v_{2}\\
...\\
v_{n}
\end{bmatrix},
u = \begin{bmatrix}
u_{1}\\
u_{2}\\
...\\
u_{n}
\end{bmatrix}
\Rightarrow
v\cdot u = v_{1} \cdot u_{1} + v_{2} \cdot u_{2} + ... + v_{n} \cdot u_{n} = num \in \mathbb{R}
```

### **1.6. Vector attributes**

- Associative property of vector addition:
  - $`(u + v) + w = u + (v + w)`$
  - $`(u.v).w = u.(v.w)`$
- Commutative property of vector addition:
  - $`u + v = v + u`$
  - $`u.v = v.u`$
- Distributive property of vector addition:
  - $`w(u + v) = wu + wv`$

### **1.7. Vector cosine**

- Vector cosine is the cosine of the angle between two vectors.
- Vector cosine is represented by the following formula:

```math
x \cdot y = \lVert x \rVert \cdot \lVert y \rVert \cdot cos(\theta) \Rightarrow cos(\theta) = \frac{x \cdot y}{\lVert x \rVert \cdot \lVert y \rVert}
```

- Some angles between two vectors:
  - $`cos(\theta) = 1 \Rightarrow \theta = 0^{\circ}`$
  - $`cos(\theta) = 0 \Rightarrow \theta = 90^{\circ}`$
  - $`cos(\theta) = -1 \Rightarrow \theta = 180^{\circ}`$

### **1.8. Vector projection**

- Vector projection is the vector component of a vector in the direction of another vector.
- Vector projection is represented by the following formula:

```math
proj_{u}v = \frac{v \cdot u}{\lVert u \rVert^{2}} \cdot u
```

- Scalar projection is the magnitude of the vector projection. It is represented by the following formula:

```math
\lVert proj_{u}v \rVert = \frac{v \cdot u}{\lVert u \rVert} = \lVert v \rVert \cdot cos(\theta)
```

- *Example*: Find the vector projection of $`r = \begin{bmatrix} 3\\ 4 \end{bmatrix}`$ onto the pair of vectors $`b_{1} = \begin{bmatrix} 2\\ 1 \end{bmatrix}`$ and $`b_{2} = \begin{bmatrix} -2\\ 4 \end{bmatrix}`$.

- *Solution*: first we need to check the pair of vectors $`b_{1}`$ and $`b_{2}`$ are orthogonal or not. If they are orthogonal, we can use the above formula to find the vector projection of $`r`$ onto $`b_{1}`$ and $`b_{2}`$. (In this chapter, we will find the orthogonal basis of a vector space. If not, we can use the Gram-Schmidt process to find the orthogonal basis of a vector space in the next chapter.)

- *Step 1*: Check the pair of vectors $`b_{1}`$ and $`b_{2}`$ are orthogonal or not.

```math
b_{1} \cdot b_{2} = 2 \cdot (-2) + 1 \cdot 4 = 0 \Rightarrow b_{1} \perp b_{2}
```

- *Step 2*: Find the vector projection of $`r`$ onto $`b_{1}`$.

```math
proj_{b_{1}}r = \frac{r \cdot b_{1}}{ \lVert b_{1} \rVert ^{2}} \cdot b_{1} = \frac{3 \cdot 2 + 4 \cdot 1}{2^{2} + 1^{2}} \cdot b_{1} = \frac{10}{5} \cdot b_{1} = 2b_{1} = 2 \cdot \begin{bmatrix} 2\\ 1 \end{bmatrix}
```

- *Step 3*: Find the vector projection of $`r`$ onto $`b_{2}`$.

```math
proj_{b_{2}}r = \frac{r \cdot b_{2}}{\lVert b_{2} \rVert ^{2}} \cdot b_{2} = \frac{3 \cdot (-2) + 4 \cdot 4}{(-2)^{2} + 4^{2}} \cdot b_{2} = \frac{10}{20} \cdot b_{2} = \frac{1}{2}b_{2} = \frac{1}{2} \cdot \begin{bmatrix} -2\\ 4 \end{bmatrix}
```

- So, the vector projection of $`r`$ onto the pair of vectors $`b_{1}`$ and $`b_{2}`$ is:

```math
r = proj_{b_{1}}r + proj_{b_{2}}r = 2b_{1} + \frac{1}{2}b_{2} =2 \cdot \begin{bmatrix} 2\\ 1 \end{bmatrix} + \frac{1}{2} \cdot \begin{bmatrix} -2\\ 4 \end{bmatrix} = \begin{bmatrix} 3\\ 4 \end{bmatrix}
```

### **1.9. Linear combination and Linear independence**

- Linear combination is the sum of scalar multiples of vectors. The result is a vector. It is represented by the following formula:

```math
v_{1}, v_{2}, ..., v_{n} \in \mathbb{R}^{n}, k_{1}, k_{2}, ..., k_{n} \in \mathbb{R} \Rightarrow k_{1}v_{1} + k_{2}v_{2} + ... + k_{n}v_{n} = v \in \mathbb{R}^{n}
```

- Linear independence is the property of a set of vectors such that no vector in the set can be represented as a linear combination of the other vectors in the set. It is represented by the following formula:

```math
v_{1}, v_{2}, ..., v_{n} \in \mathbb{R}^{n} \Rightarrow k_{1}v_{1} + k_{2}v_{2} + ... + k_{n}v_{n} = 0 \Rightarrow k_{1} = k_{2} = ... = k_{n} = 0
```

- Basis is a set of linearly independent vectors that spans a vector space. In this space, all attributes of a vector is kept. It is represented by the following formula:
  - If $` v_{1}, v_{2}, ..., v_{n} \in \mathbb{R}^{n} `$ is linearly independent. So that with $` \forall v `$ in this space we have:

```math
k_{1}v_{1} + k_{2}v_{2} + ... + k_{n}v_{n} = v \in \mathbb{R}^{n}
```
