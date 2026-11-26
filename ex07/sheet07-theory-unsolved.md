# Exercise Sheet 7 (theory part): unsolved

*Machine Learning for Data Science, Winter Semester 2026/27
Prof. Dr.-Ing. Gerhard Wunder, Freie Universität Berlin*

---

## Exercise 1: CCA and Least Squares Regression (20 P)

The least squares regression problem assumes an input matrix $X \in \mathbb{R}^{d\times N}$ and a vector of outputs $Y \in \mathbb{R}^{N}$, both centered, and has its solution given by the optimization problem:

$$
\min_{v\in\mathbb{R}^d} \|X^\top v - Y\|^2
$$

Canonical correlation analysis (CCA), on the other hand, assumes an input matrix $X \in \mathbb{R}^{d_1\times N}$ and another input matrix $Y \in \mathbb{R}^{d_2\times N}$, both centered, and the CCA solution is the leading eigenvector of the generalized eigenvalue problem:

$$
\begin{bmatrix}0 & XY^\top\\ YX^\top & 0\end{bmatrix}\begin{bmatrix}w_x\\ w_y\end{bmatrix} = \lambda \begin{bmatrix}XX^\top & 0\\ 0 & YY^\top\end{bmatrix}\begin{bmatrix}w_x\\ w_y\end{bmatrix}
$$

We would like to relate the solutions $v$ and $(w_x, w_y)$ of these two analyses for a special case.

**(a)** Show that if feeding to CCA two matrices $X \in \mathbb{R}^{d\times N}$ and $Y \in \mathbb{R}^{1\times N}$, then the resulting vector $w_x$ is equivalent (up to a scaling factor) to the solution $v$ of a least squares regression problem to which we feed the same matrices $X$ and $Y$, with the latter represented as a vector.

*Hint:* write the block system as two equations, isolate $w_x$ from the second one, and note that $w_y$ is a scalar. Compare with the closed-form least squares solution $v = (XX^\top)^{-1}XY^\top$.

**Answer:**

---

## Exercise 2: Fisher Discriminant (10 + 10 + 10 P)

The objective function to find the Fisher discriminant has the form

$$
\max_w \frac{w^\top S_B w}{w^\top S_W w}
$$

where $S_B = (m_2 - m_1)(m_2 - m_1)^\top$ is the between-class scatter matrix and $S_W$ is the within-class scatter matrix, assumed to be positive definite. Because there are infinitely many solutions (multiplying $w$ by a scalar doesn't change the objective), we can extend the objective with a constraint, e.g. one that enforces $w^\top S_W w = 1$.

**(a)** Reformulate the problem above as an optimization problem with a quadratic objective and a quadratic constraint.

**Answer:**

**(b)** Show using the method of Lagrange multipliers that the solution of the reformulated problem is also a solution of the generalized eigenvalue problem:

$$
S_B w = \lambda S_W w
$$

**Answer:**

**(c)** Show that the solution of this optimization problem is equivalent (up to a scaling factor) to

$$
w = S_W^{-1}(m_1 - m_2)
$$

*Hint:* insert $S_B = (m_2-m_1)(m_2-m_1)^\top$ into the eigenvalue equation and notice that $(m_2-m_1)^\top w$ is a scalar.

**Answer:**
