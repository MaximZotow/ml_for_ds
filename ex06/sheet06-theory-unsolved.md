# Exercise Sheet 6 (theory part): unsolved

*Machine Learning for Data Science, Winter Semester 2026/27
Prof. Dr.-Ing. Gerhard Wunder, Freie Universität Berlin*

---

## Exercise 1: Canonical Correlation Analysis (15 + 5 P)

**Recall:** For a sample of $d_1$- and $d_2$-dimensional data of size $N$, given as two data matrices $X \in \mathbb{R}^{d_1\times N}$, $Y \in \mathbb{R}^{d_2\times N}$ (assumed to be centered), canonical correlation analysis (CCA) finds a one-dimensional projection maximizing the cross-correlation for constant auto-correlation. The optimization problem is:

$$
\text{Find } w_x\in\mathbb{R}^{d_1},\, w_y\in\mathbb{R}^{d_2} \text{ maximizing } w_x^\top C_{xy} w_y \quad \text{subject to}\quad w_x^\top C_{xx} w_x = 1,\;\; w_y^\top C_{yy} w_y = 1,
$$

where $C_{xx} = \frac1N XX^\top$ and $C_{yy} = \frac1N YY^\top$ are the auto-covariance matrices, $C_{xy} = \frac1N XY^\top$ is the cross-covariance matrix, and $C_{yx} = C_{xy}^\top$.

**(a)** Show that a solution of the canonical correlation analysis can be found in some eigenvector of the generalized eigenvalue problem:

$$
\begin{bmatrix}0 & C_{xy}\\ C_{yx} & 0\end{bmatrix}\begin{bmatrix}w_x\\ w_y\end{bmatrix} = \lambda \begin{bmatrix}C_{xx} & 0\\ 0 & C_{yy}\end{bmatrix}\begin{bmatrix}w_x\\ w_y\end{bmatrix}
$$

*Hint:* use the method of Lagrange multipliers with one multiplier per constraint, set the gradients to zero, and multiply the resulting equations by $w_x^\top$ and $w_y^\top$ to show that both multipliers are equal.

**Answer:**

**(b)** Show that among all eigenvectors $(w_x, w_y)$ the solution is the one associated to the highest eigenvalue.

*Hint:* multiply the generalized eigenvalue equation by $(w_x, w_y)^\top$ on both sides and relate the left-hand side to the objective.

**Answer:**

---

## Exercise 2: CCA for High Dimensional Data (10 + 10 + 5 + 5 P)

Like for PCA, the original problem formulation involving the eigendecomposition of $d\times d$ covariance matrices does not scale well for high-dimensional data. Here, we would like to derive another formulation of the CCA problem that involves instead the eigendecomposition of a matrix whose size scales with the number of data points.

**(a)** Show that it is always possible to find an optimal solution in the span of the data, that is,

$$
w_x = X\alpha_x, \qquad w_y = Y\alpha_y
$$

with some coefficient vectors $\alpha_x\in\mathbb{R}^N$ and $\alpha_y\in\mathbb{R}^N$.

*Hint:* decompose $w_x = w_x^{(s)} + w_x^{(n)}$ into a component in the span of the data and a component orthogonal to it ($X^\top w_x^{(n)} = 0$), likewise for $w_y$, and show that the objective and the constraints do not depend on the orthogonal components.

**Answer:**

**(b)** Show that the solution of the resulting optimization problem is found in an eigenvector of the generalized eigenvalue problem

$$
\begin{bmatrix}0 & Q_{xy}\\ Q_{yx} & 0\end{bmatrix}\begin{bmatrix}\alpha_x\\ \alpha_y\end{bmatrix} = \lambda \begin{bmatrix}Q_{xx} & 0\\ 0 & Q_{yy}\end{bmatrix}\begin{bmatrix}\alpha_x\\ \alpha_y\end{bmatrix}
$$

where $Q_{xy} = \frac1N X^\top X Y^\top Y$, $Q_{xx} = \frac1N X^\top X X^\top X$ and $Q_{yy} = \frac1N Y^\top Y Y^\top Y$.

*Hint:* substitute $w_x = X\alpha_x$, $w_y = Y\alpha_y$ in the original problem and reuse the derivation of Exercise 1(a).

**Answer:**

**(c)** Show that the solution is given by the eigenvector associated to the highest eigenvalue.

**Answer:**

**(d)** Show how a solution to the original CCA problem can be obtained from the solution of the latter generalized eigenvalue problem.

**Answer:**
