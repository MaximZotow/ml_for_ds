# Exercise Sheet 3 (theory part): 

*Machine Learning for Data Science, Winter Semester 2026/27 — Prof. Dr.-Ing. Gerhard Wunder, Freie Universität Berlin

Students: Furkan Dursun, Maxim Zotov, Lewin Ahlgrimm*

---

## Exercise 1: Principal Component Analysis (15 + 15 P)

We consider a dataset $\boldsymbol{x}_1, \dots, \boldsymbol{x}_N \in \mathbb{R}^d$. Principal component analysis searches for a unit vector $\boldsymbol{u} \in \mathbb{R}^d$ such that projecting the data on that vector produces a distribution with maximum variance. Such vector can be found by solving the optimization problem:

$$
\arg\max_{\boldsymbol{u}} \; \frac{1}{N} \sum_{i=1}^{N} \left( \boldsymbol{u}^\top (\boldsymbol{x}_i - \boldsymbol{m}) \right)^2 \quad \text{with} \quad \|\boldsymbol{u}\|^2 = 1 \quad \text{and} \quad \boldsymbol{m} = \frac{1}{N} \sum_{i=1}^{N} \boldsymbol{x}_i.
$$

**(a)** Show that the problem above can be rewritten as

$$
\arg\max_{\boldsymbol{u}} \; \boldsymbol{u}^\top \Sigma \, \boldsymbol{u} \quad \text{with} \quad \|\boldsymbol{u}\|^2 = 1, \qquad \text{where} \quad \Sigma = \frac{1}{N} \sum_{i=1}^{N} (\boldsymbol{x}_i - \boldsymbol{m})(\boldsymbol{x}_i - \boldsymbol{m})^\top
$$

is the covariance matrix.

**Answer:**

As  $\boldsymbol{u}^\top (\boldsymbol{x}_i - \boldsymbol{m})$ is scalar we can write $(\boldsymbol{u}^\top (\boldsymbol{x}_i - \boldsymbol{m}) )^2$ such that:

$$
\left( \boldsymbol{u}^\top (\boldsymbol{x}_i - \boldsymbol{m}) \right)^2
=
\boldsymbol{u}^\top (\boldsymbol{x}_i - \boldsymbol{m})(\boldsymbol{x}_i - \boldsymbol{m})^\top \boldsymbol{u}.
$$

Therefore,

$$
\begin{aligned}
\frac{1}{N} \sum_{i=1}^{N} \left( \boldsymbol{u}^\top (\boldsymbol{x}_i - \boldsymbol{m}) \right)^2
&=
\frac{1}{N} \sum_{i=1}^{N}
\boldsymbol{u}^\top (\boldsymbol{x}_i - \boldsymbol{m})(\boldsymbol{x}_i - \boldsymbol{m})^\top \boldsymbol{u} \\
&=
\boldsymbol{u}^\top
\left(
\frac{1}{N} \sum_{i=1}^{N} (\boldsymbol{x}_i - \boldsymbol{m})(\boldsymbol{x}_i - \boldsymbol{m})^\top
\right)
\boldsymbol{u} \\
&=
\boldsymbol{u}^\top \Sigma \boldsymbol{u}.
\end{aligned}
$$

The constraint $\|\boldsymbol{u}\|^2 = 1$ is unchanged. Hence the original optimization problem is equivalent to

$$
\arg\max_{\boldsymbol{u}} \; \boldsymbol{u}^\top \Sigma \boldsymbol{u}
\quad \text{with} \quad \|\boldsymbol{u}\|^2 = 1.
$$

---

**(b)** Show using the method of Lagrange multipliers that the problem above can be reformulated as solving the eigenvalue problem $\Sigma \boldsymbol{u} = \lambda \boldsymbol{u}$ and retaining the eigenvector $\boldsymbol{u}$ associated to the highest eigenvalue $\lambda$.

**Answer:**

We solve

$$
\max_{\boldsymbol{u}} \; \boldsymbol{u}^\top \Sigma \boldsymbol{u}
\quad \text{subject to} \quad
\boldsymbol{u}^\top \boldsymbol{u} - 1 = 0.
$$
$$
\mathcal{L}(\boldsymbol{u}, \lambda)
=
\boldsymbol{u}^\top \Sigma \boldsymbol{u}
-
\lambda \left( \boldsymbol{u}^\top \boldsymbol{u} - 1 \right).
$$

Taking the derivative with respect to $\boldsymbol{u}$ and setting it to zero gives

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{u}}
=
2 \Sigma \boldsymbol{u} - 2 \lambda \boldsymbol{u}
=
0.
$$

Hence

$$
\Sigma \boldsymbol{u} = \lambda \boldsymbol{u}.
$$

So  any stationary point of the optimization problem is an eigenvector of $\Sigma$. If we multiply both sides by $\boldsymbol{u}^\top$ and using $\boldsymbol{u}^\top \boldsymbol{u} = 1$, we obtain

$$
\boldsymbol{u}^\top \Sigma \boldsymbol{u}
=
\lambda \boldsymbol{u}^\top \boldsymbol{u}
=
\lambda.
$$
LHS of the equation ($\boldsymbol{u}^\top \Sigma \boldsymbol{u}$) is our objective which we want to maximize. 

For the objective to be maximized, we should choose the
eigenvector in a way that the corresponding eigenvalue λ is maximum. In other words, one should choose the leading eigenvector.


---

## Exercise 2: Bounds on Eigenvalues (10 + 10 P)

Let $\lambda_1$ denote the largest eigenvalue of the matrix $\Sigma$. The eigenvalue $\lambda_1$ measures the variance of the data when projected on the first principal component. We study how the latter can be bounded with the diagonal elements of the matrix $\Sigma$.

*Preliminary observation* for all eigenvalues/eigenvectors $(\lambda_i, \boldsymbol{u}_i)$: show that $\lambda_i = \boldsymbol{u}_i^\top \Sigma \boldsymbol{u}_i \geq 0$.

**Answer (preliminary observation):**

Let $(\lambda_i, \boldsymbol{u}_i)$ be an eigenpair of $\Sigma$, so that

$$
\Sigma \boldsymbol{u}_i = \lambda_i \boldsymbol{u}_i,
\qquad
\|\boldsymbol{u}_i\| = 1.
$$

Then, take $\Sigma \boldsymbol{u}_i = \lambda_i \boldsymbol{u}_i,
\qquad$multiply both sides from left with $\boldsymbol{u}_i^\top$:

$$
\boldsymbol{u}_i^\top \Sigma \boldsymbol{u}_i
=
\boldsymbol{u}_i^\top (\lambda_i \boldsymbol{u}_i)
=
\lambda_i \boldsymbol{u}_i^\top \boldsymbol{u}_i
=
\lambda_i.
$$

Using the definition of $\Sigma$,

$$
\begin{aligned}
\lambda_i
&=
\boldsymbol{u}_i^\top \Sigma \boldsymbol{u}_i \\
&=
\boldsymbol{u}_i^\top
\left(
\frac{1}{N} \sum_{j=1}^{N} (\boldsymbol{x}_j - \boldsymbol{m})(\boldsymbol{x}_j - \boldsymbol{m})^\top
\right)
\boldsymbol{u}_i \\
&=
\frac{1}{N} \sum_{j=1}^{N}
\boldsymbol{u}_i^\top (\boldsymbol{x}_j - \boldsymbol{m})(\boldsymbol{x}_j - \boldsymbol{m})^\top \boldsymbol{u}_i \quad \text{(distribute } \boldsymbol{u}_i^\top \text{ and } \boldsymbol{u}_i \text{ into the sum)} \\
&=
\frac{1}{N} \sum_{j=1}^{N}
\left( \boldsymbol{u}_i^\top (\boldsymbol{x}_j - \boldsymbol{m}) \right)
\left( (\boldsymbol{x}_j - \boldsymbol{m})^\top \boldsymbol{u}_i \right) \quad \text{(group the terms)} \\
&=
\frac{1}{N} \sum_{j=1}^{N}
\left( \boldsymbol{u}_i^\top (\boldsymbol{x}_j - \boldsymbol{m}) \right)^2 \quad \text{(since scalar transpose } \boldsymbol{a}^\top \boldsymbol{b} = (\boldsymbol{a}^\top \boldsymbol{b})^\top = \boldsymbol{b}^\top \boldsymbol{a} \text{)} \\
&\geq 0.
\end{aligned}
$$

---

**(a)** Show that $\sum_{i=1}^{d} \Sigma_{ii}$ is an upper bound to the eigenvalue $\lambda_1$.

*Hint:* relate the trace of $\Sigma$ to its eigenvalues.

**Answer:**

The trace of $\Sigma$ is the sum of its diagonal entries:

$$
\operatorname{tr}(\Sigma)
=
\sum_{i=1}^{d} \Sigma_{ii}.
$$

Since $\Sigma$ is symmetric, it has an orthonormal eigenbasis, and the trace equals the sum of its eigenvalues:

$$
\operatorname{tr}(\Sigma)
=
\sum_{i=1}^{d} \lambda_i.
$$

Therefore,

$$
\sum_{i=1}^{d} \Sigma_{ii}
=
\sum_{i=1}^{d} \lambda_i.
$$

From the preliminary observation, all eigenvalues are nonnegative, so

$$
\sum_{i=1}^{d} \lambda_i
=
\lambda_1 + \sum_{i=2}^{d} \lambda_i
\geq
\lambda_1.
$$

Hence

$$
\sum_{i=1}^{d} \Sigma_{ii}
=
\operatorname{tr}(\Sigma)
=
\sum_{i=1}^{d} \lambda_i
\geq
\lambda_1.
$$

So $\sum_{i=1}^{d} \Sigma_{ii}$ is an upper bound on $\lambda_1$.

---

**(b)** Show that $\max_{i=1}^{d} \Sigma_{ii}$ is a lower bound to the eigenvalue $\lambda_1$.

*Hint:* $\lambda_1 = \max_{\|\boldsymbol{u}\|=1} \boldsymbol{u}^\top \Sigma \boldsymbol{u}$; restrict the maximization to the canonical basis vectors $\boldsymbol{e}_1, \dots, \boldsymbol{e}_d$.

**Answer:**

By the variational characterization of the largest eigenvalue of a symmetric matrix,

$$
\lambda_1
=
\max_{\|\boldsymbol{u}\|=1} \boldsymbol{u}^\top \Sigma \boldsymbol{u}.
$$

For every canonical basis vector $\boldsymbol{e}_i$, we have $\|\boldsymbol{e}_i\| = 1$, and

$$
\boldsymbol{e}_i^\top \Sigma \boldsymbol{e}_i
=
\Sigma_{ii}.
$$

Since $\lambda_1$ is the maximum over all unit vectors, it must be at least as large as the value obtained by any particular unit vector. In particular, for every $i$,

$$
\lambda_1
\geq
\boldsymbol{e}_i^\top \Sigma \boldsymbol{e}_i
=
\Sigma_{ii}.
$$

Therefore,

$$
\lambda_1
\geq
\max_{i=1}^{d} \Sigma_{ii}.
$$

So $\max_{i=1}^{d} \Sigma_{ii}$ is a lower bound on $\lambda_1$.

---

## Exercise 3: Iterative PCA (5 + 10 + 5 P)

When performing principal component analysis, computing the full eigendecomposition of the covariance matrix $\Sigma$ is typically slow, and we are often only interested in the first principal components. An efficient procedure to find the first principal component is *power iteration*. It starts with a random unit vector $\boldsymbol{w}^{(0)} \in \mathbb{R}^d$, and iteratively applies the parameter update

$$
\boldsymbol{w}^{(t+1)} = \frac{\Sigma \boldsymbol{w}^{(t)}}{\|\Sigma \boldsymbol{w}^{(t)}\|}
$$

until some convergence criterion is met. Here, we would like to show the exponential convergence of power iteration. For this, we look at the error terms

$$
E_k(\boldsymbol{w}) = \frac{\boldsymbol{w}^\top \boldsymbol{u}_k}{\boldsymbol{w}^\top \boldsymbol{u}_1} \qquad \text{with} \quad k = 2, \dots, d,
$$

and demonstrate that they all converge to zero as $\boldsymbol{w}$ approaches the eigenvector $\boldsymbol{u}_1$ and becomes orthogonal to the other eigenvectors. To demonstrate this, we proceed in three steps:

**(a)** Show that

$$
E_k(\boldsymbol{w}^{(t+1)}) = \frac{\boldsymbol{w}^{(t)\top} \Sigma \boldsymbol{u}_k}{\boldsymbol{w}^{(t)\top} \Sigma \boldsymbol{u}_1}.
$$

**Answer:**

$$
\begin{aligned}
E_k(\boldsymbol{w}^{(t+1)})
&= \frac{\boldsymbol{w}^{(t+1)\top}\boldsymbol{u}_k}
{\boldsymbol{w}^{(t+1)\top}\boldsymbol{u}_1}
\\[6pt]
&= \frac{
\left(\frac{\Sigma\boldsymbol{w}^{(t)}}{\|\Sigma\boldsymbol{w}^{(t)}\|}\right)^\top
\boldsymbol{u}_k
}{
\left(\frac{\Sigma\boldsymbol{w}^{(t)}}{\|\Sigma\boldsymbol{w}^{(t)}\|}\right)^\top
\boldsymbol{u}_1
}
\\[6pt]
&= \frac{
(\Sigma\boldsymbol{w}^{(t)})^\top\boldsymbol{u}_k
}{
(\Sigma\boldsymbol{w}^{(t)})^\top\boldsymbol{u}_1
}
\\[6pt]
&= \frac{
\boldsymbol{w}^{(t)\top}\Sigma^\top\boldsymbol{u}_k
}{
\boldsymbol{w}^{(t)\top}\Sigma^\top\boldsymbol{u}_1
}
\\[6pt]
&= \frac{
\boldsymbol{w}^{(t)\top}\Sigma\boldsymbol{u}_k
}{
\boldsymbol{w}^{(t)\top}\Sigma\boldsymbol{u}_1
}.
\end{aligned}
$$

The last equality follows because the covariance matrix is symmetric, i.e. $\Sigma^\top = \Sigma$. $\square$

**(b)** Starting from the result above, show that

$$
E_k(\boldsymbol{w}^{(t+1)}) = \left|\frac{\lambda_k}{\lambda_1}\right| \cdot E_k(\boldsymbol{w}^{(t)}).
$$

*Hint:* Recall that the covariance is linked to its eigenvectors and eigenvalues through the equation $\Sigma \boldsymbol{u}_i = \lambda_i \boldsymbol{u}_i$.

**Answer:**

Using the result from (a) and the eigenvalue equation, we obtain

$$
\begin{aligned}
E_k(\boldsymbol{w}^{(t+1)})
&= \frac{
\boldsymbol{w}^{(t)\top}\Sigma\boldsymbol{u}_k
}{
\boldsymbol{w}^{(t)\top}\Sigma\boldsymbol{u}_1
}
\\[6pt]
&= \frac{
\boldsymbol{w}^{(t)\top}\lambda_k\boldsymbol{u}_k
}{
\boldsymbol{w}^{(t)\top}\lambda_1\boldsymbol{u}_1
}
\\[6pt]
&= \left|\frac{\lambda_k}{\lambda_1}\right|
\frac{
\boldsymbol{w}^{(t)\top}\boldsymbol{u}_k
}{
\boldsymbol{w}^{(t)\top}\boldsymbol{u}_1
}
\\[6pt]
&= \left|\frac{\lambda_k}{\lambda_1}\right|
E_k(\boldsymbol{w}^{(t)}).
\end{aligned}
$$

Since $\Sigma$ is a covariance matrix, all its eigenvalues are nonnegative, and we assume $\lambda_1 > 0$. $\square$

**(c)** Starting from the result above, show that

$$
E_k(\boldsymbol{w}^{(T)}) = \left|\frac{\lambda_k}{\lambda_1}\right|^T \cdot E_k(\boldsymbol{w}^{(0)}),
$$

i.e. the convergence of the algorithm is exponential with the number of time steps $T$.

**Answer:**

We prove the statement by mathematical induction.

**Base case ($T=0$):**

$$
\begin{aligned}
E_k(\boldsymbol{w}^{(0)})
&= \left|\frac{\lambda_k}{\lambda_1}\right|^0
E_k(\boldsymbol{w}^{(0)})
\\
&= E_k(\boldsymbol{w}^{(0)}).
\end{aligned}
$$

Thus, the statement holds for $T=0$.

**Induction hypothesis:**

Assume that the statement holds for some $T \in \mathbb{N}_0$, i.e.

$$
E_k(\boldsymbol{w}^{(T)})
= \left|\frac{\lambda_k}{\lambda_1}\right|^T
E_k(\boldsymbol{w}^{(0)}).
$$

**Induction step ($T \rightarrow T+1$):**

Using the result from (b) and the induction hypothesis, we obtain

$$
\begin{aligned}
E_k(\boldsymbol{w}^{(T+1)})
&= \left|\frac{\lambda_k}{\lambda_1}\right|
E_k(\boldsymbol{w}^{(T)})
\\[6pt]
&= \left|\frac{\lambda_k}{\lambda_1}\right|
\left|\frac{\lambda_k}{\lambda_1}\right|^T
E_k(\boldsymbol{w}^{(0)})
\\[6pt]
&= \left|\frac{\lambda_k}{\lambda_1}\right|^{T+1}
E_k(\boldsymbol{w}^{(0)}).
\end{aligned}
$$

Therefore, the statement also holds for $T+1$.

By mathematical induction, we conclude that

$$
\boxed{
E_k(\boldsymbol{w}^{(T)})
= \left|\frac{\lambda_k}{\lambda_1}\right|^T
E_k(\boldsymbol{w}^{(0)})
}
$$

for all $T \in \mathbb{N}_0$. $\square$