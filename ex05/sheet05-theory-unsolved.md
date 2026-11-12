# Exercise Sheet 5 (theory part): unsolved

*Machine Learning for Data Science, Winter Semester 2026/27 — Prof. Dr.-Ing. Gerhard Wunder, Freie Universität Berlin*

---

## Exercise 1: K-Means Clustering (15 + 10 P)

The K-means optimization problem is given by $\arg\min_{\boldsymbol{\mu}, c} \sum_{i=1}^{N} \|\boldsymbol{x}_i - \boldsymbol{\mu}_{c_i}\|^2$ where $c \in \{1, \dots, K\}^N$ is the cluster assignment function. When considering the latter to be fixed, and only letting the centroids $\boldsymbol{\mu}$ vary, the optimization problem can be restated as:

$$
\arg\min_{\boldsymbol{\mu}} \sum_{k=1}^{K} \sum_{i \in C_k} \|\boldsymbol{x}_i - \boldsymbol{\mu}_k\|^2
$$

where $C_k$ is the set of instances that are assigned to cluster $k$.

**(a)** Show that the solution of this optimization problem is given by:

$$
\boldsymbol{\mu} = (\boldsymbol{\mu}_k)_{k=1}^{K} \quad \text{where} \quad \boldsymbol{\mu}_k = \frac{\sum_{i \in C_k} \boldsymbol{x}_i}{\sum_{i \in C_k} 1}
$$

*Hint:* the objective is convex in $\boldsymbol{\mu}$, so it is minimized where the gradient vanishes. The objective is a sum over $k$, so differentiate with respect to each $\boldsymbol{\mu}_k$ separately.

**Answer:**

**(b)** A data point $\boldsymbol{x}$ is assigned onto cluster $c$ if

$$
\forall k: k \neq c: \; \|\boldsymbol{x} - \boldsymbol{\mu}_c\| < \|\boldsymbol{x} - \boldsymbol{\mu}_k\|.
$$

Show that this condition for assignment onto cluster $c$ can be equivalently formulated as a min-pooling over affine functions, specifically, we assign to cluster $c$ if

$$
\min_{k: k \neq c} \left( \boldsymbol{w}_k^\top \boldsymbol{x} + b_k \right) > 0
$$

where $\boldsymbol{w}_k = (\boldsymbol{\mu}_c - \boldsymbol{\mu}_k)$ and $b_k = \frac{1}{2}\left(\|\boldsymbol{\mu}_k\|^2 - \|\boldsymbol{\mu}_c\|^2\right)$.

*Hint:* both distances are nonnegative, so you may square them (and scale by $\tfrac12$) without changing the inequality. Then expand the squared norms, and note that the $\|\boldsymbol{x}\|^2$ terms cancel.

**Answer:**

---

## Exercise 2: Spectral Clustering (15 + 10 P)

In the lecture, it was mentioned that the eigenvalues $\lambda$ of the Laplacian matrix $L = D - A$ (where $D$ and $A$ are the degree and adjacency matrices respectively) can be related to the corresponding eigenvector $\boldsymbol{u}$ as:

$$
\lambda = \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} A_{ij} \left( u_i - u_j \right)^2.
$$

**(a)** Prove the equation above.

*Hint:* start from $\lambda = \boldsymbol{u}^\top L \boldsymbol{u}$ (assuming $\|\boldsymbol{u}\| = 1$), split $L = D - A$, and use $D_{ii} = \sum_j A_{ij}$. Then symmetrize the first term using $A_{ij} = A_{ji}$ so that it splits as $\tfrac12 u_i^2 + \tfrac12 u_j^2$.

**Answer:**

**(b)** From the equation above, we can see that the eigenvalue $\lambda$ influences the extent by which the associated eigenvector $\boldsymbol{u}$ can vary between connected nodes.

Show that eigenvectors associated to the eigenvalue $\lambda = 0$ cannot vary within a connected component, that is, denoting by $\boldsymbol{u}$ the eigenvector, show that if $i$ and $j$ are part of the same connected component (i.e. if there is a sequence of edges connecting $i$ and $j$), then $u_i = u_j$.

*Hint:* every summand in the formula above is nonnegative, so a zero sum forces every summand to be zero. First treat directly connected nodes, then chain the equalities along a path.

**Answer:**
