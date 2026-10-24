# Exercise Sheet 2 (theory part): unsolved

*Machine Learning for Data Science, Winter Semester 2026/27 — Prof. Dr.-Ing. Gerhard Wunder, Freie Universität Berlin*

## Exercise 1: Concentration of Squared Distances (10 + 10 + 10 P)

We study the concentration of squared distances in high dimensions. Let $\boldsymbol{x} \in \mathbb{R}^d$ be drawn i.i.d. from $\boldsymbol{x} \sim \mathcal{N}(0, I)$, where $I$ is the $d \times d$ identity matrix. Consider

$$y(\boldsymbol{x}, \boldsymbol{x}') = \|\boldsymbol{x} - \boldsymbol{x}'\|^2,$$

the squared Euclidean distance between two points drawn from this distribution.

**(a)** Express the mean of $y$ as a function of the number of input dimensions $d$.

*Hint:* write $y$ as a sum over coordinates, or use the chi-squared distribution $\chi^2_d$.

**Answer:**

**(b)** Express the standard deviation of $y$ as a function of $d$.

*Hint:* $\mathrm{Var}(\chi^2_d) = 2d$ and $\mathrm{Var}(aZ) = a^2\,\mathrm{Var}(Z)$.

**Answer:**

**(c)** Show that $\mathrm{std}[y] / \mathrm{E}[y] = \sqrt{2/d}$, and that therefore squared distances concentrate more as $d$ grows.

**Answer:**

---

## Exercise 2: Gradient of T-SNE (15 + 15 P)

T-SNE minimizes the cross-entropy between two probability distributions $p$ and $q$ that describe pairwise similarities of $N$ data points in input space and in embedding space (one-dimensional here, with coordinates $y_1, \dots, y_N$). The objective is computed as

$$
\Delta_{kl} = y_k - y_l
\qquad
s_{kl} = \Delta_{kl}^2
\qquad
z_{kl} = (1 + s_{kl})^{-1}
\qquad
q_{ij} = \frac{z_{ij}}{\sum_{kl} z_{kl}}
\qquad
C = -\sum_{ij} p_{ij} \log q_{ij}
$$

where $\sum_{ij}$ means $\sum_{i=1}^N \sum_{j=1}^N$, and likewise for $\sum_{kl}$.

**(a)** Using the decomposition above ($y \to \Delta \to s \to z \to q \to C$), state the multivariate chain rule for computing $\partial C / \partial y_m$.

*Hint:* sum over all paths from $y_m$ to $C$. Note that $q_{ij}$ depends on all $z_{kl}$ through the normalization.

**Answer:**

**(b)** Starting from the chain rule, derive

$$
\frac{\partial C}{\partial y_m} = 2 \sum_{kl} (p_{kl} - q_{kl}) \cdot \frac{\Delta_{kl}}{1 + \Delta_{kl}^2} \cdot (\delta_{km} - \delta_{lm})
\qquad \text{where} \qquad
\delta_{kl} = \begin{cases} 1 & k = l \\ 0 & k \neq l \end{cases}
$$

*Hints:* let $A = \sum_{kl} z_{kl}$; compute each local derivative separately; use $\sum_{ij} p_{ij} = \sum_{ij} q_{ij} = 1$ to simplify.

**Answer:**
