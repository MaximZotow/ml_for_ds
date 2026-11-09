# Exercise Sheet 4 (theory part): solved

1. Maxim Zotov
1. Furkan Dursun
1. Lewin Ahlgrimm

*Machine Learning for Data Science, Winter Semester 2026/27 — Prof. Dr.-Ing. Gerhard Wunder, Freie Universität Berlin*

---

## Exercise 1: Convex Optimization (10 + 10 + 20 + 20 + 10 P)

The SVDD method for anomaly detection is formulated as a convex optimization problem. Here, we consider a simplified variant of SVDD, the hard-margin SVDD, consisting of finding a maximum enclosing hypersphere of the data:

$$
\min_{S,\boldsymbol{c}} S \quad \text{s.t.} \quad \forall_{i=1}^{N}: \|\boldsymbol{c} - \boldsymbol{x}_i\|^2 \leq S
$$

where $S$ denotes the squared radius of the hypersphere.

**(a)** Verify that there is a point $(S, \boldsymbol{c})$ such that all constraints above are satisfied.

*Hint:* the constraints only bound $S$ from below for a given $\boldsymbol{c}$; pick any $\boldsymbol{c}$ and choose $S$ large enough.

**Answer:**

**(b)** Write the Lagrange function associated to this optimization problem.

*Hint:* introduce one multiplier $\alpha_i \geq 0$ per inequality constraint, written in the form $g_i(S,\boldsymbol{c}) \leq 0$, and add $\sum_i \alpha_i g_i$ to the objective.

**Answer:**

**(c)** State the KKT conditions for this problem and simplify them.

*Hint:* the four groups are stationarity ($\partial L/\partial S = 0$, $\partial L/\partial \boldsymbol{c} = 0$), primal feasibility, dual feasibility ($\alpha_i \geq 0$) and complementary slackness ($\alpha_i \cdot g_i = 0$). Use stationarity to simplify, e.g. $\sum_i \alpha_i = 1$ and an expression for $\boldsymbol{c}$.

**Answer:**

**(d)** Show that the dual of SVDD is given by:

$$
\max_{\boldsymbol{\alpha}} \; \sum_{i=1}^{N} \alpha_i \|\boldsymbol{x}_i\|^2 - \sum_{i=1}^{N}\sum_{j=1}^{N} \alpha_i \alpha_j \boldsymbol{x}_i^\top \boldsymbol{x}_j
$$

subject to:

$$
\sum_{i=1}^{N} \alpha_i = 1, \qquad \forall_{i=1}^{N}: 0 \leq \alpha_i
$$

*Hint:* the dual is $\max_{\boldsymbol{\alpha}} \inf_{S,\boldsymbol{c}} L(S,\boldsymbol{c},\boldsymbol{\alpha})$. Group the terms in $S$ (they vanish when $\sum_i\alpha_i = 1$), expand $\|\boldsymbol{c}-\boldsymbol{x}_i\|^2 = \boldsymbol{c}^\top\boldsymbol{c} - 2\boldsymbol{x}_i^\top\boldsymbol{c} + \|\boldsymbol{x}_i\|^2$, then substitute $\boldsymbol{c} = \sum_i \alpha_i \boldsymbol{x}_i$ from the KKT conditions.

**Answer:**

**(e)** Express the primal variables $S, \boldsymbol{c}$ as a function of dual variables $\alpha_1, \dots, \alpha_N$.

*Hint:* $\boldsymbol{c}$ follows directly from stationarity. For $S$, use complementary slackness: for any $k$ with $\alpha_k > 0$, what must hold for the constraint $k$?

**Answer:**
