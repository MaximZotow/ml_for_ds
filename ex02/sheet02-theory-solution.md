# Exercise Sheet 2 (theory part):

*Machine Learning for Data Science, Winter Semester 2026/27 — Prof. Dr.-Ing. Gerhard Wunder, Freie Universität Berlin*

## Exercise 1: Concentration of Squared Distances (10 + 10 + 10 P)

We study the concentration of squared distances in high dimensions. Let $\boldsymbol{x} \in \mathbb{R}^d$ be drawn i.i.d. from $\boldsymbol{x} \sim \mathcal{N}(0, I)$, where $I$ is the $d \times d$ identity matrix. Consider

$$y(\boldsymbol{x}, \boldsymbol{x}') = \|\boldsymbol{x} - \boldsymbol{x}'\|^2,$$

the squared Euclidean distance between two points drawn from this distribution.

**(a)** Express the mean of $y$ as a function of the number of input dimensions $d$.

*Hint:* write $y$ as a sum over coordinates, or use the chi-squared distribution $\chi^2_d$.


We define
$$
y(\boldsymbol{x},\boldsymbol{x}')
=\|\boldsymbol{x}-\boldsymbol{x}'\|_2^2
=\sum_{i=1}^{d}(X_i-X_i')^2.
$$

<!-- Expand the square and split the sum. -->
Expanding the square gives
$$
y=\sum_{i=1}^{d}X_i^2
-2\sum_{i=1}^{d}X_iX_i'
+\sum_{i=1}^{d}(X_i')^2.
$$

<!-- The two sums of squares follow chi-squared distributions. -->
Since the coordinates are independent standard normal variables,
$$
\sum_{i=1}^{d}X_i^2\sim\chi_d^2,
\qquad
\sum_{i=1}^{d}(X_i')^2\sim\chi_d^2.
$$

<!-- Apply linearity of expectation and E[chi_d^2] = d. -->
Therefore,
$$
\begin{aligned}
\mathbb{E}[y]
&=\mathbb{E}\left[\sum_{i=1}^{d}X_i^2\right]
-2\sum_{i=1}^{d}\mathbb{E}[X_iX_i']\\
&\quad+\mathbb{E}\left[\sum_{i=1}^{d}(X_i')^2\right]\\
&=d-2\sum_{i=1}^{d}\mathbb{E}[X_iX_i']+d.
\end{aligned}
$$

<!-- Independence allows factorization; both means are zero. -->
Since \(X_i\) and \(X_i'\) are independent and have zero mean,
$$
\mathbb{E}[X_iX_i']
=\mathbb{E}[X_i]\mathbb{E}[X_i']=0.
$$

Hence,
$$
\boxed{\mathbb{E}[y]=2d}.
$$


**(b)** Express the standard deviation of $y$ as a function of $d$.

*Hint:* $\mathrm{Var}(\chi^2_d) = 2d$ and $\mathrm{Var}(aZ) = a^2\,\mathrm{Var}(Z)$.


<!-- Use the expansion from part (a). -->
$$
y=\sum_{i=1}^{d}X_i^2
-2\sum_{i=1}^{d}X_iX_i'
+\sum_{i=1}^{d}(X_i')^2.
$$

<!-- Apply the variance operator and separate the variances. -->
$$
\begin{aligned}
\operatorname{Var}(y)
&=\operatorname{Var}\left(
\sum_{i=1}^{d}X_i^2
-2\sum_{i=1}^{d}X_iX_i'
+\sum_{i=1}^{d}(X_i')^2
\right)\\
&=\operatorname{Var}\left(\sum_{i=1}^{d}X_i^2\right)
+4\operatorname{Var}\left(\sum_{i=1}^{d}X_iX_i'\right)\\
&\quad+\operatorname{Var}\left(\sum_{i=1}^{d}(X_i')^2\right).
\end{aligned}
$$

<!-- Use Var(chi_d^2) = 2d for the sums of squares. -->
The two sums of squares follow \(\chi_d^2\), so each has variance \(2d\).

<!-- Use the product variance formula for independent variables. -->
Since \(X_i\) and \(X_i'\) are independent,
$$
\begin{aligned}
\operatorname{Var}(X_iX_i')
&=\mathbb{E}[X_i]^2\operatorname{Var}(X_i')
+\mathbb{E}[X_i']^2\operatorname{Var}(X_i)\\
&\quad+\operatorname{Var}(X_i)\operatorname{Var}(X_i')\\
&=0+0+1=1.
\end{aligned}
$$

<!-- The products are independent across i, so their variances add. -->
Therefore,
$$
\operatorname{Var}\left(\sum_{i=1}^{d}X_iX_i'\right)=d.
$$

<!-- Combine the terms and take the square root. -->
$$
\operatorname{Var}(y)=2d+4d+2d=8d.
$$

Hence, the standard deviation is
$$
\boxed{\operatorname{std}[y]=\sqrt{8d}=2\sqrt{2d}}.
$$





**(c)** Show that $\mathrm{std}[y] / \mathrm{E}[y] = \sqrt{2/d}$, and that therefore squared distances concentrate more as $d$ grows.



$$
\begin{aligned}
\frac{\operatorname{std}[y]}{\mathbb{E}[y]}
&=\frac{2\sqrt{2d}}{2d}\\
&=\frac{\sqrt{2d}}{\sqrt{d^2}}\\
&=\sqrt{\frac{2d}{d^2}}\\
&=\sqrt{\frac{2}{d}}.
\end{aligned}
$$

As \(d\) increases, \(\sqrt{2/d}\) approaches zero. Therefore, the relative spread of the squared distances decreases, meaning that squared distances concentrate more around their expected value.



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

$y_m$ can affect many pairwise differences $\Delta_{kl}$. Also each difference affects $s_{kl}$, then $z_{kl}$.

$$
{
\frac{\partial C}{\partial y_m}
=
\sum_{i,j}
\frac{\partial C}{\partial q_{ij}}
\frac{\partial q_{ij}}{\partial y_{m}}
}.
$$
Because $q_{ij}$ depends on every $z_{kl}$ through the normalization constant, we must sum over both pairs of indices:

Therefore, we sum over all paths through $q_{ij}$ and $z_{kl}$ using the multivariate chain rule:

$$
\boxed{
\frac{\partial C}{\partial y_m}
=
\sum_{i,j}
\frac{\partial C}{\partial q_{ij}}
\sum_{k,l}
\frac{\partial q_{ij}}{\partial z_{kl}}
\frac{\partial z_{kl}}{\partial s_{kl}}
\frac{\partial s_{kl}}{\partial \Delta_{kl}}
\frac{\partial \Delta_{kl}}{\partial y_m}
}.
$$


**(b)** Starting from the chain rule, derive

$$
\frac{\partial C}{\partial y_m}
=2\sum_{kl}(p_{kl}-q_{kl})
\frac{\Delta_{kl}}{1+\Delta_{kl}^2}
(\delta_{km}-\delta_{lm}),
$$

where $\delta_{kl}=1$ if $k=l$ and $0$ otherwise.

*Hints:* let $A = \sum_{kl} z_{kl}$; compute each local derivative separately; use $\sum_{ij} p_{ij} = \sum_{ij} q_{ij} = 1$ to simplify.

**Answer:**

First, calculate each local derivative then substitute them into the chain rule, and then simplifying the sums.

**Step 1:  $\partial C/\partial q_{ij}$.**

$$
C=-\sum_{i,j}p_{ij}\log q_{ij}
$$

When differentiating w.r.t one particular $q_{ij}$, every other terms are constant and its derivative is 0.  
And the derivative of $\log q_{ij}$ is $1/q_{ij}$,

$$
\boxed{
\frac{\partial C}{\partial q_{ij}}=-\frac{p_{ij}}{q_{ij}}
}
$$

**Step 2: $\partial q_{ij}/\partial z_{kl}$.**

$$
A=\sum_{k,l}z_{kl},
\qquad
q_{ij}=\frac{z_{ij}}{A}
$$

Both the numerator and denominator can depend on the selected $z_{kl}$, so we use the quotient rule:

$$
\frac{\partial q_{ij}}{\partial z_{kl}}
=
\frac{
A\frac{\partial z_{ij}}{\partial z_{kl}}
-z_{ij}\frac{\partial A}{\partial z_{kl}}
}{A^2}
$$

The derivative of $z_{ij}$ w.r.t $z_{kl}$ is $1$ when $(i,j)=(k,l)$ and $0$ otherwise.

$$
\frac{\partial z_{ij}}{\partial z_{kl}}
=\delta_{ik}\delta_{jl}
$$
Also, since $A$ is the sum of all the $z$ values
$$
\frac{\partial A}{\partial z_{kl}}=1
$$
Substitute:
$$
\begin{aligned}
\frac{\partial q_{ij}}{\partial z_{kl}}
&=\frac{A\delta_{ik}\delta_{jl}-z_{ij}}{A^2}\\
\end{aligned}
$$

$$
\boxed{
\frac{\partial q_{ij}}{\partial z_{kl}}
=\frac{\delta_{ik}\delta_{jl}}{A}-\frac{z_{ij}}{A^2}
}
$$


**Step 3: $\partial z_{kl}/\partial s_{kl}$.**

$$
z_{kl}=(1+s_{kl})^{-1}
$$

$$
\begin{aligned}
\frac{\partial z_{kl}}{\partial s_{kl}}
&=-1(1+s_{kl})^{-2}
\frac{\partial}{\partial s_{kl}}(1+s_{kl})\\
&=-(1+s_{kl})^{-2}\cdot 1\\

\end{aligned}
$$

$$
\boxed{
\frac{\partial z_{kl}}{\partial s_{kl}}
=-\frac{1}{(1+s_{kl})^2}
}
$$


**Step 4: $\partial s_{kl}/\partial\Delta_{kl}$.**

Since $s_{kl}=\Delta_{kl}^2$,

$$
\boxed{
\frac{\partial s_{kl}}{\partial\Delta_{kl}}=2\Delta_{kl}
}
$$


**Step 5: $\partial\Delta_{kl}/\partial y_m$**
$$
\begin{aligned}
\frac{\partial\Delta_{kl}}{\partial y_m}
&=\frac{\partial y_k}{\partial y_m}
-\frac{\partial y_l}{\partial y_m}\\
\end{aligned}
$$
Here, $\delta_{km}=1$ if $k=m$ and $0$ otherwise, and likewise for $\delta_{lm}$. Hence,
$$
\boxed{
\frac{\partial\Delta_{kl}}{\partial y_m}
=\delta_{km}-\delta_{lm}.
}
$$


**Step 6: Substitute all local derivatives into the chain rule.**

$$
\begin{aligned}
\frac{\partial C}{\partial y_m}
={}&\sum_{i,j}\left(-\frac{p_{ij}}{q_{ij}}\right)
\sum_{k,l}
\left(\frac{\delta_{ik}\delta_{jl}}{A}
-\frac{z_{ij}}{A^2}\right)\
\left(-\frac{1}{(1+s_{kl})^2}\right)
(2\Delta_{kl})(\delta_{km}-\delta_{lm})
\end{aligned}
$$

The factors involving $s_{kl}$ and $\Delta_{kl}$ depend on $k,l$, but not on $i,j$. Therefore we can rearrange the summations to simplify the sum over $i,j$:

$$
\begin{aligned}
\frac{\partial C}{\partial y_m}
={}&\sum_{k,l}
\left[
\sum_{i,j}\left(-\frac{p_{ij}}{q_{ij}}\right)
\left(\frac{\delta_{ik}\delta_{jl}}{A}
-\frac{z_{ij}}{A^2}\right)
\right]\\
&\qquad\cdot
\left(-\frac{1}{(1+s_{kl})^2}\right)
(2\Delta_{kl})(\delta_{km}-\delta_{lm})
\end{aligned}
$$

**Step 7: Simplify the sums.**

Distribute the factor $-p_{ij}/q_{ij}$ across:

$$
\begin{aligned}
&\sum_{i,j}\left(-\frac{p_{ij}}{q_{ij}}\right)
\left(\frac{\delta_{ik}\delta_{jl}}{A}
-\frac{z_{ij}}{A^2}\right)\\
&\quad=
-\sum_{i,j}\frac{p_{ij}}{q_{ij}}
\frac{\delta_{ik}\delta_{jl}}{A}
+\sum_{i,j}\frac{p_{ij}}{q_{ij}}\frac{z_{ij}}{A^2}
\end{aligned}
$$

In the first sum, the kronecker deltas make every term zero except the term where $i=k$ and $j=l$. Therefore, the first sum becomes

$$
-\frac{p_{kl}}{q_{kl}A}
$$

For the second sum, from $q_{ij}=z_{ij}/A$ -> $z_{ij}/q_{ij} = A$ and $\sum_{i,j}p_{ij}=1$. Therefore:

$$
\begin{aligned}
\sum_{i,j}\frac{p_{ij}}{q_{ij}}\frac{z_{ij}}{A^2}
&=\sum_{i,j}\frac{p_{ij}A}{A^2}\\
&=\frac{1}{A}\sum_{i,j}p_{ij}\\
&=\frac{1}{A}
\end{aligned}
$$

Combining the two parts, the sum over $i,j$ simplifies to

$$
-\frac{p_{kl}}{q_{kl}A}+\frac{1}{A}
$$

Since $q_{kl}=z_{kl}/A$, we have $q_{kl}A=z_{kl}$. Thus,

$$
-\frac{p_{kl}}{q_{kl}A}+\frac{1}{A}
=-\frac{p_{kl}}{z_{kl}}+\frac{1}{A}
$$

And because $q_{kl}/z_{kl}=1/A$,

$$
\boxed{
\sum_{i,j}\frac{\partial C}{\partial q_{ij}}
\frac{\partial q_{ij}}{\partial z_{kl}}
=\frac{q_{kl}-p_{kl}}{z_{kl}}
}
$$

The chain rule now becomes,

$$
\frac{\partial C}{\partial y_m}
=
\sum_{k,l}\frac{q_{kl}-p_{kl}}{z_{kl}}
\left(-\frac{1}{(1+s_{kl})^2}\right)
(2\Delta_{kl})(\delta_{km}-\delta_{lm})
$$
$z_{kl}=1/(1+s_{kl})$. Thus, $1/z_{kl}=1+s_{kl}$. We use this to simplify the product involving $z_{kl}$:

$$
\begin{aligned}
\frac{1}{z_{kl}}
\left(-\frac{1}{(1+s_{kl})^2}\right)
&=-(1+s_{kl})\frac{1}{(1+s_{kl})^2}\\
&=-\frac{1}{1+s_{kl}}\\
&=-\frac{1}{1+\Delta_{kl}^2}
\end{aligned}
$$
The chain rule now becomes,

$$
\begin{aligned}
\frac{\partial C}{\partial y_m}
&=-2\sum_{k,l}(q_{kl}-p_{kl})
\frac{\Delta_{kl}}{1+\Delta_{kl}^2}
(\delta_{km}-\delta_{lm})\
\end{aligned}
$$

$$
\boxed{
\frac{\partial C}{\partial y_m}
=2\sum_{k,l}(p_{kl}-q_{kl})
\frac{\Delta_{kl}}{1+\Delta_{kl}^2}
(\delta_{km}-\delta_{lm})
}
$$