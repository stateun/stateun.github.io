---
layout: single
title: "CTGAN Code Review"
date: 2025-01-04
author_profile: true
use_math: true
categories:
  - Math
tags:
  - Lipschitz Condition
  - Gumbel Softmax
  - Variational Gaussian Mixture Model
permalink: /CTGAN-1/
---

## Lipschitz Condition

\[
\textbf{Definition (Lipschitz Condition).} \\
\text{Given a function } f: \mathcal{X} \rightarrow \mathbb{R}, \text{ for all } x, y \in \mathcal{X}, \\
\quad |f(x) - f(y)| \leq K \cdot \|x - y\| \\
\text{if this inequality holds, then } f \text{ is called a } K\text{-Lipschitz function.}
\]

- \( K \in \mathbb{R}_{\geq 0} \) is the Lipschitz constant.
- \(\|x-y\|\) represents the distance between \(x\) and \(y\) (e.g., Euclidean distance).

---

## 2. Gumbel Softmax

\[
\textbf{Definition 2 (Gumbel-Softmax).} \\
\text{Suppose we want to sample from a categorical distribution } \mathrm{Cat}(\pi_1, \ldots, \pi_K). \\
\text{For each class } i \in \{1, \ldots, K\}, \\
g_i = -\log(-\log(u_i)), \quad u_i \sim \mathrm{Uniform}(0, 1) \\
\text{is defined as independent Gumbel noise } \{g_i\}_{i=1}^K. \\
\text{Then, using a temperature parameter } \tau > 0, \text{ we define } \\
y_i = \frac{\exp\Bigl(\bigl(\log \pi_i + g_i\bigr) / \tau\Bigr)}
{\sum_{j=1}^K \exp\Bigl(\bigl(\log \pi_j + g_j\bigr) / \tau\Bigr)}.
\]
\[
\text{Here, } y = (y_1, \dots, y_K) \text{ can be interpreted as a 'soft' sample vector resembling a one-hot vector.}
\]

---

## 3. Variational Gaussian Mixture Model (V-GMM)

\[
\textbf{Definition 3 (Variational Gaussian Mixture Model).} \\
\text{Let a dataset } X = \{x_1, \ldots, x_N\} \subset \mathbb{R}^D \text{ be given.} \\
\text{A Gaussian Mixture Model (GMM) consists of } K \text{ components with } \\
\text{mixing coefficients } \{\pi_k\}_{k=1}^K, \;
\text{means } \{\mu_k\}_{k=1}^K, \;
\text{and covariances } \{\Sigma_k\}_{k=1}^K. \\
\text{For each data point } x_n, \text{ the probability density is defined as: } \\
p(x_n \mid \pi, \mu, \Sigma) 
= \sum_{k=1}^K \pi_k \,\mathcal{N}(x_n \mid \mu_k, \Sigma_k).
\]
\[
\text{Here, } \sum_{k=1}^K \pi_k = 1, \;\; \pi_k \geq 0.
\]
\[
\text{From a Bayesian perspective, priors are assigned to the parameters } (\pi, \{\mu_k\}, \{\Sigma_k\}), \\
\text{and a latent variable } z_n \in \{1, \ldots, K\} \text{ is introduced for each } x_n \text{ such that: } \\
p(z_n = k) = \pi_k, \quad p(x_n \mid z_n = k) = \mathcal{N}(x_n \mid \mu_k, \Sigma_k).
\]
\[
\text{However, since directly computing the posterior } p(\pi, \mu, \Sigma, z \mid X) \text{ is intractable,} \\
\text{we use variational inference to approximate it.}
\]
\[
\text{Specifically, we introduce } q(\pi, \mu, \Sigma, z) \text{ such that:} \\
q(\pi, \mu, \Sigma, z) \approx p(\pi, \mu, \Sigma, z \mid X).
\]
\[
\text{We optimize the Evidence Lower Bound (ELBO) } \mathcal{L}(q) \text{ to learn } q:
\]
\[
\mathcal{L}(q) 
= \mathbb{E}_{q}\bigl[\log p(X, \pi, \mu, \Sigma, z)\bigr]
- \mathbb{E}_{q}\bigl[\log q(\pi, \mu, \Sigma, z)\bigr].
\]
\[
\text{Through this process, } q \text{ is guided to approximate the true posterior.}
\]
