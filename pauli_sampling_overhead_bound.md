# Pauli Sampling Overhead Bound

This note records the sampling-overhead bound used for the numerical
Pauli-distribution sampler, as opposed to the light-cone rejection-sampling
bound in the main theorem.

## Basic Principle

For one compensated Trotter step with step size

```tex
x = T / r,
```

let `W(x)` be the actual one-step sampling 1-norm of the implemented Pauli
sampler. After `r` steps, the total sampling overhead is

```tex
\ell = W(T/r)^r.
```

To keep the Monte Carlo repetition count at the standard scale, we impose the
constant-overhead condition

```tex
W(T/r)^r \le e.
```

Equivalently, for numerical evaluation,

```tex
r \log W(T/r) \le 1.
```

This avoids replacing `1+x` by `exp(x)` and gives a slightly tighter minimum
step count.

## N = 8 Heisenberg Parameters

For the periodic one-dimensional Heisenberg model used in the numerical
experiments,

```tex
N=8,\quad J=h=1,\quad K=1.
```

Direct Pauli-string calculation gives

```tex
\|\Phi_2\|_{P,1}=64,\qquad
\|\Phi_3\|_{P,1}=416,\qquad
\|\Phi_4\|_{P,1}=\frac{4576}{3}.
```

Define

```tex
\eta(x)=64x^2+416x^3.
```

## Original NCC, s0 = 3

The original NCC sampler keeps the paired orders `s=2,3`. The operator-level
one-step norm is

```tex
W_{\rm op}^{\rm orig}(x)
=
\sqrt{1+\eta(x)^2}.
```

The channel-level one-step norm is

```tex
W_{\rm ch}^{\rm orig}(x)
=
\sqrt{1+4\eta(x)^2}.
```

The corresponding minimum `r` from `W(T/r)^r <= e` is:

| T | original operator r_min | original channel r_min |
|---:|---:|---:|
| 0.2 | 2 | 3 |
| 0.4 | 5 | 7 |
| 0.6 | 9 | 12 |
| 0.8 | 12 | 18 |
| 1.0 | 16 | 24 |
| 1.2 | 20 | 30 |
| 1.4 | 25 | 37 |
| 1.6 | 29 | 44 |
| 1.8 | 34 | 51 |
| 2.0 | 39 | 58 |

## Log-Precision NCC, Channel s0 = 4

For the channel-level log-precision experiments with `q0=s0=4`, the channel
sampler has

```tex
W_{\rm ch}^{(s_0=4)}(x)
=
\sqrt{1+4\eta(x)^2}
+
\frac{33728}{3}x^4.
```

The minimum `r` from `W(T/r)^r <= e` is:

| T | channel log r_min |
|---:|---:|
| 0.2 | 4 |
| 0.4 | 9 |
| 0.6 | 15 |
| 0.8 | 22 |
| 1.0 | 29 |
| 1.2 | 37 |
| 1.4 | 45 |
| 1.6 | 53 |
| 1.8 | 62 |
| 2.0 | 72 |

For the left N-sweep plot at fixed `T=2.0`, the same calculation with the
N-dependent Pauli 1-norm coefficients gives the following floors:

| N | original operator r_min | operator log r_min | original channel r_min | channel log r_min |
|---:|---:|---:|---:|---:|
| 4 | 26 | 30 | 38 | 46 |
| 5 | 26 | 31 | 38 | 47 |
| 6 | 33 | 41 | 49 | 61 |
| 7 | 33 | 40 | 49 | 60 |
| 8 | 39 | 48 | 58 | 72 |

## Log-Precision NCC, Operator s0 = 6

For the operator-level log-precision experiments with `q0=s0=6`, the exact
operator-level Pauli 1-norm coefficients are

```tex
\|\tilde F_2\|_{P,1}=64,\qquad
\|\tilde F_3\|_{P,1}=416,\qquad
\|\tilde F_4\|_{P,1}=\frac{8320}{3},
```

```tex
\|\tilde F_5\|_{P,1}=15123.2,\qquad
\|\tilde F_6\|_{P,1}\approx 76110.9333333291.
```

Thus

```tex
W_{\rm op}^{(s_0=6)}(x)
=
\sqrt{1+\eta(x)^2}
+
\frac{8320}{3}x^4
+
15123.2x^5
+
76110.9333333291x^6.
```

The minimum `r` from `W(T/r)^r <= e` is:

| T | operator log s0=6 r_min |
|---:|---:|
| 0.2 | 3 |
| 0.4 | 6 |
| 0.6 | 10 |
| 0.8 | 15 |
| 1.0 | 20 |
| 1.2 | 25 |
| 1.4 | 30 |
| 1.6 | 36 |
| 1.8 | 42 |
| 2.0 | 48 |
