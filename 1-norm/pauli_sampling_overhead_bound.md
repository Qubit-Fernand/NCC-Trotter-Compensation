# Strict-Paired Channel Pauli Sampling Overhead Bound

This note records the strict-paired channel-level Pauli-sampling 1-norm floor
used by the scaling notebooks.  Every retained single \(\Phi_q\) is identity
paired; only multi-commutator Taylor terms remain as direct sampling tails.

## Archived Historical Tables

The files under `archive/` are historical references only.  Some used the
wrong pairing-angle conversion, namely a one-step channel pair scale

```tex
\sqrt{1+4\eta(x)^2},
```

which corresponds to the previous incorrect equation
`sin(theta)/cos(theta)^2 = eta`.  The active channel sampler now uses the
unbiased condition

```tex
\tan(\theta)=\eta.
```

Under this corrected condition, the signed channel-pair normalization is

```tex
1+2\eta(x)^2.
```

The `correct_pair_theta_reference` files in the archive fix this angle but
still use the old \(K=1,s_0=4\) convention in which \(\Phi_4\) is a direct
tail rather than a paired single term.  Do not use any archived CSV for new
strict-paired plots or bounds.

## Basic Principle

For one compensated Trotter step with step size

```tex
x = T / r,
```

let `W(x)` be the one-step sampling 1-norm of the strict-paired channel
sampler.
After `r` steps, the total sampling overhead is

```tex
\ell = W(T/r)^r.
```

The current floor tables impose the stricter cap

```tex
W(T/r)^r \le 1.75.
```

Equivalently,

```tex
r \log W(T/r) \le \log(1.75).
```

The active `cap=1.75` channel-floor CSVs are:

```text
pauli_sampling_overhead_rmin_N8_cap_1p75.csv
pauli_sampling_overhead_rmin_N_sweep_T2_cap_1p75.csv
```

To reproduce the active tables (and to generate another cap), run

```bash
python 1-norm/recompute_strict_paired_channel_floors.py --cap 1.75
```

The generator obtains the exact sparse-Pauli decompositions of
\(\Phi_2,\Phi_3,\Phi_4\), then scans integer \(r\) directly for the first
value satisfying the bound.  It does not rely on a binary-search monotonicity
assumption.

## N = 8 Heisenberg Parameters

For the periodic one-dimensional Heisenberg model used in the numerical
experiments,

```tex
N=8,\quad J=h=1,\quad K=1.
```

The paired low-order channel weight is

```tex
\eta(x)=64x^2+416x^3.
```

## Original NCC, \(K=1,s_0=3\)

Original NCC pairs the only retained single terms, \(\Phi_2,\Phi_3\).  With
`tan(theta)=eta`, its one-step sampling norm is

```tex
W_{\rm ch}^{\rm orig}(x)
=
1+2\eta(x)^2.
```

## HNCC, Strict Pairing, \(K=1,q_0=s_0=4\)

Strict HNCC pairs \(\Phi_2,\Phi_3,\Phi_4\) together.  The only direct tail
through total order four is

```tex
\frac{1}{2!}\operatorname{ad}_{\Phi_2}^2,
```

whose executable channel-sampling 1-norm is \(2a_2^2x^4\).  Hence, with
\(a_q=\|\Phi_q\|_{\mathrm{Pauli},1}\),

```tex
W_{\rm ch}^{\rm HNCC,strict}(x)
=
1+2\bigl(a_2x^2+a_3x^3+a_4x^4\bigr)^2+2a_2^2x^4.
```

For \(N=8\), the exact coefficients are

```tex
a_2=64,\qquad a_3=416,\qquad a_4=\frac{4576}{3},
```

so the direct tail is \(8192x^4\).  This replaces the old non-strict formula
\(1+2(a_2x^2+a_3x^3)^2+(2a_4+2a_2^2)x^4\), which treated \(\Phi_4\) as a
tail.

The corrected `N=8` channel floors under `W(T/r)^r <= 1.75` are:

| T | original channel r_min | strict HNCC channel r_min |
|---:|---:|---:|
| 0.2 | 4 | 4 |
| 0.4 | 9 | 10 |
| 0.6 | 15 | 17 |
| 0.8 | 21 | 25 |
| 1.0 | 29 | 34 |
| 1.2 | 36 | 42 |
| 1.4 | 44 | 52 |
| 1.6 | 52 | 62 |
| 1.8 | 61 | 72 |
| 2.0 | 70 | 83 |

For the N-sweep at fixed `T=2.0`, using the N-dependent Pauli 1-norm
coefficients gives:

| N | original channel r_min | strict HNCC channel r_min |
|---:|---:|---:|
| 4 | 45 | 53 |
| 5 | 46 | 53 |
| 6 | 59 | 69 |
| 7 | 58 | 69 |
| 8 | 70 | 83 |
