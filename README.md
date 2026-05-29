# Gaussian-Process Surrogate for Black-Hole QNM Shifts

This implements the stronger "second version" of the project:

```text
(w, Q) -> (deltaOmega, deltaLambda) -> omega_QNM
```

The forward model computes the unstable circular null orbit of a small
fluid-inspired perturbation of Schwarzschild, then uses the eikonal relation

```text
omega_QNM = ell * Omega_c - i * (n + 1/2) * lambda_c
```

where `Omega_c` is the photon-sphere orbital frequency and `lambda_c` is the
Lyapunov exponent. Two Gaussian Process regressors learn `deltaOmega` and
`deltaLambda`, including predictive uncertainty.

## Run

```powershell
python qnm_surrogate.py
```

The script writes:

- `outputs/qnm_dataset.csv`
- `outputs/gp_metrics.csv`
- `outputs/parameter_search.csv`
- `outputs/deltaOmega_prediction.png`
- `outputs/deltaOmega_uncertainty.png`
- `outputs/deltaLambda_prediction.png`
- `outputs/deltaLambda_uncertainty.png`

## What to replace later

The function `metric_A` is the only intentionally toy part. Replace it with the
exact perturbative metric functions from the paper or from your Mathematica
solver. The rest of the pipeline can stay the same:

1. solve the photon-sphere equation `r A'(r) - 2 A(r) = 0`;
2. compute `Omega_c`;
3. compute `lambda_c`;
4. train two GPs;
5. infer `(w, Q)` from a target complex QNM frequency.
