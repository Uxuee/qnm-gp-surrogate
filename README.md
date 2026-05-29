# QNM Gaussian Process Surrogate

Gaussian Process surrogate modelling for fast parameter search in black-hole
ringdown shifts.

This project implements a sparse-data, two-output version of the idea:

```text
(w, Q) -> (deltaOmega, deltaLambda) -> omega_QNM
```

The code computes photon-sphere observables from a small fluid-inspired
perturbation of Schwarzschild, trains two uncertainty-aware Gaussian Process
models on a limited number of simulated points, compares against baseline
regressors, and then uses the learned surrogate to infer the physical
parameters that reproduce a target complex quasinormal-mode frequency.

## Motivation

Ringdown observables can be expensive to evaluate if every point in parameter
space requires solving the full physical model. A surrogate model gives a fast
approximation while also estimating where it is uncertain.

Here the input parameters are:

- `w`: equation-of-state/profile parameter.
- `Q`: perturbation strength.

The learned targets are:

- `deltaOmega`: shift in photon-sphere orbital frequency.
- `deltaLambda`: shift in the Lyapunov exponent of the unstable null orbit.

Those are converted into the eikonal QNM estimate

```text
omega_QNM = ell * Omega_c - i * (n + 1/2) * lambda_c
```

where `Omega_c` is the circular null-orbit frequency, `lambda_c` is the
Lyapunov exponent, `ell` is the angular mode, and `n` is the overtone number.

## What the Script Does

1. Defines a perturbative black-hole metric model.
2. Solves the photon-sphere condition:

   ```text
   r A'(r) - 2 A(r) = 0
   ```

3. Computes `Omega_c`, `lambda_c`, and the complex eikonal QNM frequency.
4. Generates a grid dataset over `(w, Q)`.
5. Trains two Gaussian Process regressors on a sparse subset of the grid:

   ```text
   GP_deltaOmega(w, Q)
   GP_deltaLambda(w, Q)
   ```

6. Tests on withheld grid points.
7. Compares the GP against linear, polynomial, and random-forest baselines.
8. Produces prediction, uncertainty, and actual-error maps.
9. Builds a learning curve versus number of simulations.
10. Performs inverse parameter search from a target QNM frequency.

## Repository Layout

```text
.
|-- qnm_surrogate.py          # Main physics + ML pipeline
|-- requirements.txt          # Python dependencies
|-- CITATION.cff              # Citation metadata for GitHub
|-- LICENSE                   # MIT license
|-- outputs/
|   |-- qnm_dataset.csv       # Generated training/evaluation grid
|   |-- gp_metrics.csv        # Test-set metrics
|   |-- baseline_metrics.csv  # GP versus baseline regressors
|   |-- learning_curve.csv    # Error versus number of training simulations
|   |-- parameter_search.csv  # Inverse QNM parameter recovery result
|   `-- *.png                 # Prediction and uncertainty maps
`-- README.md
```

## Quick Start

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the full pipeline:

```powershell
python qnm_surrogate.py
```

The script regenerates the dataset, trains sparse-data GP models, compares
baselines, runs parameter recovery, and writes all outputs to `outputs/`.

## Example Results

The current main run trains on only 80 simulations and tests on the remaining
496 grid points.

| Target | RMSE | MAE | R2 | Mean GP sigma |
| --- | ---: | ---: | ---: | ---: |
| `deltaOmega` | `2.64e-06` | `1.01e-06` | `0.9999998743` | `2.34e-06` |
| `deltaLambda` | `2.61e-06` | `9.91e-07` | `0.9999998448` | `2.25e-06` |

Baseline comparison with the same 80 training simulations:

| Target | Model | RMSE | R2 |
| --- | --- | ---: | ---: |
| `deltaOmega` | Gaussian Process | `2.64e-06` | `0.9999998743` |
| `deltaOmega` | Linear regression | `1.01e-03` | `0.981675` |
| `deltaOmega` | Polynomial degree 3 | `4.60e-05` | `0.999962` |
| `deltaOmega` | Random forest | `8.82e-04` | `0.985951` |
| `deltaLambda` | Gaussian Process | `2.61e-06` | `0.9999998448` |
| `deltaLambda` | Linear regression | `1.34e-03` | `0.959115` |
| `deltaLambda` | Polynomial degree 3 | `4.35e-05` | `0.999957` |
| `deltaLambda` | Random forest | `1.03e-03` | `0.975866` |

Inverse search example:

| Quantity | Value |
| --- | ---: |
| Target `w` | `0.630000` |
| Target `Q` | `0.072000` |
| Recovered `w` | `0.630000` |
| Recovered `Q` | `0.072000` |
| Target `Re(omega)` | `0.420320` |
| Target `Im(omega)` | `-0.104041` |

The target is off-grid relative to the sparse training subset, so this is a
surrogate-based inverse search rather than a lookup.

## Figures

### GP Prediction for `deltaOmega`

![GP prediction for deltaOmega](outputs/deltaOmega_prediction.png)

### GP Uncertainty for `deltaOmega`

![GP uncertainty for deltaOmega](outputs/deltaOmega_uncertainty.png)

### Absolute Error for `deltaOmega`

![Absolute error for deltaOmega](outputs/deltaOmega_error.png)

### GP Prediction for `deltaLambda`

![GP prediction for deltaLambda](outputs/deltaLambda_prediction.png)

### GP Uncertainty for `deltaLambda`

![GP uncertainty for deltaLambda](outputs/deltaLambda_uncertainty.png)

### Absolute Error for `deltaLambda`

![Absolute error for deltaLambda](outputs/deltaLambda_error.png)

### Learning Curve

![Sparse-data learning curve](outputs/learning_curve.png)

## Scientific Status

This is a research-prototype pipeline. The machine-learning and QNM workflow is
real, but the default metric perturbation in `metric_A` is intentionally a toy
model. It is designed to be easy to replace with the exact perturbative metric
functions from the paper or from a symbolic/numerical solver.

The current dataset is synthetic/model-generated. The goal is to demonstrate
the surrogate-modelling workflow before replacing the toy equations with more
expensive numerical simulations.

The important reusable structure is:

```text
metric model -> photon sphere -> Omega/lambda -> QNM -> GP surrogate -> inverse search
```

## How to Connect It to the Paper

Replace the implementation of `metric_A` and, if needed, `metric_B` in
`qnm_surrogate.py` with the metric functions from the theoretical model. The
rest of the pipeline can remain almost unchanged:

1. solve the circular null-orbit condition;
2. compute `Omega_c`;
3. compute `lambda_c`;
4. train one GP for `deltaOmega`;
5. train one GP for `deltaLambda`;
6. infer `(w, Q)` from a target complex QNM.

## Next Improvements

- Replace the toy perturbation with the exact metric from the paper.
- Add active learning: start with a small training set, then sample where GP
  uncertainty is largest.
- Add cross-validation across different grid resolutions.
- Save trained models with `joblib`.
- Add a notebook version for presentation and plots.

## License

MIT License. See `LICENSE`.
