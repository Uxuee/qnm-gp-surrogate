from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MetricParams:
    """Toy perturbative black-hole environment inspired by a fluid-dressed metric."""

    mass: float = 1.0
    r0: float = 6.0


@dataclass(frozen=True)
class QNMMode:
    ell: int = 2
    overtone: int = 0


def metric_A(r: np.ndarray | float, w: float, Q: float, params: MetricParams) -> np.ndarray | float:
    """Time-time metric function A(r) for ds^2=-A dt^2 + B dr^2 + r^2 dOmega^2.

    The perturbation is deliberately small and smooth.  It gives the ML pipeline a
    physics-shaped forward model while keeping the implementation easy to replace
    with exact formulas from a symbolic or numerical relativity solver later.
    """

    M = params.mass
    r_arr = np.asarray(r)
    schwarzschild = 1.0 - 2.0 * M / r_arr
    # Let w affect both the amplitude and radial falloff.  This avoids a
    # one-combination degeneracy Q*f(w), making inverse QNM search nontrivial.
    falloff_scale = params.r0 * (0.72 + 0.55 * w)
    profile = (2.0 * M / r_arr) * np.exp(-(r_arr - 2.0 * M) / falloff_scale)
    eos_shape = 1.0 + 0.35 * w + 0.12 * np.sin(np.pi * w)
    curvature_shape = 1.0 + 0.08 * w * (r_arr - 3.0 * M) / M
    return schwarzschild + Q * eos_shape * curvature_shape * profile


def metric_B(r: np.ndarray | float, w: float, Q: float, params: MetricParams) -> np.ndarray | float:
    return 1.0 / metric_A(r, w, Q, params)


def derivative(fun, x: float, step: float = 1.0e-4) -> float:
    return (fun(x + step) - fun(x - step)) / (2.0 * step)


def second_derivative(fun, x: float, step: float = 1.0e-3) -> float:
    return (fun(x + step) - 2.0 * fun(x) + fun(x - step)) / step**2


def photon_sphere_radius(w: float, Q: float, params: MetricParams) -> float:
    """Solve r A'(r)-2 A(r)=0 for the unstable circular null orbit."""

    M = params.mass

    def equation(r: float) -> float:
        A = lambda x: metric_A(x, w, Q, params)
        return r * derivative(A, r) - 2.0 * A(r)

    # The root stays near r=3M for the small perturbations used here.
    scan = np.linspace(2.05 * M, 12.0 * M, 400)
    values = np.array([equation(r) for r in scan])
    crossings = np.where(np.signbit(values[:-1]) != np.signbit(values[1:]))[0]
    if len(crossings) == 0:
        raise RuntimeError(f"No photon-sphere root found for w={w:.3g}, Q={Q:.3g}")
    i = crossings[0]
    return brentq(equation, scan[i], scan[i + 1])


def qnm_ingredients(w: float, Q: float, params: MetricParams) -> tuple[float, float, float]:
    """Return photon-sphere radius, orbital frequency Omega_c, and Lyapunov exponent."""

    rc = photon_sphere_radius(w, Q, params)
    Ac = metric_A(rc, w, Q, params)
    Bc = metric_B(rc, w, Q, params)
    L2 = rc**2 / Ac
    tdot = 1.0 / Ac

    def radial_rhs(r: float) -> float:
        A = metric_A(r, w, Q, params)
        B = metric_B(r, w, Q, params)
        return (1.0 / B) * (1.0 / A - L2 / r**2)

    omega_c = np.sqrt(Ac / rc**2)
    lambda_sq = second_derivative(radial_rhs, rc) / (2.0 * tdot**2)
    lambda_c = np.sqrt(max(lambda_sq, 0.0))
    return rc, omega_c, lambda_c


def eikonal_qnm(omega_c: float, lambda_c: float, mode: QNMMode) -> complex:
    """omega_QNM = ell Omega_c - i (n + 1/2) lambda_c."""

    return mode.ell * omega_c - 1j * (mode.overtone + 0.5) * lambda_c


def generate_dataset(grid_size: int, params: MetricParams, mode: QNMMode) -> pd.DataFrame:
    rows = []
    _, omega0, lambda0 = qnm_ingredients(w=0.0, Q=0.0, params=params)
    qnm0 = eikonal_qnm(omega0, lambda0, mode)

    for w in np.linspace(0.0, 1.0, grid_size):
        for Q in np.linspace(0.0, 0.1, grid_size):
            rc, omega_c, lambda_c = qnm_ingredients(w=w, Q=Q, params=params)
            qnm = eikonal_qnm(omega_c, lambda_c, mode)
            rows.append(
                {
                    "w": w,
                    "Q": Q,
                    "r_ph": rc,
                    "Omega": omega_c,
                    "lambda": lambda_c,
                    "deltaOmega": omega_c - omega0,
                    "deltaLambda": lambda_c - lambda0,
                    "omega_re": qnm.real,
                    "omega_im": qnm.imag,
                    "delta_omega_re": qnm.real - qnm0.real,
                    "delta_omega_im": qnm.imag - qnm0.imag,
                }
            )

    return pd.DataFrame(rows)


def build_gp(random_state: int = 7):
    kernel = (
        ConstantKernel(1.0, (1.0e-3, 1.0e3))
        * Matern(length_scale=[0.35, 0.035], length_scale_bounds=(1.0e-3, 100.0), nu=2.5)
        + WhiteKernel(noise_level=1.0e-8, noise_level_bounds=(1.0e-12, 1.0e-5))
    )
    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=8,
            random_state=random_state,
        ),
    )


def train_models(df: pd.DataFrame, random_state: int = 7):
    X = df[["w", "Q"]].to_numpy()
    targets = {
        "deltaOmega": df["deltaOmega"].to_numpy(),
        "deltaLambda": df["deltaLambda"].to_numpy(),
    }
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=random_state
    )

    models = {}
    metrics = []
    for name, y in targets.items():
        model = build_gp(random_state=random_state)
        model.fit(X[train_idx], y[train_idx])
        pred, std = model.predict(X[test_idx], return_std=True)
        models[name] = model
        metrics.append(
            {
                "target": name,
                "rmse": mean_squared_error(y[test_idx], pred, squared=False),
                "mae": mean_absolute_error(y[test_idx], pred),
                "r2": r2_score(y[test_idx], pred),
                "mean_gp_sigma": float(np.mean(std)),
            }
        )

    return models, pd.DataFrame(metrics), train_idx, test_idx


def predict_qnm(
    models: dict[str, object],
    w: float,
    Q: float,
    params: MetricParams,
    mode: QNMMode,
) -> tuple[complex, float, float]:
    _, omega0, lambda0 = qnm_ingredients(w=0.0, Q=0.0, params=params)
    x = np.array([[w, Q]])
    d_omega, s_omega = models["deltaOmega"].predict(x, return_std=True)
    d_lambda, s_lambda = models["deltaLambda"].predict(x, return_std=True)
    qnm = eikonal_qnm(omega0 + d_omega[0], lambda0 + d_lambda[0], mode)
    return qnm, s_omega[0], s_lambda[0]


def infer_parameters_from_qnm(
    models: dict[str, object],
    target_qnm: complex,
    params: MetricParams,
    mode: QNMMode,
) -> dict[str, float]:
    def objective(x: np.ndarray) -> float:
        qnm, s_omega, s_lambda = predict_qnm(models, x[0], x[1], params, mode)
        residual = abs(qnm - target_qnm) ** 2
        uncertainty_penalty = 0.05 * (s_omega**2 + s_lambda**2)
        return float(residual + uncertainty_penalty)

    result = differential_evolution(objective, bounds=[(0.0, 1.0), (0.0, 0.1)], seed=11)
    qnm_pred, s_omega, s_lambda = predict_qnm(models, result.x[0], result.x[1], params, mode)
    return {
        "w": result.x[0],
        "Q": result.x[1],
        "objective": result.fun,
        "omega_re_pred": qnm_pred.real,
        "omega_im_pred": qnm_pred.imag,
        "sigma_deltaOmega": s_omega,
        "sigma_deltaLambda": s_lambda,
    }


def make_plots(df: pd.DataFrame, models: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    w_grid = np.linspace(0.0, 1.0, 120)
    q_grid = np.linspace(0.0, 0.1, 120)
    W, Q = np.meshgrid(w_grid, q_grid)
    Xplot = np.column_stack([W.ravel(), Q.ravel()])

    for target in ["deltaOmega", "deltaLambda"]:
        pred, std = models[target].predict(Xplot, return_std=True)
        Z = pred.reshape(W.shape)
        S = std.reshape(W.shape)

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        im = ax.pcolormesh(W, Q, Z, shading="auto", cmap="viridis")
        ax.scatter(df["w"], df["Q"], s=6, c="white", alpha=0.45, linewidths=0)
        ax.set_xlabel("w")
        ax.set_ylabel("Q")
        ax.set_title(f"GP prediction for {target}")
        fig.colorbar(im, ax=ax)
        fig.savefig(out_dir / f"{target}_prediction.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        im = ax.pcolormesh(W, Q, S, shading="auto", cmap="magma")
        ax.set_xlabel("w")
        ax.set_ylabel("Q")
        ax.set_title(f"GP uncertainty for {target}")
        fig.colorbar(im, ax=ax)
        fig.savefig(out_dir / f"{target}_uncertainty.png", dpi=180)
        plt.close(fig)


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    params = MetricParams()
    mode = QNMMode(ell=2, overtone=0)

    df = generate_dataset(grid_size=24, params=params, mode=mode)
    df.to_csv(out_dir / "qnm_dataset.csv", index=False)

    models, metrics, _, _ = train_models(df)
    metrics.to_csv(out_dir / "gp_metrics.csv", index=False)

    true_w, true_Q = 0.63, 0.072
    _, omega_true, lambda_true = qnm_ingredients(true_w, true_Q, params)
    target_qnm = eikonal_qnm(omega_true, lambda_true, mode)
    inference = infer_parameters_from_qnm(models, target_qnm, params, mode)
    pd.DataFrame(
        [
            {
                "target_w": true_w,
                "target_Q": true_Q,
                "target_omega_re": target_qnm.real,
                "target_omega_im": target_qnm.imag,
                **inference,
            }
        ]
    ).to_csv(out_dir / "parameter_search.csv", index=False)

    make_plots(df, models, out_dir)

    print("Wrote:")
    print(f"  {out_dir / 'qnm_dataset.csv'}")
    print(f"  {out_dir / 'gp_metrics.csv'}")
    print(f"  {out_dir / 'parameter_search.csv'}")
    print("  outputs/*_prediction.png")
    print("  outputs/*_uncertainty.png")
    print()
    print(metrics.to_string(index=False))
    print()
    print(pd.read_csv(out_dir / "parameter_search.csv").to_string(index=False))


if __name__ == "__main__":
    main()
