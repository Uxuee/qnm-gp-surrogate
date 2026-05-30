"""Legacy toy demonstration.

This script uses an artificial metric perturbation and is not the
paper-connected model. The main scientific script is `kiselev_gp_surrogate.py`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


@dataclass(frozen=True)
class MetricParams:
    """Toy perturbative black-hole environment inspired by a fluid-dressed metric."""

    mass: float = 1.0
    r0: float = 6.0


@dataclass(frozen=True)
class QNMMode:
    ell: int = 4
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
            n_restarts_optimizer=4,
            random_state=random_state,
        ),
    )


def make_sparse_split(
    df: pd.DataFrame,
    n_train: int = 80,
    random_state: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    if n_train >= len(df):
        raise ValueError("n_train must be smaller than the full dataset size.")

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), train_size=n_train, random_state=random_state
    )
    return train_idx, test_idx


def train_models(
    df: pd.DataFrame,
    n_train: int = 80,
    random_state: int = 7,
):
    X = df[["w", "Q"]].to_numpy()
    targets = {
        "deltaOmega": df["deltaOmega"].to_numpy(),
        "deltaLambda": df["deltaLambda"].to_numpy(),
    }
    train_idx, test_idx = make_sparse_split(df, n_train=n_train, random_state=random_state)

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
                "model": "GaussianProcess",
                "n_train": n_train,
                "n_test": len(test_idx),
                "rmse": mean_squared_error(y[test_idx], pred, squared=False),
                "mae": mean_absolute_error(y[test_idx], pred),
                "r2": r2_score(y[test_idx], pred),
                "mean_gp_sigma": float(np.mean(std)),
            }
        )

    return models, pd.DataFrame(metrics), train_idx, test_idx


def baseline_models(random_state: int = 7) -> dict[str, object]:
    return {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "PolynomialDegree3": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=3, include_bias=False),
            LinearRegression(),
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
        ),
    }


def compare_baselines(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    gp_models: dict[str, object],
    random_state: int = 7,
) -> pd.DataFrame:
    X = df[["w", "Q"]].to_numpy()
    targets = {
        "deltaOmega": df["deltaOmega"].to_numpy(),
        "deltaLambda": df["deltaLambda"].to_numpy(),
    }
    rows = []

    for target, y in targets.items():
        models = {"GaussianProcess": gp_models[target], **baseline_models(random_state)}
        for model_name, model in models.items():
            if model_name != "GaussianProcess":
                model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "rmse": mean_squared_error(y[test_idx], pred, squared=False),
                    "mae": mean_absolute_error(y[test_idx], pred),
                    "r2": r2_score(y[test_idx], pred),
                }
            )

    return pd.DataFrame(rows)


def learning_curve(
    df: pd.DataFrame,
    train_sizes: list[int],
    random_state: int = 7,
) -> pd.DataFrame:
    X = df[["w", "Q"]].to_numpy()
    targets = {
        "deltaOmega": df["deltaOmega"].to_numpy(),
        "deltaLambda": df["deltaLambda"].to_numpy(),
    }
    rows = []

    for n_train in train_sizes:
        train_idx, test_idx = make_sparse_split(df, n_train=n_train, random_state=random_state)
        for target, y in targets.items():
            model = build_gp(random_state=random_state)
            model.fit(X[train_idx], y[train_idx])
            pred, std = model.predict(X[test_idx], return_std=True)
            rows.append(
                {
                    "target": target,
                    "n_train": n_train,
                    "n_test": len(test_idx),
                    "rmse": mean_squared_error(y[test_idx], pred, squared=False),
                    "mae": mean_absolute_error(y[test_idx], pred),
                    "r2": r2_score(y[test_idx], pred),
                    "mean_gp_sigma": float(np.mean(std)),
                }
            )

    return pd.DataFrame(rows)


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


def make_plots(
    df: pd.DataFrame,
    models: dict[str, object],
    train_idx: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    w_grid = np.linspace(0.0, 1.0, 120)
    q_grid = np.linspace(0.0, 0.1, 120)
    W, Q = np.meshgrid(w_grid, q_grid)
    Xplot = np.column_stack([W.ravel(), Q.ravel()])
    W_eval, Q_eval = np.meshgrid(
        np.sort(df["w"].unique()),
        np.sort(df["Q"].unique()),
    )

    for target in ["deltaOmega", "deltaLambda"]:
        pred, std = models[target].predict(Xplot, return_std=True)
        full_pred = models[target].predict(df[["w", "Q"]].to_numpy())
        Z = pred.reshape(W.shape)
        S = std.reshape(W.shape)
        err_df = df.assign(abs_error=np.abs(df[target].to_numpy() - full_pred))
        err = err_df.pivot(index="Q", columns="w", values="abs_error").to_numpy()

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        im = ax.pcolormesh(W, Q, Z, shading="auto", cmap="viridis")
        train = df.iloc[train_idx]
        ax.scatter(train["w"], train["Q"], s=12, c="white", alpha=0.75, linewidths=0)
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

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        im = ax.pcolormesh(
            W_eval,
            Q_eval,
            err,
            shading="auto",
            cmap="inferno",
        )
        ax.scatter(train["w"], train["Q"], s=12, c="white", alpha=0.75, linewidths=0)
        ax.set_xlabel("w")
        ax.set_ylabel("Q")
        ax.set_title(f"Absolute prediction error for {target}")
        fig.colorbar(im, ax=ax)
        fig.savefig(out_dir / f"{target}_error.png", dpi=180)
        plt.close(fig)


def plot_learning_curve(curve: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for target, group in curve.groupby("target"):
        ordered = group.sort_values("n_train")
        ax.plot(ordered["n_train"], ordered["rmse"], marker="o", label=target)
    ax.set_xlabel("Number of training simulations")
    ax.set_ylabel("RMSE on withheld grid points")
    ax.set_yscale("log")
    ax.set_title("Sparse-data learning curve")
    ax.legend()
    fig.savefig(out_dir / "learning_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    params = MetricParams()
    mode = QNMMode(ell=4, overtone=0)

    df = generate_dataset(grid_size=24, params=params, mode=mode)
    df.to_csv(out_dir / "qnm_dataset.csv", index=False)

    models, metrics, train_idx, test_idx = train_models(df, n_train=80)
    metrics.to_csv(out_dir / "gp_metrics.csv", index=False)
    baselines = compare_baselines(df, train_idx, test_idx, models)
    baselines.to_csv(out_dir / "baseline_metrics.csv", index=False)
    curve = learning_curve(df, train_sizes=[20, 40, 80, 160, 300])
    curve.to_csv(out_dir / "learning_curve.csv", index=False)

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

    make_plots(df, models, train_idx, out_dir)
    plot_learning_curve(curve, out_dir)

    print("Wrote:")
    print(f"  {out_dir / 'qnm_dataset.csv'}")
    print(f"  {out_dir / 'gp_metrics.csv'}")
    print(f"  {out_dir / 'baseline_metrics.csv'}")
    print(f"  {out_dir / 'learning_curve.csv'}")
    print(f"  {out_dir / 'parameter_search.csv'}")
    print("  outputs/*_prediction.png")
    print("  outputs/*_uncertainty.png")
    print("  outputs/*_error.png")
    print("  outputs/learning_curve.png")
    print()
    print(metrics.to_string(index=False))
    print()
    print(pd.read_csv(out_dir / "parameter_search.csv").to_string(index=False))


if __name__ == "__main__":
    main()
