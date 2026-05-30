from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class KiselevParams:
    mass: float = 1.0

    @property
    def photon_sphere_radius(self) -> float:
        return 3.0 * self.mass

    @property
    def omega0(self) -> float:
        return 1.0 / (3.0 * np.sqrt(3.0) * self.mass)

    @property
    def lambda0(self) -> float:
        return 1.0 / (3.0 * np.sqrt(3.0) * self.mass)


@dataclass(frozen=True)
class QNMMode:
    ell: int = 4
    overtone: int = 0


def kiselev_f(r: np.ndarray | float, w_q: float, k: float, params: KiselevParams) -> np.ndarray | float:
    """Static Kiselev metric function f(r)=1-2M/r-k/r^(1+3w_q)."""

    radius = np.asarray(r)
    return 1.0 - 2.0 * params.mass / radius - k / radius ** (1.0 + 3.0 * w_q)


def analytic_omega_star(w_q: float, k: float, params: KiselevParams) -> float:
    M = params.mass
    denominator = 2.0 * (3.0 * M) ** (1.0 + 3.0 * w_q)
    return params.omega0 * (1.0 - 3.0 * k / denominator)


def analytic_lambda_star(w_q: float, k: float, params: KiselevParams) -> float:
    M = params.mass
    numerator = (3.0 * w_q * (1.0 + w_q) - 2.0) * k
    denominator = 4.0 * 3.0 ** (3.0 * w_q) * M ** (1.0 + 3.0 * w_q)
    return params.lambda0 * (1.0 + numerator / denominator)


def eikonal_qnm(omega_star: float, lambda_star: float, mode: QNMMode) -> complex:
    return mode.ell * omega_star - 1j * (mode.overtone + 0.5) * lambda_star


def generate_dataset(
    grid_size: int,
    params: KiselevParams,
    mode: QNMMode,
    w_q_bounds: tuple[float, float] = (-1.0, -1.0 / 3.0),
    k_bounds: tuple[float, float] = (-0.02, 0.02),
) -> pd.DataFrame:
    rows = []
    omega0 = params.omega0
    lambda0 = params.lambda0
    qnm0 = eikonal_qnm(omega0, lambda0, mode)

    for w_q in np.linspace(w_q_bounds[0], w_q_bounds[1], grid_size):
        for k in np.linspace(k_bounds[0], k_bounds[1], grid_size):
            f_at_r0 = kiselev_f(params.photon_sphere_radius, w_q, k, params)
            omega_star = analytic_omega_star(w_q, k, params)
            lambda_star = analytic_lambda_star(w_q, k, params)
            qnm = eikonal_qnm(omega_star, lambda_star, mode)
            rows.append(
                {
                    "w_q": w_q,
                    "k": k,
                    "f_r0": f_at_r0,
                    "Omega_star": omega_star,
                    "lambda_star": lambda_star,
                    "deltaOmega_over_Omega0": omega_star / omega0 - 1.0,
                    "deltaLambda_over_lambda0": lambda_star / lambda0 - 1.0,
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
        * Matern(length_scale=[0.35, 0.012], length_scale_bounds=(1.0e-3, 100.0), nu=2.5)
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


def target_columns() -> list[str]:
    return ["deltaOmega_over_Omega0", "deltaLambda_over_lambda0"]


def train_gp_models(
    df: pd.DataFrame,
    n_train: int = 80,
    random_state: int = 7,
):
    features = df[["w_q", "k"]].to_numpy()
    train_idx, test_idx = make_sparse_split(df, n_train=n_train, random_state=random_state)

    models = {}
    metrics = []
    for target in target_columns():
        target_values = df[target].to_numpy()
        model = build_gp(random_state=random_state)
        model.fit(features[train_idx], target_values[train_idx])
        prediction, sigma = model.predict(features[test_idx], return_std=True)
        models[target] = model
        metrics.append(
            {
                "target": target,
                "model": "GaussianProcess",
                "n_train": n_train,
                "n_test": len(test_idx),
                "rmse": mean_squared_error(target_values[test_idx], prediction, squared=False),
                "mae": mean_absolute_error(target_values[test_idx], prediction),
                "r2": r2_score(target_values[test_idx], prediction),
                "mean_gp_sigma": float(np.mean(sigma)),
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
    features = df[["w_q", "k"]].to_numpy()
    rows = []

    for target in target_columns():
        target_values = df[target].to_numpy()
        models = {"GaussianProcess": gp_models[target], **baseline_models(random_state)}
        for model_name, model in models.items():
            if model_name != "GaussianProcess":
                model.fit(features[train_idx], target_values[train_idx])
            prediction = model.predict(features[test_idx])
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "rmse": mean_squared_error(target_values[test_idx], prediction, squared=False),
                    "mae": mean_absolute_error(target_values[test_idx], prediction),
                    "r2": r2_score(target_values[test_idx], prediction),
                }
            )

    return pd.DataFrame(rows)


def learning_curve(
    df: pd.DataFrame,
    train_sizes: list[int],
    random_state: int = 7,
) -> pd.DataFrame:
    features = df[["w_q", "k"]].to_numpy()
    rows = []

    for n_train in train_sizes:
        train_idx, test_idx = make_sparse_split(df, n_train=n_train, random_state=random_state)
        for target in target_columns():
            target_values = df[target].to_numpy()
            model = build_gp(random_state=random_state)
            model.fit(features[train_idx], target_values[train_idx])
            prediction, sigma = model.predict(features[test_idx], return_std=True)
            rows.append(
                {
                    "target": target,
                    "n_train": n_train,
                    "n_test": len(test_idx),
                    "rmse": mean_squared_error(target_values[test_idx], prediction, squared=False),
                    "mae": mean_absolute_error(target_values[test_idx], prediction),
                    "r2": r2_score(target_values[test_idx], prediction),
                    "mean_gp_sigma": float(np.mean(sigma)),
                }
            )

    return pd.DataFrame(rows)


def predict_qnm_from_gp(
    gp_models: dict[str, object],
    w_q: float,
    k: float,
    params: KiselevParams,
    mode: QNMMode,
) -> tuple[complex, float, float]:
    features = np.array([[w_q, k]])
    delta_omega_rel, sigma_omega = gp_models["deltaOmega_over_Omega0"].predict(
        features, return_std=True
    )
    delta_lambda_rel, sigma_lambda = gp_models["deltaLambda_over_lambda0"].predict(
        features, return_std=True
    )
    omega_star = params.omega0 * (1.0 + delta_omega_rel[0])
    lambda_star = params.lambda0 * (1.0 + delta_lambda_rel[0])
    qnm = eikonal_qnm(omega_star, lambda_star, mode)
    return qnm, sigma_omega[0], sigma_lambda[0]


def infer_parameters_from_qnm(
    gp_models: dict[str, object],
    target_qnm: complex,
    params: KiselevParams,
    mode: QNMMode,
    w_q_bounds: tuple[float, float] = (-1.0, -1.0 / 3.0),
    k_bounds: tuple[float, float] = (-0.02, 0.02),
) -> dict[str, float]:
    def objective(candidate: np.ndarray) -> float:
        qnm, sigma_omega, sigma_lambda = predict_qnm_from_gp(
            gp_models, candidate[0], candidate[1], params, mode
        )
        residual = abs(qnm - target_qnm) ** 2
        uncertainty_penalty = 0.05 * (sigma_omega**2 + sigma_lambda**2)
        return float(residual + uncertainty_penalty)

    result = differential_evolution(
        objective,
        bounds=[w_q_bounds, k_bounds],
        seed=11,
    )
    qnm_pred, sigma_omega, sigma_lambda = predict_qnm_from_gp(
        gp_models, result.x[0], result.x[1], params, mode
    )
    return {
        "w_q": result.x[0],
        "k": result.x[1],
        "objective": result.fun,
        "omega_re_pred": qnm_pred.real,
        "omega_im_pred": qnm_pred.imag,
        "sigma_deltaOmega_over_Omega0": sigma_omega,
        "sigma_deltaLambda_over_lambda0": sigma_lambda,
    }


def plot_gp_maps(
    df: pd.DataFrame,
    gp_models: dict[str, object],
    train_idx: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    w_q_dense = np.linspace(df["w_q"].min(), df["w_q"].max(), 140)
    k_dense = np.linspace(df["k"].min(), df["k"].max(), 140)
    w_q_mesh, k_mesh = np.meshgrid(w_q_dense, k_dense)
    dense_features = np.column_stack([w_q_mesh.ravel(), k_mesh.ravel()])

    w_q_eval_mesh, k_eval_mesh = np.meshgrid(
        np.sort(df["w_q"].unique()),
        np.sort(df["k"].unique()),
    )
    train = df.iloc[train_idx]

    for target in target_columns():
        prediction, sigma = gp_models[target].predict(dense_features, return_std=True)
        full_prediction = gp_models[target].predict(df[["w_q", "k"]].to_numpy())
        prediction_grid = prediction.reshape(w_q_mesh.shape)
        sigma_grid = sigma.reshape(w_q_mesh.shape)
        error_df = df.assign(abs_error=np.abs(df[target].to_numpy() - full_prediction))
        error_grid = error_df.pivot(index="k", columns="w_q", values="abs_error").to_numpy()

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        im = ax.pcolormesh(w_q_mesh, k_mesh, prediction_grid, shading="auto", cmap="viridis")
        ax.scatter(train["w_q"], train["k"], s=12, c="white", alpha=0.75, linewidths=0)
        ax.set_xlabel("w_q")
        ax.set_ylabel("k")
        ax.set_title(f"GP prediction for {target}")
        fig.colorbar(im, ax=ax)
        fig.savefig(out_dir / f"{target}_prediction.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        im = ax.pcolormesh(w_q_mesh, k_mesh, sigma_grid, shading="auto", cmap="magma")
        ax.set_xlabel("w_q")
        ax.set_ylabel("k")
        ax.set_title(f"GP uncertainty for {target}")
        fig.colorbar(im, ax=ax)
        fig.savefig(out_dir / f"{target}_uncertainty.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        im = ax.pcolormesh(
            w_q_eval_mesh,
            k_eval_mesh,
            error_grid,
            shading="auto",
            cmap="inferno",
        )
        ax.scatter(train["w_q"], train["k"], s=12, c="white", alpha=0.75, linewidths=0)
        ax.set_xlabel("w_q")
        ax.set_ylabel("k")
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
    ax.set_title("Kiselev sparse-data learning curve")
    ax.legend()
    fig.savefig(out_dir / "learning_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    out_dir = Path("outputs") / "kiselev"
    out_dir.mkdir(parents=True, exist_ok=True)
    params = KiselevParams(mass=1.0)
    mode = QNMMode(ell=4, overtone=0)

    df = generate_dataset(grid_size=28, params=params, mode=mode)
    df.to_csv(out_dir / "kiselev_qnm_dataset.csv", index=False)

    gp_models, gp_metrics, train_idx, test_idx = train_gp_models(df, n_train=80)
    gp_metrics.to_csv(out_dir / "gp_metrics.csv", index=False)

    baselines = compare_baselines(df, train_idx, test_idx, gp_models)
    baselines.to_csv(out_dir / "baseline_metrics.csv", index=False)

    curve = learning_curve(df, train_sizes=[20, 40, 80, 160, 300])
    curve.to_csv(out_dir / "learning_curve.csv", index=False)

    target_w_q = -0.72
    target_k = 0.012
    target_omega = analytic_omega_star(target_w_q, target_k, params)
    target_lambda = analytic_lambda_star(target_w_q, target_k, params)
    target_qnm = eikonal_qnm(target_omega, target_lambda, mode)
    inference = infer_parameters_from_qnm(gp_models, target_qnm, params, mode)
    pd.DataFrame(
        [
            {
                "target_w_q": target_w_q,
                "target_k": target_k,
                "target_omega_re": target_qnm.real,
                "target_omega_im": target_qnm.imag,
                **inference,
            }
        ]
    ).to_csv(out_dir / "parameter_search.csv", index=False)

    plot_gp_maps(df, gp_models, train_idx, out_dir)
    plot_learning_curve(curve, out_dir)

    print("Wrote Kiselev outputs to:")
    print(f"  {out_dir}")
    print()
    print(gp_metrics.to_string(index=False))
    print()
    print(pd.read_csv(out_dir / "parameter_search.csv").to_string(index=False))


if __name__ == "__main__":
    main()
