from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kiselev_gp_surrogate import (
    KiselevParams,
    QNMMode,
    analytic_lambda_star,
    analytic_omega_star,
    eikonal_qnm,
)


def qnm_to_observables(omega_qnm: complex, ell: int, n: int) -> tuple[float, float]:
    """Extract Omega and lambda from one complex eikonal QNM frequency.

    This uses omega_QNM = ell*Omega - i*(n+1/2)*lambda.
    One complex QNM gives two real observables; it does not by itself determine
    the full density profile rho(r) or w_theta separately.
    """

    if ell == 0:
        raise ValueError("ell must be nonzero.")
    if n <= -0.5:
        raise ValueError("n must satisfy n + 1/2 > 0.")

    omega_obs = omega_qnm.real / ell
    lambda_obs = -omega_qnm.imag / (n + 0.5)
    return omega_obs, lambda_obs


def observables_to_relative_shifts(
    Omega_obs: float,
    lambda_obs: float,
    M: float,
) -> tuple[float, float]:
    Omega0 = 1.0 / (3.0 * np.sqrt(3.0) * M)
    lambda0 = 1.0 / (3.0 * np.sqrt(3.0) * M)
    A = Omega_obs / Omega0 - 1.0
    B = lambda_obs / lambda0 - 1.0
    return A, B


def shifts_to_matter_diagnostics(A: float, B: float, M: float) -> dict[str, float]:
    """Infer static anisotropic-fluid diagnostic combinations.

    The returned quantities are combinations constrained by one complex QNM:
    delta_f(r0), an integrated density diagnostic, and rho(r0)*(1+w_theta).
    They are not a unique reconstruction of rho(r) or w_theta.
    """

    r0 = 3.0 * M
    return {
        "A": A,
        "B": B,
        "delta_f_r0": (2.0 / 3.0) * A,
        "I_rho": r0 * A / (12.0 * np.pi),
        "local_combo": (A - B) / (4.0 * np.pi * r0**2),
    }


def qnm_to_matter_diagnostics(
    omega_qnm: complex,
    ell: int,
    n: int,
    M: float,
) -> dict[str, float]:
    Omega_obs, lambda_obs = qnm_to_observables(omega_qnm, ell, n)
    A, B = observables_to_relative_shifts(Omega_obs, lambda_obs, M)
    diagnostics = shifts_to_matter_diagnostics(A, B, M)
    diagnostics.update(
        {
            "Omega_obs": Omega_obs,
            "lambda_obs": lambda_obs,
        }
    )
    return diagnostics


def reconstruct_exponential_profile(A: float, B: float, M: float, L: float) -> dict[str, float]:
    """Conditionally reconstruct rho0 and w_theta for a fixed density ansatz.

    Assumes rho(r) = rho0*exp(-(r-r0)/L). This is not a model-independent matter
    reconstruction; it is conditional on the chosen profile and fixed L.
    """

    if L <= 0.0:
        raise ValueError("L must be positive.")

    r0 = 3.0 * M
    diagnostics = shifts_to_matter_diagnostics(A, B, M)
    profile_integral = L * (r0**2 + 2.0 * r0 * L + 2.0 * L**2)
    rho0 = diagnostics["I_rho"] / profile_integral
    if np.isclose(rho0, 0.0):
        w_theta = np.nan
    else:
        w_theta = diagnostics["local_combo"] / rho0 - 1.0

    return {
        "L": L,
        "rho0": rho0,
        "w_theta": w_theta,
        "profile_integral": profile_integral,
        **diagnostics,
    }


def qnm_from_relative_shifts(A: float, B: float, M: float, mode: QNMMode) -> complex:
    Omega0 = 1.0 / (3.0 * np.sqrt(3.0) * M)
    lambda0 = 1.0 / (3.0 * np.sqrt(3.0) * M)
    return eikonal_qnm(Omega0 * (1.0 + A), lambda0 * (1.0 + B), mode)


def bardeen_shifts(q: float, M: float) -> tuple[float, float]:
    A = q**2 / (6.0 * M**2)
    B = -q**2 / (9.0 * M**2)
    return A, B


def hayward_shifts(q: float, M: float) -> tuple[float, float]:
    A = q**3 / (27.0 * M**3)
    B = -2.0 * q**3 / (27.0 * M**3)
    return A, B


def build_diagnostic_rows(M: float, mode: QNMMode) -> pd.DataFrame:
    rows = []
    params = KiselevParams(mass=M)

    for w_q in [-0.95, -0.8, -0.65, -0.5]:
        for k in [-0.015, -0.005, 0.005, 0.015]:
            omega_star = analytic_omega_star(w_q, k, params)
            lambda_star = analytic_lambda_star(w_q, k, params)
            omega_qnm = eikonal_qnm(omega_star, lambda_star, mode)
            diagnostics = qnm_to_matter_diagnostics(omega_qnm, mode.ell, mode.overtone, M)
            direct_A = omega_star / params.omega0 - 1.0
            direct_B = lambda_star / params.lambda0 - 1.0
            rows.append(
                {
                    "model": "Kiselev",
                    "parameter_value": k,
                    "q": np.nan,
                    "w_q": w_q,
                    "k": k,
                    "omega_re": omega_qnm.real,
                    "omega_im": omega_qnm.imag,
                    "A": diagnostics["A"],
                    "B": diagnostics["B"],
                    "delta_f_r0": diagnostics["delta_f_r0"],
                    "I_rho": diagnostics["I_rho"],
                    "local_combo": diagnostics["local_combo"],
                    "direct_A": direct_A,
                    "direct_B": direct_B,
                    "abs_A_error": abs(diagnostics["A"] - direct_A),
                    "abs_B_error": abs(diagnostics["B"] - direct_B),
                }
            )

    for model, shift_function in [
        ("Bardeen", bardeen_shifts),
        ("Hayward", hayward_shifts),
    ]:
        for q in np.linspace(0.0, 0.8, 25):
            direct_A, direct_B = shift_function(q, M)
            omega_qnm = qnm_from_relative_shifts(direct_A, direct_B, M, mode)
            diagnostics = qnm_to_matter_diagnostics(omega_qnm, mode.ell, mode.overtone, M)
            rows.append(
                {
                    "model": model,
                    "parameter_value": q,
                    "q": q,
                    "w_q": np.nan,
                    "k": np.nan,
                    "omega_re": omega_qnm.real,
                    "omega_im": omega_qnm.imag,
                    "A": diagnostics["A"],
                    "B": diagnostics["B"],
                    "delta_f_r0": diagnostics["delta_f_r0"],
                    "I_rho": diagnostics["I_rho"],
                    "local_combo": diagnostics["local_combo"],
                    "direct_A": direct_A,
                    "direct_B": direct_B,
                    "abs_A_error": abs(diagnostics["A"] - direct_A),
                    "abs_B_error": abs(diagnostics["B"] - direct_B),
                }
            )

    return pd.DataFrame(rows)


def build_profile_reconstructions(df: pd.DataFrame, M: float, L_values: list[float]) -> pd.DataFrame:
    rows = []
    sample = df.groupby("model", group_keys=False).head(4)
    for _, row in sample.iterrows():
        for L in L_values:
            reconstruction = reconstruct_exponential_profile(row["A"], row["B"], M, L)
            rows.append(
                {
                    "model": row["model"],
                    "parameter_value": row["parameter_value"],
                    "q": row["q"],
                    "w_q": row["w_q"],
                    "k": row["k"],
                    **reconstruction,
                }
            )

    return pd.DataFrame(rows)


def plot_shift_curves(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for model, group in df.groupby("model"):
        if model == "Kiselev":
            for w_q, subgroup in group.groupby("w_q"):
                ordered = subgroup.sort_values("k")
                label = f"Kiselev w_q={w_q:g}"
                axes[0].plot(ordered["k"], ordered["A"], marker="o", label=label)
                axes[1].plot(ordered["k"], ordered["B"], marker="o", label=label)
        else:
            ordered = group.sort_values("q")
            axes[0].plot(ordered["q"], ordered["A"], marker="o", label=model)
            axes[1].plot(ordered["q"], ordered["B"], marker="o", label=model)

    axes[0].set_title("Relative Omega shift")
    axes[0].set_xlabel("model parameter: q or k")
    axes[0].set_ylabel("A = deltaOmega/Omega0")
    axes[1].set_title("Relative lambda shift")
    axes[1].set_xlabel("model parameter: q or k")
    axes[1].set_ylabel("B = deltaLambda/lambda0")
    axes[1].legend(fontsize=8)
    fig.savefig(out_dir / "relative_shifts_by_model.png", dpi=180)
    plt.close(fig)


def plot_diagnostic(df: pd.DataFrame, out_dir: Path, column: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for model, group in df.groupby("model"):
        if model == "Kiselev":
            for w_q, subgroup in group.groupby("w_q"):
                ordered = subgroup.sort_values("k")
                ax.plot(ordered["k"], ordered[column], marker="o", label=f"Kiselev w_q={w_q:g}")
        else:
            ordered = group.sort_values("q")
            ax.plot(ordered["q"], ordered[column], marker="o", label=model)

    ax.set_xlabel("model parameter: q or k")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def plot_model_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        df.groupby("model")[["A", "B", "I_rho", "local_combo"]]
        .agg(["min", "max"])
        .reset_index()
    )
    labels = summary["model"]
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for offset, column in zip([-1.5, -0.5, 0.5, 1.5], ["A", "B", "I_rho", "local_combo"]):
        values = summary[(column, "max")] - summary[(column, "min")]
        ax.bar(x + offset * width, values, width=width, label=f"{column} range")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("diagnostic range")
    ax.set_title("Inferred diagnostic trend ranges by model")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "diagnostic_trend_comparison.png", dpi=180)
    plt.close(fig)


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_shift_curves(df, out_dir)
    plot_diagnostic(df, out_dir, "I_rho", "I_rho", "I_rho_by_model.png")
    plot_diagnostic(
        df,
        out_dir,
        "local_combo",
        "local_combo = rho(r0)(1+w_theta)",
        "local_combo_by_model.png",
    )
    plot_model_comparison(df, out_dir)


def main() -> None:
    out_dir = Path("outputs") / "inverse_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    M = 1.0
    mode = QNMMode(ell=4, overtone=0)
    diagnostics = build_diagnostic_rows(M, mode)
    diagnostics.to_csv(out_dir / "inverse_diagnostics.csv", index=False)

    profile_reconstruction = build_profile_reconstructions(diagnostics, M, L_values=[0.5, 1.0, 2.0])
    profile_reconstruction.to_csv(out_dir / "profile_reconstruction.csv", index=False)

    make_plots(diagnostics, out_dir)

    print("Wrote inverse diagnostic outputs to:")
    print(f"  {out_dir}")
    print()
    print(diagnostics.head().to_string(index=False))
    print()
    print(profile_reconstruction.head().to_string(index=False))


if __name__ == "__main__":
    main()
