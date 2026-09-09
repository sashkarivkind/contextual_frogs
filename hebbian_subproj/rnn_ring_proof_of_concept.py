#!/usr/bin/env python3
"""Reproduce the low-rank RNN benchmark and a sigmoid single-repeller construction.

Model
-----
    x[t+1] = sigmoid(W x[t])
    W_* = (g/N) (z1 z1^T + z2 z2^T)
    W_e = W_* + (epsilon/N) eta eta^T

The script uses deterministic population quadrature for the analytic-limit
figures and an exact low-rank finite-N implementation (it never forms W).

Outputs
-------
    figure1_benchmark_angular_drift.png
    figure2_rank_one_mechanism.png
    figure3_population_proof.png
    figure4_finite_n_validation.png
    rnn_ring_results.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, root
from scipy.special import expit, roots_hermitenorm


@dataclass(frozen=False)
class Parameters:
    gain: float = 4.20
    pattern_mean_parameter: float = -0.65
    epsilon: float = 0.04


def expectation(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values))


def normal_nodes(order: int):
    nodes, weights = roots_hermitenorm(order)
    return nodes, weights / np.sqrt(2.0 * np.pi)


def one_dimensional_nodes(distribution: str, order: int):
    if distribution == "rademacher":
        return np.array([-1.0, 1.0]), np.array([0.5, 0.5])
    if distribution == "uniform":
        nodes, weights = leggauss(order)
        return np.sqrt(3.0) * nodes, weights / 2.0
    if distribution == "gaussian":
        return normal_nodes(order)
    if distribution == "laplace":
        nodes, weights = laggauss(order)
        scale = 1.0 / np.sqrt(2.0)
        values = np.concatenate((-scale * nodes, scale * nodes))
        probabilities = np.concatenate((0.5 * weights, 0.5 * weights))
        return values, probabilities
    raise ValueError(f"Unknown distribution: {distribution}")


def product_nodes(distribution: str, order: int):
    nodes, weights = one_dimensional_nodes(distribution, order)
    x1, x2 = np.meshgrid(nodes, nodes, indexing="ij")
    w1, w2 = np.meshgrid(weights, weights, indexing="ij")
    return x1.ravel(), x2.ravel(), (w1 * w2).ravel()


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def tanh_overlap_map(m, gain, x1, x2, weights):
    field = gain * (x1 * m[0] + x2 * m[1])
    output = np.tanh(field)
    return np.array(
        [
            expectation(x1 * output, weights),
            expectation(x2 * output, weights),
        ]
    )


def radial_root(theta, gain, x1, x2, weights):
    direction = np.array([np.cos(theta), np.sin(theta)])

    def residual(radius):
        return (
            np.dot(
                direction,
                tanh_overlap_map(radius * direction, gain, x1, x2, weights),
            )
            - radius
        )

    lower = 1e-9
    upper = 0.25
    while residual(upper) > 0.0 and upper < 8.0:
        upper *= 1.6
    if residual(lower) <= 0.0 or residual(upper) >= 0.0:
        raise RuntimeError("Unable to bracket the nonzero radial fixed point")
    return brentq(residual, lower, upper, xtol=2e-13, rtol=2e-12)


def benchmark_profile(distribution, gain, theta, order=52):
    x1, x2, weights = product_nodes(distribution, order)
    radius = np.empty_like(theta)
    angular_drift = np.empty_like(theta)
    for index, angle in enumerate(theta):
        radius[index] = radial_root(angle, gain, x1, x2, weights)
        point = radius[index] * np.array([np.cos(angle), np.sin(angle)])
        mapped = tanh_overlap_map(point, gain, x1, x2, weights)
        angular_drift[index] = wrap_angle(
            np.arctan2(mapped[1], mapped[0]) - angle
        )
    return radius, angular_drift


def gaussian_ring_radius(gain: float, order: int = 240):
    z, weights = normal_nodes(order)

    def residual(radius):
        probability = expit(gain * radius * z)
        derivative = probability * (1.0 - probability)
        return gain * expectation(derivative, weights) - 1.0

    return brentq(residual, 1e-10, 4.0, xtol=1e-14, rtol=2e-13)


def sigmoid_ring_radial_multiplier(gain: float, radius: float, order=300):
    z, weights = normal_nodes(order)
    probability = expit(gain * radius * z)
    first = probability * (1.0 - probability)
    second = first * (1.0 - 2.0 * probability)
    q_prime = gain**2 * expectation(z * second, weights)
    return 1.0 + radius * q_prime


def activation(field: np.ndarray, parameters: Parameters):
    """Standard logistic sigmoid; parameters is accepted for a uniform API."""
    del parameters
    return expit(field)


def activation_prime(field: np.ndarray, parameters: Parameters):
    probability = activation(field, parameters)
    return probability * (1.0 - probability)


def added_pattern(z1: np.ndarray, parameters: Parameters):
    mu = parameters.pattern_mean_parameter
    return (mu + z1) / np.sqrt(1.0 + mu * mu)


def population_grid(order: int):
    z, weights_1d = normal_nodes(order)
    z1, z2 = np.meshgrid(z, z, indexing="ij")
    weights = np.outer(weights_1d, weights_1d)
    return z1.ravel(), z2.ravel(), weights.ravel()


def population_map(state, parameters, z1, z2, weights):
    eta = added_pattern(z1, parameters)
    field = parameters.gain * (z1 * state[0] + z2 * state[1])
    field += parameters.epsilon * eta * state[2]
    output = activation(field, parameters)
    return np.array(
        [
            expectation(z1 * output, weights),
            expectation(z2 * output, weights),
            expectation(eta * output, weights),
        ]
    )


def population_jacobian(state, parameters, z1, z2, weights):
    eta = added_pattern(z1, parameters)
    output_patterns = np.stack((z1, z2, eta))
    input_patterns = np.stack(
        (
            parameters.gain * z1,
            parameters.gain * z2,
            parameters.epsilon * eta,
        )
    )
    field = input_patterns.T @ state
    derivative = activation_prime(field, parameters)
    return np.einsum(
        "ki,li,i,i->kl",
        output_patterns,
        input_patterns,
        derivative,
        weights,
    )


def unperturbed_pattern_overlap(theta, radius, parameters, z1, z2, weights):
    eta = added_pattern(z1, parameters)
    field = parameters.gain * radius * (
        z1 * np.cos(theta) + z2 * np.sin(theta)
    )
    return expectation(eta * activation(field, parameters), weights)


def unperturbed_state(theta, radius, parameters, z1, z2, weights):
    return np.array(
        [
            radius * np.cos(theta),
            radius * np.sin(theta),
            unperturbed_pattern_overlap(
                theta, radius, parameters, z1, z2, weights
            ),
        ]
    )


def find_population_fixed_points(parameters, order=140, seed_count=128):
    z1, z2, weights = population_grid(order)
    radius = gaussian_ring_radius(parameters.gain)
    seeds = [np.zeros(3)]
    for theta in np.linspace(0.0, 2.0 * np.pi, seed_count, endpoint=False):
        seeds.append(
            unperturbed_state(
                theta, radius, parameters, z1, z2, weights
            )
        )
    # The positive marked axis contains an outer tangential saddle and,
    # at these parameters, a second inner source.  Axial seeds make the
    # fixed-point inventory reproducible instead of relying on chance.
    for m1 in np.linspace(-0.35, 0.25, 31):
        for pattern_overlap in np.linspace(-0.55, 0.10, 17):
            seeds.append(np.array([m1, 0.0, pattern_overlap]))
    rng = np.random.default_rng(19)
    seeds.extend(
        rng.uniform([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], size=(128, 3))
    )

    fixed_points = []
    for seed in seeds:
        solution = root(
            lambda state: population_map(
                state, parameters, z1, z2, weights
            )
            - state,
            seed,
            method="hybr",
            options={"xtol": 1e-11, "maxfev": 3000},
        )
        residual = np.linalg.norm(
            population_map(solution.x, parameters, z1, z2, weights)
            - solution.x
        )
        if not solution.success or residual > 1e-8:
            continue
        if any(
            np.linalg.norm(solution.x - item["state"]) < 1e-6
            for item in fixed_points
        ):
            continue
        jacobian = population_jacobian(
            solution.x, parameters, z1, z2, weights
        )
        eigenvalues = np.linalg.eigvals(jacobian)
        fixed_points.append(
            {
                "state": solution.x,
                "angle": float(
                    np.arctan2(solution.x[1], solution.x[0])
                    % (2.0 * np.pi)
                ),
                "radius": float(np.linalg.norm(solution.x[:2])),
                "residual": float(residual),
                "eigenvalues": eigenvalues,
                "spectral_radius": float(np.max(np.abs(eigenvalues))),
                "stable": bool(np.max(np.abs(eigenvalues)) < 1.0),
            }
        )
    fixed_points.sort(
        key=lambda item: (
            np.linalg.norm(item["state"][:2]) < 1e-7,
            item["angle"],
        )
    )
    return fixed_points, (z1, z2, weights), radius


def iterate_population_batch(
    states,
    parameters,
    z1,
    z2,
    weights,
    steps=900,
    record_every=3,
):
    eta = added_pattern(z1, parameters)
    patterns = np.stack((z1, z2, eta), axis=1)
    states = np.asarray(states, dtype=float)
    records = [states.copy()]
    for step in range(1, steps + 1):
        field = (
            parameters.gain
            * (
                z1[:, None] * states[None, :, 0]
                + z2[:, None] * states[None, :, 1]
            )
            + parameters.epsilon
            * eta[:, None]
            * states[None, :, 2]
        )
        output = activation(field, parameters)
        states = (patterns.T @ (weights[:, None] * output)).T
        if step % record_every == 0:
            records.append(states.copy())
    return np.asarray(records)


def first_order_profiles(theta, radius, parameters):
    """Weak-update drift for centered versus standard logistic sigmoid."""
    mu = parameters.pattern_mean_parameter
    normalization = np.sqrt(1.0 + mu * mu)
    # sigma(h)-1/2 is odd.  The mean component of eta then drops out,
    # leaving two equally stable represented poles.
    conventional_overlap = radius * np.cos(theta) / normalization
    conventional_derivative = -radius * np.sin(theta) / normalization
    conventional_drift = (
        parameters.epsilon
        * conventional_overlap
        * conventional_derivative
        / (parameters.gain * radius**2)
    )

    # For standard logistic sigma, E sigma(grZ)=1/2 exactly.  The ring
    # identity gives E[z1 sigma(gr u.z)]=r cos(theta).
    proposed_overlap = (
        radius * np.cos(theta) + mu / 2.0
    ) / normalization
    derivative = -radius * np.sin(theta) / normalization
    proposed_drift = (
        parameters.epsilon
        * proposed_overlap
        * derivative
        / (parameters.gain * radius**2)
    )
    return {
        "conventional_overlap": conventional_overlap,
        "conventional_drift": conventional_drift,
        "proposed_overlap": proposed_overlap,
        "proposed_drift": proposed_drift,
    }


def finite_n_profiles(
    theta,
    radius,
    parameters,
    size,
    seed,
    diagnostic_states=None,
):
    rng = np.random.default_rng(seed)
    # float32 keeps the exact low-rank N=25,000,000 validation below 0.6 GB.
    # Accumulations are explicitly promoted to float64.
    z1 = rng.standard_normal(size, dtype=np.float32)
    z2 = rng.standard_normal(size, dtype=np.float32)
    eta = added_pattern(z1, parameters).astype(np.float32)
    baseline_drift = np.empty_like(theta)
    perturbed_drift = np.empty_like(theta)

    for index, angle in enumerate(theta):
        m1 = radius * np.cos(angle)
        m2 = radius * np.sin(angle)
        baseline_field = (
            parameters.gain * (z1 * m1 + z2 * m2)
        ).astype(np.float32)
        baseline_output = activation(baseline_field, parameters)
        p0 = float(np.mean(eta * baseline_output, dtype=np.float64))

        baseline_mapped = np.array(
            [
                np.mean(z1 * baseline_output, dtype=np.float64),
                np.mean(z2 * baseline_output, dtype=np.float64),
            ]
        )
        perturbed_output = activation(
            baseline_field + parameters.epsilon * eta * p0,
            parameters,
        )
        perturbed_mapped = np.array(
            [
                np.mean(z1 * perturbed_output, dtype=np.float64),
                np.mean(z2 * perturbed_output, dtype=np.float64),
            ]
        )
        baseline_drift[index] = wrap_angle(
            np.arctan2(baseline_mapped[1], baseline_mapped[0]) - angle
        )
        perturbed_drift[index] = wrap_angle(
            np.arctan2(perturbed_mapped[1], perturbed_mapped[0]) - angle
        )

    induced_drift = perturbed_drift - baseline_drift
    metrics = {
        "network_size": int(size),
        "seed": int(seed),
        "max_unperturbed_drift": float(np.max(np.abs(baseline_drift))),
        "max_induced_drift": float(np.max(np.abs(induced_drift))),
        "max_ratio": float(
            np.max(np.abs(induced_drift))
            / np.max(np.abs(baseline_drift))
        ),
        "rms_unperturbed_drift": float(
            np.sqrt(np.mean(baseline_drift**2))
        ),
        "rms_induced_drift": float(np.sqrt(np.mean(induced_drift**2))),
        "rms_ratio": float(
            np.sqrt(np.mean(induced_drift**2))
            / np.sqrt(np.mean(baseline_drift**2))
        ),
    }
    finite_jacobians = {}
    cached_state = None
    cached_map = None
    cached_jacobian = None

    def sample_map_and_jacobian(state):
        nonlocal cached_state, cached_map, cached_jacobian
        state = np.asarray(state, dtype=float)
        if cached_state is not None and np.array_equal(state, cached_state):
            return cached_map, cached_jacobian
        field = (
            parameters.gain * (z1 * state[0] + z2 * state[1])
            + parameters.epsilon * eta * state[2]
        ).astype(np.float32)
        probability = expit(field)
        derivative = probability * (1.0 - probability)
        mapped = np.array(
            [
                np.mean(z1 * probability, dtype=np.float64),
                np.mean(z2 * probability, dtype=np.float64),
                np.mean(eta * probability, dtype=np.float64),
            ]
        )
        moments = {
            "z1z1": np.mean(z1 * z1 * derivative, dtype=np.float64),
            "z1z2": np.mean(z1 * z2 * derivative, dtype=np.float64),
            "z1eta": np.mean(z1 * eta * derivative, dtype=np.float64),
            "z2z2": np.mean(z2 * z2 * derivative, dtype=np.float64),
            "z2eta": np.mean(z2 * eta * derivative, dtype=np.float64),
            "etaeta": np.mean(eta * eta * derivative, dtype=np.float64),
        }
        jacobian = np.array(
            [
                [
                    parameters.gain * moments["z1z1"],
                    parameters.gain * moments["z1z2"],
                    parameters.epsilon * moments["z1eta"],
                ],
                [
                    parameters.gain * moments["z1z2"],
                    parameters.gain * moments["z2z2"],
                    parameters.epsilon * moments["z2eta"],
                ],
                [
                    parameters.gain * moments["z1eta"],
                    parameters.gain * moments["z2eta"],
                    parameters.epsilon * moments["etaeta"],
                ],
            ]
        )
        cached_state = state.copy()
        cached_map = mapped
        cached_jacobian = jacobian
        return mapped, jacobian

    for name, state in (diagnostic_states or {}).items():
        solution = root(
            lambda current: sample_map_and_jacobian(current)[0] - current,
            state,
            jac=lambda current: (
                sample_map_and_jacobian(current)[1] - np.eye(3)
            ),
            method="hybr",
            options={"xtol": 2e-10, "maxfev": 100},
        )
        mapped, jacobian = sample_map_and_jacobian(solution.x)
        residual = float(np.linalg.norm(mapped - solution.x))
        eigenvalues = np.linalg.eigvals(jacobian)
        finite_jacobians[name] = {
            "state": solution.x.tolist(),
            "root_success": bool(solution.success or residual < 2e-7),
            "residual": residual,
            "jacobian": jacobian.tolist(),
            "eigenvalues": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in eigenvalues
            ],
            "spectral_radius": float(np.max(np.abs(eigenvalues))),
        }
    metrics["fixed_point_jacobians"] = finite_jacobians
    return baseline_drift, perturbed_drift, induced_drift, metrics


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 120,
            "savefig.facecolor": "white",
        }
    )


def save_benchmark_figure(output_dir, gain):
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    distributions = ["rademacher", "uniform", "gaussian", "laplace"]
    labels = {
        "rademacher": r"Rademacher ($\mu_4=1$)",
        "uniform": r"Uniform ($\mu_4=1.8$)",
        "gaussian": r"Gaussian ($\mu_4=3$)",
        "laplace": r"Laplace ($\mu_4=6$)",
    }
    data = {}
    figure, axis = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    for distribution in distributions:
        radius, drift = benchmark_profile(
            distribution, gain, theta, order=54
        )
        data[distribution] = {
            "theta": theta.tolist(),
            "radius": radius.tolist(),
            "angular_drift": drift.tolist(),
        }
        axis.plot(theta, drift, linewidth=2.2, label=labels[distribution])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title=f"Component kurtosis selects the angular attractors ($g={gain:.2f}$)",
        xlabel=r"Angle on radial fixed-point curve, $\theta$",
        ylabel=r"One-step angular drift, $\arg(m^+)-\theta$",
        xlim=(0.0, 2.0 * np.pi),
    )
    axis.set_xticks(
        [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi],
        ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, ncol=2)
    path = output_dir / "figure1_benchmark_angular_drift.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path, data


def save_mechanism_figure(
    output_dir, theta, profiles, parameters
):
    figure, axes = plt.subplots(
        2, 1, figsize=(9.4, 7.4), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        theta,
        profiles["conventional_drift"],
        linewidth=2.2,
        label=r"Centered sigmoid $\sigma(h)-1/2$",
    )
    axes[0].plot(
        theta,
        profiles["proposed_drift"],
        linewidth=2.2,
        label=r"Standard sigmoid $\sigma(h)$",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel(r"First-order angular drift, $\Delta\theta$")
    axes[0].set_title(
        "The sigmoid offset reverses the marked-pole drift"
    )
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.18)

    axes[1].plot(
        theta,
        profiles["conventional_overlap"] ** 2,
        linewidth=2.2,
        label=r"Centered-sigmoid $p_0(\theta)^2$",
    )
    axes[1].plot(
        theta,
        profiles["proposed_overlap"] ** 2,
        linewidth=2.2,
        label=r"Standard-sigmoid $p_0(\theta)^2$",
    )
    axes[1].set(
        xlabel=r"Ring angle, $\theta$",
        ylabel=r"Squared baseline overlap, $p_0(\theta)^2$",
    )
    axes[1].set_xticks(
        [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi],
        ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.18)
    figure.text(
        0.99,
        0.005,
        (
            rf"$g={parameters.gain}$, $\epsilon={parameters.epsilon}$, "
            rf"$\mu={parameters.pattern_mean_parameter}$"
        ),
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.35",
    )
    path = output_dir / "figure2_rank_one_mechanism.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def save_population_figure(
    output_dir,
    trajectories,
    initial_angles,
    fixed_points,
    ring_radius,
):
    figure, axes = plt.subplots(
        1, 2, figsize=(12.0, 5.3), constrained_layout=True
    )
    ring_theta = np.linspace(0.0, 2.0 * np.pi, 500)
    axes[0].plot(
        ring_radius * np.cos(ring_theta),
        ring_radius * np.sin(ring_theta),
        linestyle="--",
        linewidth=1.8,
        color="0.65",
        label="Unperturbed ring",
    )
    for index in range(trajectories.shape[1]):
        axes[0].plot(
            trajectories[:, index, 0],
            trajectories[:, index, 1],
            linewidth=1.25,
            alpha=0.82,
        )
    for item in fixed_points:
        if np.linalg.norm(item["state"][:2]) < 1e-7:
            continue
        if item["stable"]:
            axes[0].scatter(
                item["state"][0],
                item["state"][1],
                s=105,
                marker="o",
                color="#1f77b4",
                edgecolor="white",
                linewidth=0.8,
                zorder=6,
                label="Stable opposite attractor",
            )
        else:
            axes[0].scatter(
                item["state"][0],
                item["state"][1],
                s=115,
                marker="X",
                color="#d62728",
                edgecolor="white",
                linewidth=0.8,
                zorder=6,
                label="Unstable marked-axis fixed point",
            )
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), frameon=False)
    axes[0].axhline(0.0, color="0.9", linewidth=0.7)
    axes[0].axvline(0.0, color="0.9", linewidth=0.7)
    axes[0].set(
        xlabel=r"$m_1$",
        ylabel=r"$m_2$",
        title="Population trajectories",
        aspect="equal",
    )

    time = np.arange(trajectories.shape[0]) * 3
    for index, initial_angle in enumerate(initial_angles):
        unwrapped = np.unwrap(
            np.arctan2(
                trajectories[:, index, 1], trajectories[:, index, 0]
            )
        )
        # Put all branches on a common 0-to-pi destination display.
        if initial_angle > np.pi:
            unwrapped = 2.0 * np.pi - unwrapped
        axes[1].plot(time, unwrapped, linewidth=1.25, alpha=0.82)
    axes[1].axhline(
        np.pi,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.2,
        label=r"Opposite pole, $\pi$",
    )
    axes[1].set(
        xlabel="Iteration",
        ylabel="Unwrapped angular distance from marked pole",
        title="Tangential escape and convergence",
        ylim=(-0.05, np.pi + 0.2),
    )
    axes[1].set_yticks(
        [0.0, np.pi / 2.0, np.pi],
        ["0", r"$\pi/2$", r"$\pi$"],
    )
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False)
    path = output_dir / "figure3_population_proof.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def save_finite_n_figure(
    output_dir,
    theta,
    baseline_drift,
    induced_drift,
    metrics,
):
    figure, axis = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    axis.plot(
        theta,
        baseline_drift,
        linewidth=1.8,
        label="Unperturbed finite-$N$ drift",
    )
    axis.plot(
        theta,
        induced_drift,
        linewidth=2.2,
        label="Drift induced by rank-one term",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title=(
            "Finite-network signal exceeds the baseline corrugation "
            f"({metrics['max_ratio']:.1f}x)"
        ),
        xlabel=r"Baseline ring angle, $\theta$",
        ylabel="One-step angular drift",
        xlim=(0.0, 2.0 * np.pi),
    )
    axis.set_xticks(
        [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi],
        ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    axis.text(
        0.99,
        0.03,
        (
            f"N={metrics['network_size']:,}; "
            f"max induced={metrics['max_induced_drift']:.4f}; "
            f"max baseline={metrics['max_unperturbed_drift']:.4f}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.35",
    )
    path = output_dir / "figure4_finite_n_validation.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def serializable_fixed_points(fixed_points):
    result = []
    for item in fixed_points:
        result.append(
            {
                "state": item["state"].tolist(),
                "angle": item["angle"],
                "radius": item["radius"],
                "residual": item["residual"],
                "eigenvalues": [
                    {"real": float(value.real), "imag": float(value.imag)}
                    for value in item["eigenvalues"]
                ],
                "spectral_radius": item["spectral_radius"],
                "stable": item["stable"],
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="rnn_sigmoid_results")
    parser.add_argument("--finite-n", type=int, default=25_000_000)
    parser.add_argument("--seed", type=int, default=20_260_905)
    parser.add_argument('--gain', type=float, default=4.2, help='Gain parameter for the model')
    parser.add_argument('--epsilon', type=float, default=0.04, help='Epsilon parameter for the model')
    parser.add_argument('--pattern_mean_parameter', type=float, default=-0.65, help='Pattern mean parameter for the model')

    args = parser.parse_args()

    setup_plot_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = Parameters()
    parameters.gain = args.gain
    parameters.epsilon = args.epsilon
    parameters.pattern_mean_parameter = args.pattern_mean_parameter

    figure1, benchmark_data = save_benchmark_figure(
        output_dir, gain=1.08
    )

    fixed_points, grid, ring_radius = find_population_fixed_points(
        parameters, order=140
    )
    z1, z2, weights = grid
    baseline_radial_multiplier = sigmoid_ring_radial_multiplier(
        parameters.gain, ring_radius
    )
    condition_ratio = (
        -parameters.pattern_mean_parameter / (2.0 * ring_radius)
    )
    normalization_squared = 1.0 + parameters.pattern_mean_parameter**2
    weak_marked_slope = (
        -parameters.epsilon
        * (ring_radius + parameters.pattern_mean_parameter / 2.0)
        / (parameters.gain * ring_radius * normalization_squared)
    )
    weak_opposite_slope = (
        parameters.epsilon
        * (parameters.pattern_mean_parameter / 2.0 - ring_radius)
        / (parameters.gain * ring_radius * normalization_squared)
    )
    centered_sigmoid_pole_slope = (
        -parameters.epsilon
        / (parameters.gain * normalization_squared)
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    profiles = first_order_profiles(theta, ring_radius, parameters)
    figure2 = save_mechanism_figure(
        output_dir, theta, profiles, parameters
    )

    initial_angles = np.array(
        [0.025, 0.08, 0.20, 0.55, 1.1, 1.8, 2.6,
         3.7, 4.6, 5.35, 5.9, 6.20]
    )
    initial_states = np.stack(
        [
            unperturbed_state(
                angle, ring_radius, parameters, z1, z2, weights
            )
            for angle in initial_angles
        ]
    )
    trajectories = iterate_population_batch(
        initial_states,
        parameters,
        z1,
        z2,
        weights,
        steps=1200,
        record_every=3,
    )
    figure3 = save_population_figure(
        output_dir,
        trajectories,
        initial_angles,
        fixed_points,
        ring_radius,
    )

    rng = np.random.default_rng(41)
    random_initial_states = rng.uniform(
        [-0.8, -0.8, -0.8], [0.8, 0.8, 0.8], size=(100, 3)
    )
    random_trajectories = iterate_population_batch(
        random_initial_states,
        parameters,
        z1,
        z2,
        weights,
        steps=1500,
        record_every=1500,
    )
    stable_points = [item for item in fixed_points if item["stable"]]
    stable_state = stable_points[0]["state"]
    random_convergence_count = int(
        np.sum(
            np.linalg.norm(
                random_trajectories[-1] - stable_state[None, :], axis=1
            )
            < 1e-6
        )
    )

    amplitude = 0.22
    normalization = np.sqrt(normalization_squared)
    off_manifold_initial_states = np.array(
        [
            [amplitude / normalization, 0.0, amplitude],
            [amplitude / normalization, 0.003, amplitude],
            [amplitude / normalization, -0.003, amplitude],
        ]
    )
    off_manifold_trajectories = iterate_population_batch(
        off_manifold_initial_states,
        parameters,
        z1,
        z2,
        weights,
        steps=1500,
        record_every=1500,
    )

    marked_unstable = max(
        (
            item
            for item in fixed_points
            if (not item["stable"]) and item["state"][0] > 0.0
        ),
        key=lambda item: item["radius"],
    )
    stable_opposite = stable_points[0]
    finite_theta = np.linspace(0.0, 2.0 * np.pi, 41, endpoint=False)
    baseline_drift, perturbed_drift, induced_drift, finite_metrics = (
        finite_n_profiles(
            finite_theta,
            ring_radius,
            parameters,
            size=args.finite_n,
            seed=args.seed,
            diagnostic_states={
                "marked_outer_saddle": marked_unstable["state"],
                "opposite_attractor": stable_opposite["state"],
            },
        )
    )
    figure4 = save_finite_n_figure(
        output_dir,
        finite_theta,
        baseline_drift,
        induced_drift,
        finite_metrics,
    )

    result = {
        "model": {
            "parameters": asdict(parameters),
            "activation": "sigma(z) = 1 / (1 + exp(-z))",
            "added_pattern": "(mu + z1) / sqrt(1 + mu^2)",
        },
        "analytic": {
            "gaussian_ring_radius": ring_radius,
            "critical_gain": 4.0,
            "unperturbed_radial_multiplier": baseline_radial_multiplier,
            "single_repeller_condition": "mu / 2 < -r_star",
            "condition_ratio_minus_mu_over_2radius": condition_ratio,
            "weak_drift_slope_at_marked_pole": weak_marked_slope,
            "weak_drift_slope_at_opposite_pole": weak_opposite_slope,
            "centered_sigmoid_pole_slope_ablation": centered_sigmoid_pole_slope,
        },
        "population": {
            "quadrature_order": 140,
            "fixed_points": serializable_fixed_points(fixed_points),
            "random_initial_conditions": 100,
            "random_convergence_count": random_convergence_count,
            "trajectory_initial_angles": initial_angles.tolist(),
            "trajectory_final_states": trajectories[-1].tolist(),
            "off_manifold_initial_states": off_manifold_initial_states.tolist(),
            "off_manifold_final_states": off_manifold_trajectories[-1].tolist(),
            "off_manifold_note": (
                "The exactly aligned state lies on the saddle's invariant "
                "stable manifold; +/-0.003 transverse perturbations escape "
                "to the opposite attractor."
            ),
        },
        "finite_n": {
            **finite_metrics,
            "theta": finite_theta.tolist(),
            "unperturbed_angular_drift": baseline_drift.tolist(),
            "perturbed_angular_drift": perturbed_drift.tolist(),
            "induced_angular_drift": induced_drift.tolist(),
        },
        "benchmark": benchmark_data,
        "first_order_profiles": {
            "theta": theta.tolist(),
            **{key: value.tolist() for key, value in profiles.items()},
        },
        "figures": [
            figure1.name,
            figure2.name,
            figure3.name,
            figure4.name,
        ],
    }
    results_path = output_dir / "rnn_ring_results.json"
    results_path.write_text(json.dumps(result, indent=2))

    print(json.dumps({
        "parameters": asdict(parameters),
        "ring_radius": ring_radius,
        "condition_ratio_minus_mu_over_2radius": condition_ratio,
        "fixed_points": serializable_fixed_points(fixed_points),
        "random_convergence": (
            f"{random_convergence_count}/100"
        ),
        "finite_n": finite_metrics,
        "results": str(results_path),
    }, indent=2))


if __name__ == "__main__":
    main()
