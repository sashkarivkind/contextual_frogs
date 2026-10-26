"""Low-rank Hebbian dynamics without constructing an n-by-n weight matrix.

The weight operator is represented by a list of rank-one components ``(u, v)``:

    W = sum(outer(u, v) for u, v in components)

Consequently, ``x @ W`` is evaluated as ``sum((x @ u) * v)``.  The simulation
stores only projections of each state onto the Hebbian patterns, not the full
state trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
DISTRIBUTION = "laplace"       # "gaussian", "laplace", or "uniform"
SAMPLE_FROM = "equally_spaced_from_circle"
N = 2_000_000                     # state/pattern dimension
P = 2                          # number of stored patterns
N_INITIAL_STATES = 40
NUM_STEPS = 100
GAIN = 4.6
SEED = None                    # set to an integer for reproducible results
DTYPE = np.float64             # np.float32 roughly halves working memory

# Run one condition for every entry. None means no selected-pattern change.
# Example: [None, 0, 1] compares baseline with strengthening patterns 0 and 1.
DIMENSIONS_TO_PERTURB = [None]
PATTERN_STRENGTH_DELTA = 0.0   # 0.10 strengthens the selected pattern by 10%

# Optional explicit perturbation alpha * q q.T, where
# q = (xi[OUTER_PERTURB_PATTERN] - OUTER_PERTURB_OFFSET) / sqrt(N).
OUTER_PERTURB_AMP = 0.0
OUTER_PERTURB_PATTERN = 0
OUTER_PERTURB_OFFSET = 0.4


def generate_patterns(distribution, num_patterns, pattern_size, rng, dtype):
    """Generate a (num_patterns, pattern_size) array."""
    if distribution == "gaussian":
        values = rng.standard_normal((num_patterns, pattern_size))
    elif distribution == "laplace":
        values = rng.laplace(0.0, 1.0, (num_patterns, pattern_size))
    elif distribution == "uniform":
        values = rng.uniform(-2.0, 2.0, (num_patterns, pattern_size))
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")
    return values.astype(dtype, copy=False)


def sigmoid(x):
    """Numerically stable-enough logistic sigmoid for this experiment."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def sample_initial_mixtures(sample_from, num_states, num_patterns, rng, dtype):
    """Return normalized pattern-mixture coefficients with shape (k, p)."""
    if sample_from == "positive_orthant":
        bound_off_zero = 0.1
        k = rng.random((num_states, num_patterns)) * (1.0 - bound_off_zero)
        k += bound_off_zero
    elif sample_from == "random_everywhere":
        k = rng.random((num_states, num_patterns)) - 0.5
    elif sample_from == "equally_spaced_from_circle":
        if num_patterns != 2:
            raise ValueError("equally_spaced_from_circle requires num_patterns == 2")
        angles = np.linspace(0.0, 2.0 * np.pi, num_states, endpoint=False)
        k = np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        raise ValueError(f"Unsupported initial-state sampling mode: {sample_from}")

    k = k.astype(dtype, copy=False)
    norms = np.linalg.norm(k, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero initial-mixture vector")
    return k / norms


def hebbian_rank_one_components(
    xi, pattern_weights=None, output_transform=None, normalize=True
):
    """Represent the Hebbian matrix as a list of ``(u, v)`` components.

    This represents

        W = output_patterns.T @ pattern_weights @ xi / n,
        output_patterns = output_transform @ xi.

    Only p components are needed even when pattern_weights is a dense p-by-p
    matrix: component a uses output_patterns[a] as u and
    pattern_weights[a] @ xi as v.
    """
    xi = np.asarray(xi)
    if xi.ndim != 2:
        raise ValueError("xi must have shape (num_patterns, pattern_size)")
    num_patterns, pattern_size = xi.shape

    if pattern_weights is None:
        pattern_weights = np.eye(num_patterns, dtype=xi.dtype)
    else:
        pattern_weights = np.asarray(pattern_weights, dtype=xi.dtype)
        if pattern_weights.shape != (num_patterns, num_patterns):
            raise ValueError(
                "pattern_weights must have shape "
                f"{(num_patterns, num_patterns)}, got {pattern_weights.shape}"
            )

    if output_transform is None:
        output_patterns = xi
    else:
        output_transform = np.asarray(output_transform, dtype=xi.dtype)
        if output_transform.shape != (num_patterns, num_patterns):
            raise ValueError(
                "output_transform must have shape "
                f"{(num_patterns, num_patterns)}, got {output_transform.shape}"
            )
        output_patterns = output_transform @ xi

    scale = 1.0 / pattern_size if normalize else 1.0
    weighted_inputs = (pattern_weights @ xi) * scale
    return [(output_patterns[a], weighted_inputs[a]) for a in range(num_patterns)]


def apply_rank_one_operator(x, components):
    """Compute x @ W for W = sum(outer(u, v)) without materializing W."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must have shape (batch_size, state_size)")

    result = np.zeros_like(x)
    for u, v in components:
        u = np.asarray(u)
        v = np.asarray(v)
        if u.shape != (x.shape[1],) or v.shape != (x.shape[1],):
            raise ValueError(
                "Every rank-one component must contain two vectors of length "
                f"{x.shape[1]}; got {u.shape} and {v.shape}"
            )
        result += (x @ u)[:, None] * v[None, :]
    return result


def run_dynamics_projected(x0, components, projection_patterns, phi, g=2.0,
                           num_steps=100):
    """Run dynamics while retaining only pattern projections over time.

    The current full state is necessarily retained because an elementwise
    nonlinear phi generally moves it outside the pattern span.  Full states
    and preactivations from earlier steps are discarded.
    """
    x = np.asarray(x0).copy()
    projection_patterns = np.asarray(projection_patterns)
    num_patterns = projection_patterns.shape[0]
    projections = np.empty(
        (num_steps + 1, x.shape[0], num_patterns), dtype=x.dtype
    )
    projections[0] = x @ projection_patterns.T

    for t in range(1, num_steps + 1):
        h = g * apply_rank_one_operator(x, components)
        x = phi(h)
        projections[t] = x @ projection_patterns.T

    return x, projections


def make_condition_components(xi, selected_pattern, pattern_strength_delta,
                              outer_perturb_amp):
    """Build the low-rank component list for one experimental condition."""
    num_patterns = xi.shape[0]
    pattern_weights = np.eye(num_patterns, dtype=xi.dtype)
    if selected_pattern is not None:
        if not 0 <= selected_pattern < num_patterns:
            raise ValueError(f"Pattern index {selected_pattern} is out of range")
        pattern_weights[selected_pattern, selected_pattern] += pattern_strength_delta

    components = hebbian_rank_one_components(xi, pattern_weights=pattern_weights)

    if outer_perturb_amp != 0.0:
        q = (xi[OUTER_PERTURB_PATTERN] - OUTER_PERTURB_OFFSET) / np.sqrt(xi.shape[1])
        # alpha*q*q.T is represented as outer(alpha*q, q).
        components.append((outer_perturb_amp * q, q))

    return components


def plot_results(all_projections, xi, dimensions_to_perturb):
    """Reproduce the trajectory, angle, and element-distribution plots."""
    if xi.shape[0] < 2:
        raise ValueError("The trajectory plot requires at least two patterns")

    max_abs = max(np.max(np.abs(projections[1:])) for projections in all_projections)
    lim = 1.1 * max_abs if max_abs > 0 else 1.0
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(all_projections)))

    plt.figure(figsize=(9, 3))
    for condition, (projections, selected_pattern) in enumerate(
        zip(all_projections, dimensions_to_perturb)
    ):
        projs = projections[1:]
        color = colors[condition]
        for i in range(projs.shape[1]):
            plt.subplot(1, 3, 1)
            plt.plot(projs[:, i, 0], projs[:, i, 1], ".-", alpha=0.1, color=color)
            plt.plot(projs[-1, i, 0], projs[-1, i, 1], "x", alpha=0.1, color=color)
            plt.plot(projs[0, i, 0], projs[0, i, 1], "o", alpha=0.1, color=color)

            plt.subplot(1, 3, 2)
            # Argument order intentionally matches the original notebook.
            angles = np.arctan2(projs[:, i, 0], projs[:, i, 1])
            plt.plot(angles, ".-", alpha=0.1, color=color)

    plt.subplot(1, 3, 1)
    plt.xlabel("projection on pattern 1 [au]", fontsize=12)
    plt.ylabel("projection on pattern 2 [au]", fontsize=12)
    if len(colors) > 1:
        arrow_color_1 = colors[1]
        arrow_color_2 = colors[min(2, len(colors) - 1)]
    else:
        arrow_color_1 = "r"
        arrow_color_2 = "g"
    plt.arrow(0, 0, lim * 0.8, 0, head_width=lim * 0.05,
              head_length=lim * 0.1, fc=arrow_color_1, ec=arrow_color_1)
    plt.arrow(0, 0, 0, lim * 0.8, head_width=lim * 0.05,
              head_length=lim * 0.1, fc=arrow_color_2, ec=arrow_color_2)
    plt.box(False)

    plt.subplot(1, 3, 2)
    plt.xlim([0, min(50, NUM_STEPS)])
    plt.xlabel("dynamics step")
    plt.ylabel("mixture angle [rad]")

    plt.subplot(1, 3, 3)
    plt.hist(xi.ravel(), bins=50, color="gray", alpha=0.7, density=True)
    plt.title("pattern elements\ndistribution")
    plt.tight_layout()
    plt.show()


def main():
    rng = np.random.default_rng(SEED)
    xi = generate_patterns(DISTRIBUTION, P, N, rng, DTYPE)
    k = sample_initial_mixtures(
        SAMPLE_FROM, N_INITIAL_STATES, P, rng, DTYPE
    )
    x0 = k @ xi

    all_projections = []
    final_states = []
    for selected_pattern in DIMENSIONS_TO_PERTURB:
        components = make_condition_components(
            xi,
            selected_pattern=selected_pattern,
            pattern_strength_delta=PATTERN_STRENGTH_DELTA,
            outer_perturb_amp=OUTER_PERTURB_AMP,
        )
        final_state, projections = run_dynamics_projected(
            x0,
            components,
            projection_patterns=xi,
            phi=sigmoid,
            g=GAIN,
            num_steps=NUM_STEPS,
        )
        final_states.append(final_state)
        all_projections.append(projections)

    plot_results(all_projections, xi, DIMENSIONS_TO_PERTURB)
    return final_states, all_projections, xi


if __name__ == "__main__":
    main()
