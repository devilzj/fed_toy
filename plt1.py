from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from jax import grad

from federated.learning.algorithms import *
from federated.objectives.quadratic import *
from federated.utils.plotting import *

# Monkey-patch Quadratic to make it compatible with SGD + inject some noise.
noise_scale = 1.0

plt.rc('font', family='Times New Roman')


@functools.partial(jit, static_argnums=(0,))
def _grad(_, params, prng_key, x, **kwargs):
    g = grad(Quadratic.eval, argnums=1)(params, x)
    sg = g + noise_scale * random.normal(prng_key, g.shape)
    return sg


Quadratic._grad = _grad

# Define quadratics.
weights = np.array([1., 1.])
q1 = Quadratic(A=np.diag([2, 15]) - 1.5 * np.ones((2, 2)), b=np.array([5, 10]))
q2 = Quadratic(A=np.diag([15, 2]) - 1.5 * np.ones((2, 2)), b=np.array([10, 5]))
q = create_global_quadratic([q1, q2], weights)

# Solve quadratics exactly.
q1_opt, q2_opt, q_opt = q1.solve(), q2.solve(), q.solve()


def _client_learning_rate_schedule(_):
    return 10 ** (-3 / 2)


def _server_learning_rate_schedule(t):
    return 1.0


def _server_learning_rate_schedule_fedpa(t):
    return 0.1


client_momentum = 0.0
server_momentum = 0.0

fed_avg_solvers = {
    client_steps_per_round: create_fed_avg(
        client_steps_per_round=client_steps_per_round,
        client_learning_rate_schedule=_client_learning_rate_schedule,
        server_learning_rate_schedule=_server_learning_rate_schedule,
        client_momentum=client_momentum,
        server_momentum=server_momentum,
    )
    for client_steps_per_round in [10, 100]
}

# Run federated learning experiments.
seeds = range(3)
num_rounds = 10
num_clients_per_round = 2
for o in [q1, q2]:
    o.data = o.params
    o.num_points = 1
    o.batch_size = 1

# Note: we define specific init states for visualization purposes.
init_states = [
    jnp.array([-40., -20.]),
    jnp.array([-40., 40.]),
    jnp.array([0., 60.]),
    jnp.array([60., 40.]),
    jnp.array([60., -20.]),
    jnp.array([20., -20.]),
]

results = pd.DataFrame()
trajectories = defaultdict(list)
dist_to_opt = defaultdict(list)

# FedAvg.
print("Running FedAvg...")
for steps, solver in fed_avg_solvers.items():
    for i, seed in enumerate(seeds):
        prng_key = random.PRNGKey(seed)
        prng_key, subkey = random.split(prng_key)
        # init_state = random.normal(subkey, (2,)) * 50.
        init_state = init_states[i]
        traj, _ = solver(
            client_objectives=[q1, q2],
            init_state=init_state,
            num_rounds=num_rounds,
            num_clients_per_round=num_clients_per_round,
            prng_key=prng_key,
        )
        key = f"FedAvg - {steps} steps"
        trajectories[key].append([s.x for s in traj])
        dist_to_opt[key].append([np.linalg.norm(s.x - q_opt) for s in traj])
        results = results.append(pd.DataFrame({
            "step": np.arange(len(traj)),
            "dist_to_opt": np.asarray([np.linalg.norm(s.x - q_opt) for s in traj]),
            "method": [key] * len(traj),
            "seed": [seed] * len(traj),
        }), ignore_index=True)

print("Done.")

fig, axes = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)

# Plot quadratics.
ax = axes
ax.set_title("Inter-client inconsistency", fontsize=18)
_, (xlim, ylim) = plot_objective_contours(
    ax=ax,
    objectives=[q1, q2],
    global_objective=q,
    mesh_margins=(25., 25.),
    num_levels=10,
    max_level_coeff=0.02,
    contour_alpha=0.9,
    contour_linewidth=1.0,
    contour_cmaps=("Oranges", "Greens"),
)
ax.set_xlabel(r"$\theta_1$", fontsize=16)
ax.set_ylabel(r"$\theta_2$", fontsize=16)

# # Annotations.
ax.plot(*zip(q1_opt, q2_opt), "--", color="black")
ax.plot(*q1_opt, marker='o', color='orange', markersize=12)
ax.plot(*q2_opt, marker='o', color='g', markersize=12)
ax.plot(*q_opt, marker='*', color="black", markersize=14)

# ax = axes
ax.text(-22, 24, "client 1", color='g', fontsize=20)
ax.text(20, -19, "client 2", color='orange', fontsize=20)
ax.annotate(
    "global optima",
    xy=q_opt + np.array([-1.0, -1.0]),
    xytext=(-22, -20),
    fontsize=20,
    color="black",
    arrowprops=dict(
        arrowstyle="-|>",
        color="black",
        linewidth=1.5,
    ),
    bbox=dict(pad=5, facecolor="none", edgecolor="none"),
)
ax.annotate(
    "local optima",
    xy=q1_opt + np.array([0.5, 1.0]),
    xytext=(13, 28),
    fontsize=20,
    color="white",
    arrowprops=dict(
        arrowstyle="-|>",
        color="black",
        linewidth=1.5,
    ),
    bbox=dict(pad=5, facecolor="none", edgecolor="none"),
)
ax.annotate(
    "local optima",
    xy=q2_opt + np.array([1.0, 1.0]),
    xytext=(13, 31),
    fontsize=20,
    color="black",
    arrowprops=dict(
        arrowstyle="-|>",
        color="black",
        linewidth=1.5,
    ),
    bbox=dict(pad=5, facecolor="none", edgecolor="none"),
)
plt.tight_layout()
plt.twinx()
plt.ylabel(" ")
plt.yticks([])
# fig.savefig("illustration2d.pdf")
plt.show()
print("done")
