import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ax = plt.gca()
swap_matrix = np.loadtxt(f"experiments/south_america/results/test/K3/mc3_swaps_K3_0.txt", dtype=int)
n_chains = len(swap_matrix)
swap_mask = np.tri(n_chains, n_chains, 0, dtype=bool)

sns.heatmap(swap_matrix, cmap="inferno", annot=True, fmt=".0f", mask=swap_mask, vmin=0, ax=ax)
ax.set_title("Number of swaps between chains", pad=20)
ax.set_ylabel("Chain 1 index", labelpad=10)
ax.set_xlabel("Chain 2 index", labelpad=10)
plt.tight_layout()
plt.show()
