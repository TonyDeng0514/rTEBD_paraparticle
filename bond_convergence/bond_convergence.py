import argparse
import numpy as np
from pathlib import Path
from paraparticles.MPDO import MPDO

RESULTS_DIR = Path(__file__).parent / "results"

# parameters
g = 1
L = 6
N = 4
Na = 2
t_hop = 1.0
W = 0.2
dt = 0.005
T = 2.0
seed = 1234
rng = np.random.default_rng(seed)
parser = argparse.ArgumentParser()
parser.add_argument("--chi", type=int, required=True, help="bond dimension")
chi = parser.parse_args().chi
Nt = int(T / dt)
t_grid = np.linspace(0, T + dt, Nt + 1)

tr_TB = []
ni = np.zeros((L, Nt + 1))

mps_evolve = MPDO(L, N, Na, t_hop, W, dt, T, chi, seed, g)
for j in range(L):
    ni[j][0] = mps_evolve.ni_persite[j]

for i in range(1, Nt + 1):
    mps_evolve.sweepU()
    for j in range(L):
        ni[j][i] = mps_evolve.ni_persite[j]
    tr_TB.append(mps_evolve.tr_TEBD)

tr_TB = np.array(tr_TB)  # convert list → array before saving

# Save everything in one .npz file
np.savez(
    RESULTS_DIR / f"ni_L{L}_N{N}_chi{chi}_g{g}.npz",
    ni=ni,
    tr_TB=tr_TB,
    t_grid=t_grid,
    # store parameters as 0-d arrays so they round-trip cleanly
    L=L, N=N, Na=Na, t_hop=t_hop, W=W, dt=dt, T=T, chi=chi, g=g, seed=seed,
)
print("Saved.")