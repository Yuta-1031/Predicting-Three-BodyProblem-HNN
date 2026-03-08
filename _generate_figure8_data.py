
#docker run --rm --gpus all -v "%cd%":/app -w /app hnn-project python -u _generate_figure8_data.py
import subprocess
import sys
try:
    import japanize_matplotlib
except ImportError:
    print("japanize-matplotlib をコンテナ内にインストールしています...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "japanize-matplotlib"])
    import japanize_matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def n_body_hamiltonian(t, y):
    q = y[:6].reshape(3, 2)
    p = y[6:].reshape(3, 2)
    m = np.array([1.0, 1.0, 1.0])
    G = 1.0
    
    dqdt = p / m[:, np.newaxis]
    dpdt = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i != j:
                r = q[j] - q[i]
                dist = np.linalg.norm(r)
                dpdt[i] += G * m[i] * m[j] * r / (dist**3 + 1e-12)
    return np.concatenate([dqdt.flatten(), dpdt.flatten()])

# 初期値 (Chenciner & Montgomery)
x1, y1 = 0.97000436, -0.24308753
v1, v2 = 0.466203685, 0.43236573
initial_state = [x1, y1, -x1, -y1, 0, 0, v1, v2, v1, v2, -2*v1, -2*v2]

# 💡 3周期
num_cycles = 3
single_period = 6.3259
t_end = single_period * num_cycles
num_points = 10000 * num_cycles 

t_span = (0, t_end)
t_eval = np.linspace(0, t_end, num_points)

print(f"Generating 3 cycles of data...")
sol = solve_ivp(n_body_hamiltonian, t_span, initial_state, t_eval=t_eval, rtol=1e-13, atol=1e-13)

np.savez("Figure8_Dataset.npz", y=sol.y.T, t=sol.t)
print(f"Dataset saved: {len(sol.t)} samples.")

# プロット
plt.figure(figsize=(8, 6))
plt.plot(sol.y[0, :], sol.y[1, :], 'r', alpha=0.5, label='天体 1')
plt.plot(sol.y[2, :], sol.y[3, :], 'b', alpha=0.5, label='天体 2')
plt.plot(sol.y[4, :], sol.y[5, :], 'g', alpha=0.5, label='天体 3')

plt.axis('equal')
plt.title("3-Cycle Ground Truth")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("figure8_ground_truth_3.png")
plt.axis('equal')