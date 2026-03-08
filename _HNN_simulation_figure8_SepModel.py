# docker run --rm --gpus all -v "%cd%":/app -w /app hnn-project python -u _hnn_simulation_figure8_SepModel.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

# 日本語表示の設定
try:
    import japanize_matplotlib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "japanize-matplotlib"])
    import japanize_matplotlib

CONFIG = {
    "model_path": "polished_boss_hnn_figure8.pth",
    "hidden_dim": 512,
    "omega_0": 15.0,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- モデル定義 学習で用いたものと一致させる ---
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, omega_0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_f, 1 / in_f)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_f) / omega_0, 
                                            np.sqrt(6 / in_f) / omega_0)
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SeparableFigure8HNN(nn.Module):
    def __init__(self, hidden_dim, omega_0, stats):
        super().__init__()
        self.stats = stats
        self.v_net = nn.Sequential(
            SirenLayer(6, hidden_dim, omega_0, is_first=True),
            SirenLayer(hidden_dim, hidden_dim, omega_0),
            SirenLayer(hidden_dim, hidden_dim, omega_0),
            SirenLayer(hidden_dim, hidden_dim, omega_0),
            SirenLayer(hidden_dim, hidden_dim, omega_0),
            SirenLayer(hidden_dim, hidden_dim, omega_0),
            SirenLayer(hidden_dim, hidden_dim, omega_0),
            nn.Linear(hidden_dim, 1) #Linear
        )

        self.log_vars = nn.Parameter(torch.zeros(4))

        with torch.no_grad():
            torch.nn.init.uniform_(self.v_net[-1].weight, -1e-4, 1e-4)
            torch.nn.init.zeros_(self.v_net[-1].bias)


    def forward(self, q):
        q1, q2, q3 = q[:, 0:2], q[:, 2:4], q[:, 4:6]
        r = torch.cat([torch.norm(q1-q2, dim=1, keepdim=True),
                       torch.norm(q2-q3, dim=1, keepdim=True),
                       torch.norm(q3-q1, dim=1, keepdim=True)], dim=1)
        
        inv_r = 1.0 / (r + 0.01)
        features = torch.cat([r, inv_r], dim=1)

        f_mean = torch.tensor(self.stats['f_mean'], device=q.device).float()
        f_std = torch.tensor(self.stats['f_std'], device=q.device).float()
        f_norm = (features - f_mean) / f_std
        
        return self.v_net(f_norm)

#吉田4次シンプレクティック積分器
def yoshida_4th_step(model, q, p, dt):
    w1 = 1.3512071919596575
    w0 = -1.702414383919315
    c = np.array([w1/2, (w1+w0)/2, (w1+w0)/2, w1/2])
    d = np.array([w1, w0, w1, 0])

    q_curr, p_curr = q.clone(), p.clone()

    for i in range(4):
        # 位置q の更新 (dq/dt = p)
        q_curr = q_curr + c[i] * dt * p_curr
        
        if d[i] == 0: break

        # 運動量p の更新 (dp/dt = -dV/dq)
        with torch.enable_grad():
            q_in = q_curr.detach().requires_grad_(True)
            V = model(q_in)
            grad_V = torch.autograd.grad(V.sum(), q_in)[0]
        p_curr = p_curr - d[i] * dt * grad_V

    # ハミルトニアン投影
    alpha = 0.0005
    with torch.enable_grad():
        u_q = q_curr.detach().requires_grad_(True)
        u_p = p_curr.detach().requires_grad_(True)
        T = 0.5 * torch.sum(u_p**2)
        V = model(u_q)
        H = T + V
        
        if torch.abs(H) > 1e-7:
            grad_H_q = torch.autograd.grad(H, u_q, retain_graph=True)[0]
            grad_H_p = torch.autograd.grad(H, u_p)[0]
            norm_sq = torch.sum(grad_H_q**2) + torch.sum(grad_H_p**2) + 1e-9

            correction = H / norm_sq
            correction = torch.clamp(correction, -0.1, 0.1)

            q_curr = q_curr - alpha * correction * grad_H_q
            p_curr = p_curr - alpha * correction * grad_H_p # typo fix: should use p_curr

    return q_curr.detach(), p_curr.detach()

def main():
    # 学習データのロード
    checkpoint = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
    stats = checkpoint['stats']

    # モデルを作る
    model = SeparableFigure8HNN(CONFIG["hidden_dim"], CONFIG["omega_0"], stats).to(CONFIG["device"])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 初期値 教師データと一致させる
    x1, y1 = 0.97000436, -0.24308753
    v1, v2 = 0.466203685, 0.43236573
    q = torch.tensor([[x1, y1], [-x1, -y1], [0.0, 0.0]], device=CONFIG["device"]).float().view(1, 6)
    p = torch.tensor([[v1, v2], [v1, v2], [-2*v1, -2*v2]], device=CONFIG["device"]).float().view(1, 6)

    dt = 0.0001 #刻み幅
    max_steps = 100000 #ステップ シミュレーションの総時間はdt × max_steps
    history = []
    energies = []

    print(f"--- シミュレーション開始 ---")
    for step in range(max_steps):
        q, p = yoshida_4th_step(model, q, p, dt)
        
        history.append(torch.cat([q, p], dim=1).cpu().numpy().flatten())
        
        with torch.no_grad():
            V = model(q)
            T = 0.5 * torch.sum(p**2)
            energies.append((T + V).item())

        if (step+1) % 10000 == 0: print(f"Step {step+1} / {max_steps} 完了...")

    history = np.array(history)
    
    # 軌道プロット
    plt.figure(figsize=(10, 8))
    for i in range(3):
        plt.plot(history[:, 2*i], history[:, 2*i+1], label=f'天体 {i+1}', alpha=0.8)
    plt.axis('equal'); plt.grid(True); plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("8の字軌道のシミュレーション結果")
    plt.savefig("separable_hnn_orbit_result.png")

    # エネルギープロット
    period_const = 6.32591398
    periods_x = np.arange(len(energies)) * dt / period_const
    plt.figure(figsize=(10, 4))
    plt.plot(periods_x, energies)
    plt.title("ハミルトニアンの安定性")
    plt.xlabel("周期 [cycles]")
    plt.ylabel("ハミルトニアン $H$")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig("separable_hnn_energy_stability.png")
    
    print("完了。画像を確認してください。")

if __name__ == "__main__":
    main()