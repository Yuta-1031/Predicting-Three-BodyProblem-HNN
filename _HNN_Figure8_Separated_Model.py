# docker run --rm --gpus all -v "%cd%":/app -w /app hnn-project python -u _HNN_Figure8_Separated_Model.py

'''
本学習用スクリプト
'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import subprocess
import sys

# 日本語表示の設定
try:
    import japanize_matplotlib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "japanize-matplotlib"])
    import japanize_matplotlib

CONFIG = {
    "data_path": "Figure8_Dataset.npz",
    "model_path": "Base_hnn_figure8.pth",
    "hidden_dim": 512,    # ノード数
    "lr": 5e-5,           # 学習率
    "epochs": 150000,
    "omega_0": 15.0,      # スケーリング周波数
    "batch_size": 4096,   # バッチ GPUに一度に送るサイズ
    "noise_std": 0.001,   # 入力に与えるノイズ 追加学習する場合、なくてもよい
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

#SIRENの定義
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, omega_0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_f, 1 / in_f)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_f) / omega_0, np.sqrt(6 / in_f) / omega_0)
    def forward(self, x): return torch.sin(self.omega_0 * self.linear(x))

# HNNモデルの定義
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
            nn.Linear(hidden_dim, 1)
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
        
        #1/r の特徴量を追加
        inv_r = 1.0 / (r + 0.01)

        # r と 1/r を結合して 6次元の特徴量にする
        features = torch.cat([r, inv_r], dim=1)

        f_mean = torch.tensor(self.stats['f_mean'], device=q.device).float()
        f_std = torch.tensor(self.stats['f_std'], device=q.device).float()
        f_norm = (features - f_mean) / f_std
        
        return self.v_net(f_norm)

#学習部
def main():
    #データセットの読み込み
    data = np.load(CONFIG["data_path"])
    y_raw, t_raw = data['y'], data['t']
    q_raw, p_raw = y_raw[:, :6], y_raw[:, 6:]
    dp_dt_target = np.gradient(p_raw, axis=0) / (t_raw[1] - t_raw[0])
    
    # 相対距離の計算
    q1_r, q2_r, q3_r = q_raw[:, 0:2], q_raw[:, 2:4], q_raw[:, 4:6]
    r_raw = np.concatenate([np.linalg.norm(q1_r-q2_r, axis=1, keepdims=True),
                             np.linalg.norm(q2_r-q3_r, axis=1, keepdims=True),
                             np.linalg.norm(q3_r-q1_r, axis=1, keepdims=True)], axis=1)
    
    # 1/r も計算して結合した特徴量を作る
    inv_r_raw = 1.0 / (r_raw + 0.01)
    features_raw = np.concatenate([r_raw, inv_r_raw], axis=1)

    stats = {
        'f_mean': np.mean(features_raw, axis=0), 
        'f_std': np.std(features_raw, axis=0) + 1e-6
    }

    #HNNモデル読み込み
    loader = DataLoader(TensorDataset(torch.tensor(q_raw).float(), torch.tensor(p_raw).float(),
        torch.tensor(dp_dt_target).float()), batch_size=CONFIG["batch_size"], shuffle=True)
    
    model = SeparableFigure8HNN(CONFIG["hidden_dim"], CONFIG["omega_0"], stats).to(CONFIG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000, verbose=True)
    best_loss = float('inf')

    print(f"--- Stage 1---", flush=True)

    for epoch in range(CONFIG["epochs"]):
        epoch_l_grad, epoch_l_zero = 0, 0
        for b_q, b_p, b_target in loader:
            b_q = b_q.to(CONFIG["device"]).requires_grad_(True)
            b_p = b_p.to(CONFIG["device"])
            b_target = b_target.to(CONFIG["device"])

            bq1, bq2, bq3 = b_q[:, 0:2], b_q[:, 2:4], b_q[:, 4:6]
            br12 = torch.norm(bq1 - bq2, dim=1, keepdim=True)
            br23 = torch.norm(bq2 - bq3, dim=1, keepdim=True)
            br31 = torch.norm(bq3 - bq1, dim=1, keepdim=True)
            q_noisy = b_q + torch.randn_like(b_q) * CONFIG["noise_std"]

            V = model(q_noisy)
            T = 0.5 * torch.sum(b_p**2, dim=1, keepdim=True)
            grad_V = torch.autograd.grad(V.sum(), b_q, create_graph=True)[0]
            
            # Loss計算
            with torch.no_grad():
                # バッチに最小距離
                br_min = torch.min(torch.min(br12, br23), br31)
                # r_minが小さい（接近している）ほど重みを大きくする
                w_hard = 1.0 / (br_min + 0.2) 
                w_hard = w_hard / w_hard.mean() # 平均で正規化して学習を安定させる

            # 勾配ロスに難所の重みを適用
            l_grad = torch.mean(w_hard * torch.sum((-grad_V - b_target)**2, dim=1))

            # 次の点とを滑らかにつなぐことを要請する
            #l_valley = torch.mean((V - model(q_noisy))**2)
            
            #重心が一定になるように
            #l_com = torch.mean(torch.sum(grad_V.view(-1, 3, 2), dim=1)**2)

            #T+V=0を初期値に
            l_zero = torch.mean((T + V)**2)

            # 同分散不確実性による自動統合 動的重み付け しかしうまくいかなかった
            # Loss = exp(-s) * L + 0.5 * s
            #loss_grad = torch.exp(-model.log_vars[0]) * (2000.0 * l_grad) + 0.5 * model.log_vars[0]
            #loss_valley = torch.exp(-model.log_vars[1]) * l_valley + 0.5 * model.log_vars[1]
            #loss_zero = torch.exp(-model.log_vars[2]) * l_zero + 0.5 * model.log_vars[2]
            #loss_com = torch.exp(-model.log_vars[3]) * l_com + 0.5 * model.log_vars[3]

            loss = 1000 * l_grad + 500 * l_zero
            #2:1だからといって2*l_grad + l_zeroとはしない
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #値が急に跳ねないようにする
            optimizer.step()
            epoch_l_grad += l_grad.item()
            epoch_l_zero += l_zero.item()

        if (epoch+1) % 1000 == 0:
            avg_l_grad = epoch_l_grad / len(loader)
            avg_l_zero = epoch_l_zero / len(loader)
            current_total = 1000.0 * avg_l_grad + 500.0* avg_l_zero
            scheduler.step(current_total)
            
            if current_total < best_loss:
                best_loss = current_total
                torch.save({'model_state': model.state_dict(), 'stats': stats}, CONFIG["model_path"])
                print(f"--- Best Model Saved (Total Loss: {best_loss:.2e}) ---")

            print(
                f"Epoch {epoch+1:6d} | Grad_MSE: {avg_l_grad:.2e} | Zero_MSE: {avg_l_zero:.2e} | LR: {optimizer.param_groups[0]['lr']:.1e}")

    print("Stage1 Model Saved with Learned Weights.", flush=True)

if __name__ == "__main__": main()