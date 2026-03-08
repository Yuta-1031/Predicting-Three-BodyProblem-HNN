# docker run --rm --gpus all -v "%cd%":/app -w /app hnn-project python -u _HNN_Figure8_FineTune.py

'''
追加学習用スクリプト
'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

CONFIG = {
    "base_model": "Base_hnn_figure8.pth", # Stage 1の成果をロード
    "save_model": "polished_hnn_figure8.pth",
    "data_path": "Figure8_Dataset.npz",
    "hidden_dim": 512,
    "lr": 0.1,           # 学習率 Stage1の学習率とは扱いが違うので注意
    "epochs": 1000,
    "omega_0": 15.0,
    "batch_size": 4096,
    "noise_std": 0.001,   # 微小なノイズで表面の微振動を消す
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

#本学習でのモデルと一致させる。
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
        
        #1/r の特徴量を追加 (安定のため微小値を加算)
        inv_r = 1.0 / (r + 0.01)
        # r と 1/r を結合して 6次元の特徴量にする
        features = torch.cat([r, inv_r], dim=1)

        f_mean = torch.tensor(self.stats['f_mean'], device=q.device).double()
        f_std = torch.tensor(self.stats['f_std'], device=q.device).double()
        f_norm = (features - f_mean) / f_std
        
        return self.v_net(f_norm)

def main():
    data = np.load(CONFIG["data_path"])
    y_raw, t_raw = data['y'], data['t']
    q_raw, p_raw = y_raw[:, :6], y_raw[:, 6:]
    dp_dt_target = np.gradient(p_raw, axis=0) / (t_raw[1] - t_raw[0])
    
    checkpoint = torch.load(CONFIG["base_model"], map_location=CONFIG["device"])
    stats = checkpoint['stats']
    
    full_batch_size = len(q_raw)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(q_raw).double(), 
            torch.tensor(p_raw).double(),
            torch.tensor(dp_dt_target).double()
            ), 
        #batch_size=CONFIG["batch_size"],
        batch_size = full_batch_size,
        shuffle=False)
    
    model = SeparableFigure8HNN(
        CONFIG["hidden_dim"], 
        CONFIG["omega_0"], 
        stats).to(CONFIG["device"]).double()
    
    model.load_state_dict(checkpoint['model_state'])

    # --- main関数内の model.load_state_dict(checkpoint['model_state']) の直後 ---
    with torch.no_grad():
        has_nan = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"重み '{name}' に nan が含まれています.ただし.pthファイルを読み込んでください。")
                has_nan = True
        if not has_nan:
            print("ロードされたモデルは正常です.")
    #optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"]) #adam-ver

    # L-BFGS ver
    optimizer_L = torch.optim.LBFGS(
        model.parameters(), 
        lr=CONFIG["lr"], 
        max_iter=40, 
        history_size=20,
        line_search_fn='strong_wolfe'
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_L, mode='min', factor=0.5, patience=3000, verbose=True)

    best_loss = float('inf')

    print(f"--- Stage 2: 研磨開始 ---")

    for epoch in range(CONFIG["epochs"]):
        epoch_l_grad, epoch_l_zero, epoch_l_valley = 0, 0, 0
        
        with torch.no_grad():
            for p in model.parameters():
                if torch.isnan(p).any():
                    print("nan が発生しました。学習率をさらに下げてください。")
                    return
            
        for b_q, b_p, b_target in loader:
            b_q = b_q.to(CONFIG["device"]).requires_grad_(True)
            b_p, b_target = b_p.to(CONFIG["device"]), b_target.to(CONFIG["device"])
            
            current_l_grad = 0.0; current_l_zero = 0.0; current_l_valley = 0.0
            def closure():
                nonlocal current_l_grad, current_l_zero, current_l_valley
                optimizer_L.zero_grad()
                V = model(b_q)
                T = 0.5 * torch.sum(b_p**2, dim=1, keepdim=True)
                grad_V = torch.autograd.grad(V.sum(), b_q, create_graph=True)[0]
                        
                # 物理誤差と難所重み付け
                bq1, bq2, bq3 = b_q[:, 0:2], b_q[:, 2:4], b_q[:, 4:6]
                br_min = torch.min(torch.norm(bq1-bq2,dim=1), torch.min(torch.norm(bq2-bq3,dim=1), torch.norm(bq3-bq1,dim=1)))
                w_hard = 1.0 / (br_min + 0.2)
                w_hard = w_hard / w_hard.mean()
                
                l_grad = torch.mean(w_hard * torch.sum((-grad_V - b_target)**2, dim=1))
                l_zero = torch.mean((T + V)**2)

                q_noisy = b_q + torch.randn_like(b_q) * CONFIG["noise_std"]
                l_valley = torch.mean((V - model(q_noisy))**2)
                
                loss = 1000.0 * l_grad + 2000.0 * l_zero + 0.5* l_valley
                loss.backward()

                current_l_grad = l_grad.item()
                current_l_zero = l_zero.item()
                current_l_valley = l_valley.item()

                return loss
            
            #optimizer.zero_grad(); loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 命綱
            optimizer_L.step(closure)
            
            epoch_l_grad += current_l_grad
            epoch_l_zero += current_l_zero
            epoch_l_valley += current_l_valley

        if (epoch+1) % 100 == 0:
            avg_g, avg_z, avg_v = epoch_l_grad/len(loader), epoch_l_zero/len(loader), epoch_l_valley/len(loader)
            current_total = 1000.0 * avg_g + 2000.0 * avg_z + 0.5 * avg_v
            scheduler.step(current_total)

            if current_total < best_loss:
                best_loss = current_total
                torch.save({'model_state': model.state_dict(), 'stats': stats}, CONFIG["save_model"])
                print(f"--- Best Model Saved (Total Loss: {best_loss:.2e}) ---")

            print(f"Epoch {epoch+1:5d} | Grad: {avg_g:.2e} | Zero: {avg_z:.2e} | Valley: {avg_v:.2e} | LR: {optimizer_L.param_groups[0]['lr']:.1e}")

    print(f"研磨完了: {CONFIG['save_model']}")

if __name__ == "__main__": main()