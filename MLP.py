import os
import time
import logging
import psutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ========== 配置日志 ==========
os.makedirs('./save', exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = f'./save/mlp_training_{timestamp}.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    logging.info(f"[Memory] {note} - Memory usage: {mem:.2f} MB")

# ========== 数据加载 ==========
def load_and_process_data(feature_path, train_path):
    logging.info("Loading and processing data...")
    log_memory_usage("Before loading data")
    start_time = time.time()

    df_features = pd.read_csv(feature_path)
    df_train = pd.read_csv(train_path)
    df_bert = pd.read_csv('./features/weibo_bert_features_train_32.csv')

    content_features = df_bert['content_feature'].str.split(' ', expand=True).astype(float)
    content_features.columns = [f'feature_{i}' for i in range(32)]
    df_base = pd.concat([df_bert[['uid', 'mid']], content_features], axis=1)

    merged_df = pd.merge(df_train, df_features, on=['uid', 'mid'], how='inner')
    merged_df = pd.merge(merged_df, df_base, on=['uid', 'mid'], how='inner')

    end_time = time.time()
    logging.info(f"Data loaded in {end_time - start_time:.2f}s")
    log_memory_usage("After loading data")
    return merged_df

# ========== 定义MLP ==========
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 输出3个目标
        )

    def forward(self, x):
        return self.model(x)

# ========== 模型训练 ==========
def train_mlp(X, y, params, device):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'])

    model = MLP(input_dim=X.shape[1], hidden_dim=params['hidden_dim'], dropout=params['dropout']).to(device)
    # 加载已有模型
    model_path = f'./save/mlp_model_joint_{timestamp}.pt'
    if os.path.exists(model_path):
        logging.info(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience, wait = 5, 0

    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['epochs']}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logging.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logging.info("Early stopping triggered.")
                break

    return model

# ========== 主训练流程 ==========
def train_and_predict(merged_df, param_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    targets = ['forward_count', 'comment_count', 'like_count']
    features = merged_df.columns.drop(['uid', 'mid', 'time'] + targets)
    X_all = merged_df[features].astype(np.float32).values
    y_all = np.log1p(merged_df[targets].astype(np.float32).values)

    model = train_mlp(X_all, y_all, param_list, device)

    # 模型保存
    model_path = f'./save/mlp_model_joint_{timestamp}.pt'
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")

    # 模型推理
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_all).float().to(device)).cpu().numpy()
        preds = np.expm1(preds).round().clip(0, 100).astype(int)

    result_df = merged_df[['uid', 'mid'] + targets].copy()
    for i, target in enumerate(targets):
        result_df[f'pred_{target}'] = preds[:, i]

    result_path = f'./save/predictions_mlp_joint_{timestamp}.csv'
    result_df.to_csv(result_path, index=False)
    logging.info(f"Predictions saved to {result_path}")
    return result_df

# ========== Main ==========
if __name__ == '__main__':
    feature_path = './features/train_user_mid_feature_scalar.csv'
    train_path = './features/weibo_train.csv'

    param_list = {
        'hidden_dim': 1024,
        'dropout': 0.3,
        'lr': 1e-4,
        'batch_size': 4096,
        'epochs': 200
    }

    try:
        merged_data = load_and_process_data(feature_path, train_path)
        logging.info("Data processed successfully")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

    try:
        result = train_and_predict(merged_data, param_list)
        print("MLP Training completed successfully. Results saved.")
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        raise
