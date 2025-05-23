import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import glob
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            nn.Linear(hidden_dim, 3)  # 3 outputs: forward, comment, like
        )

    def forward(self, x):
        return self.model(x)

def load_latest_model(model_dir='./save'):
    """加载最新的 MLP 模型"""
    model_files = glob.glob(os.path.join(model_dir, 'mlp_model_*.pt'))
    if not model_files:
        raise FileNotFoundError("No model checkpoint found.")
    latest_model_path = sorted(model_files, key=os.path.getctime, reverse=True)[0]
    logging.info(f"Loading model from {latest_model_path}")

    model = MLP(input_dim=68, hidden_dim=1024)  # 固定输入维度为 68
    model.load_state_dict(torch.load(latest_model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def predict_pipeline():
    predict_path = './features/weibo_predict.csv'
    feature_path = './features/predict_user_mid_feature_scalar.csv'
    output_dir = './save'

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: 读取特征和原始数据
        logging.info("Loading data...")
        df_features = pd.read_csv(feature_path)
        df_train = pd.read_csv(predict_path)
        df_bert = pd.read_csv('./features/weibo_bert_features_predict_32.csv')

        content_features = df_bert['content_feature'].str.split(' ', expand=True).astype(float)
        content_features.columns = [f'feature_{i}' for i in range(32)]
        df_base = pd.concat([df_bert[['uid', 'mid']], content_features], axis=1)

        # Step 2: 合并
        merged_df = pd.merge(df_train, df_features, on=['uid', 'mid'], how='inner')
        merged_df = pd.merge(merged_df, df_base, on=['uid', 'mid'], how='inner')

        features = merged_df.columns.drop(['uid', 'mid', 'time'])
        X_raw = merged_df[features].values.astype(np.float32)

        # Step 3: 加载模型
        model = load_latest_model(output_dir)
        X_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)

        # Step 4: 模型预测
        logging.info("Predicting...")
        with torch.no_grad():
            outputs = model(X_tensor).cpu().numpy()
            preds = np.expm1(outputs).round().clip(0, 100).astype(int)

        # Step 5: 添加预测结果
        merged_df['pred_forward_count'] = preds[:, 0]
        merged_df['pred_comment_count'] = preds[:, 1]
        merged_df['pred_like_count'] = preds[:, 2]

        # Step 6: 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'predictions_{timestamp}.csv')
        merged_df[['uid', 'mid', 'pred_forward_count', 'pred_comment_count', 'pred_like_count']].to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise

if __name__ == '__main__':
    predict_pipeline()