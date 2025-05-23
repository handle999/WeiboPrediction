import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import time
import logging
import os
import psutil
from contextlib import redirect_stdout

# 配置日志记录
os.makedirs('./save', exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = f'./save/training_{timestamp}.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    logging.info(f"[Memory] {note} - Memory usage: {mem:.2f} MB")

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
    logging.info(f"Data loaded and processed in {end_time - start_time:.2f} seconds")
    log_memory_usage("After loading data")
    return merged_df

def train_and_predict(merged_df, oob=False):
    os.makedirs('./save', exist_ok=True)

    targets = ['forward_count', 'comment_count', 'like_count']
    features = merged_df.columns.drop(['uid', 'mid', 'time'] + targets)
    X = merged_df[features]
    predictions = {}
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for target in targets:
        log_file_path = f'./save/model_{target}_{timestamp}.txt'
        with open(log_file_path, 'w') as log_file:
            logging.info(f"Training model for {target}")
            with redirect_stdout(log_file):
                print(f"===== Training model for {target} =====")
                start_time = time.time()
                log_memory_usage("Before training")

                y = merged_df[target]
                y_trans = np.log1p(y)

                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_trans, test_size=0.2, random_state=42
                )

                # ✅ 固定的参数配置
                model = RandomForestRegressor(
                    n_estimators=1000,
                    max_depth=50,
                    min_samples_leaf=10,
                    min_samples_split=50,
                    oob_score=oob,
                    random_state=42,
                    n_jobs=-1,
                    max_features="sqrt"
                )

                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                rmse = mean_squared_error(y_val, preds, squared=False)
                print(f"Validation RMSE for {target}: {rmse:.4f}")

                model_filename = f'./save/model_{target}_{timestamp}.pkl'
                joblib.dump(model, model_filename)
                print(f"Model saved to {model_filename}")

                end_time = time.time()
                print(f"\nTotal time: {end_time - start_time:.2f} seconds")
                log_memory_usage("After training")

                y_pred = model.predict(X)
                predictions[target] = np.expm1(y_pred).round().clip(0, 100).astype(int)

    result_df = merged_df[['uid', 'mid'] + targets].copy()
    result_df['pred_forward_count'] = predictions['forward_count']
    result_df['pred_comment_count'] = predictions['comment_count']
    result_df['pred_like_count'] = predictions['like_count']

    result_filename = f'./save/predictions_{timestamp}.csv'
    result_df.to_csv(result_filename, index=False)
    logging.info(f"Predictions saved to {result_filename}")
    return result_df

if __name__ == '__main__':
    feature_path = './features/train_user_mid_feature_scalar.csv'
    train_path = './features/weibo_train.csv'

    try:
        merged_data = load_and_process_data(feature_path, train_path)
        logging.info("Data processed successfully")
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        raise

    try:
        result = train_and_predict(merged_data)
        print("Training completed successfully. Results saved to ./save/")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
