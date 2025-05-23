import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
import time
import logging
import os
import psutil
import optuna
from optuna.integration import LightGBMPruningCallback
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
    """加载并合并所有特征"""
    logging.info("Loading and processing data...")
    log_memory_usage("Before loading data")
    start_time = time.time()

    # 加载基础特征
    df_features = pd.read_csv(feature_path)
    content_features = df_features['content_feature'].str.split(' ', expand=True).astype(float)
    content_features.columns = [f'feature_{i}' for i in range(32)]
    df_base = pd.concat([df_features[['uid', 'mid']], content_features], axis=1)

    # 加载其他特征
    df_time = pd.read_csv('./features/train_time_features.csv')
    df_text1 = pd.read_csv('./features/train_text_features.csv')
    df_text2 = pd.read_csv('./features/train_text_features2.csv')
    
    # 加载训练标签数据
    df_train = pd.read_csv(train_path)
    
    # 多阶段合并
    merged_df = pd.merge(df_train, df_base, on=['uid', 'mid'], how='inner')
    merged_df = pd.merge(merged_df, df_time, on=['uid', 'mid'], how='inner')
    merged_df = pd.merge(merged_df, df_text1, on=['uid', 'mid'], how='inner')
    merged_df = pd.merge(merged_df, df_text2, on=['uid', 'mid'], how='inner')

    end_time = time.time()
    logging.info(f"Data loaded and processed in {end_time - start_time:.2f} seconds")
    log_memory_usage("After loading data")
    return merged_df

def train_and_predict(merged_df):
    """训练模型并预测结果"""
    os.makedirs('./save', exist_ok=True)

    # 定义特征和标签
    targets = ['forward_count', 'comment_count', 'like_count']
    features = merged_df.columns.drop(['uid', 'mid', 'time'] + targets)
    print(targets)
    print(features)
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
                
                # 分割数据集
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_trans, test_size=0.2, random_state=42
                )

                # 定义Optuna目标函数
                def objective(trial):
                    params = {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'verbosity': -1,
                        'num_leaves': trial.suggest_int('num_leaves', 30, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
                        'max_depth': trial.suggest_int('max_depth', 4, 15),
                        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 20),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 20),
                        'n_estimators': 2000,
                    }

                    model = lgb.LGBMRegressor(**params)
                    pruning_callback = LightGBMPruningCallback(trial, 'rmse')

                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric='rmse',
                        callbacks=[
                            pruning_callback,
                            lgb.early_stopping(stopping_rounds=50, verbose=False),
                            lgb.log_evaluation(period=100)
                        ]
                    )
                    return model.best_score_['valid_0']['rmse']

                # 创建并运行Optuna study
                study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
                study.optimize(objective, n_trials=50, show_progress_bar=True)

                # 输出最佳参数
                print(f"\nBest parameters for {target}:")
                best_params = study.best_params
                for key, value in best_params.items():
                    print(f"{key}: {value}")
                print(f"Best RMSE: {study.best_value:.4f}\n")

                # 使用最佳参数训练最终模型
                final_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1,
                    **best_params,
                    'n_estimators': 2000
                }

                final_model = lgb.LGBMRegressor(**final_params)
                final_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='rmse',
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(100)
                    ]
                )

                # 保存模型
                model_filename = f'./save/model_{target}_{timestamp}.pkl'
                joblib.dump(final_model, model_filename)
                print(f"Model saved to {model_filename}")

                # 记录时间和内存
                end_time = time.time()
                print(f"\nTotal time: {end_time - start_time:.2f} seconds")
                log_memory_usage("After training")

                # 生成预测
                y_pred = final_model.predict(X)
                predictions[target] = np.expm1(y_pred).round().clip(0, 100).astype(int)

    # 构建结果DataFrame
    result_df = merged_df[['uid', 'mid'] + targets].copy()
    result_df['pred_forward_count'] = predictions['forward_count']
    result_df['pred_comment_count'] = predictions['comment_count']
    result_df['pred_like_count'] = predictions['like_count']

    result_filename = f'./save/predictions_{timestamp}.csv'
    result_df.to_csv(result_filename, index=False)
    logging.info(f"Predictions saved to {result_filename}")
    return result_df

if __name__ == '__main__':
    feature_path = './features/weibo_bert_features_train_32.csv'
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
