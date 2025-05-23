import pandas as pd
import numpy as np
import joblib
import os
import glob
import logging
from datetime import datetime
import xgboost as xgb

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_latest_model(target_name):
    """自动加载最新保存的模型"""
    model_files = glob.glob(f'./save/model_{target_name}_*.pkl')
    if not model_files:
        raise FileNotFoundError(f"No models found for {target_name}")
    # 按时间戳排序选择最新模型
    latest_model = sorted(model_files, key=os.path.getctime, reverse=True)[0]
    logging.info(f"Loading model: {latest_model}")
    return joblib.load(latest_model)

def predict_pipeline():
    """完整的预测流程"""
    # 路径配置
    predict_path = './features/weibo_predict.csv'
    feature_path = './features/predict_user_mid_feature_scalar.csv'
    output_dir = './save'
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 1. 加载数据
        logging.info("Loading data...")
        df_features = pd.read_csv(feature_path)
        
        # 2. 加载训练标签数据
        df_train = pd.read_csv(predict_path)
        # 加载基础特征
        df_bert = pd.read_csv('./features/weibo_bert_features_predict_32.csv')
        content_features = df_bert['content_feature'].str.split(' ', expand=True).astype(float)
        content_features.columns = [f'feature_{i}' for i in range(32)]
        df_base = pd.concat([df_bert[['uid', 'mid']], content_features], axis=1)
        
        # 3. 合并
        merged_df = pd.merge(df_train, df_features, on=['uid', 'mid'], how='inner')
        merged_df = pd.merge(merged_df, df_base, on=['uid', 'mid'], how='inner')
        X_pred = merged_df

        # 4. 加载模型
        logging.info("Loading models...")
        forward_model = load_latest_model('forward_count')
        comment_model = load_latest_model('comment_count')
        like_model = load_latest_model('like_count')

        # 5. 生成预测
        logging.info("Making predictions...")
        features = X_pred.columns.drop(['uid', 'mid', 'time'])
        print("features: ", features)
        
        # 将特征转换为 DMatrix
        dtest = xgb.DMatrix(X_pred[features])
        # 打印 DMatrix 的形状
        print("Shape of DMatrix (rows, columns):", (dtest.num_row(), dtest.num_col()))
        logging.info(f"Shape of DMatrix (rows, columns): ({dtest.num_row()}, {dtest.num_col()})")
        
        pred_forward = np.expm1(forward_model.predict(dtest))
        pred_comment = np.expm1(comment_model.predict(dtest))
        pred_like = np.expm1(like_model.predict(dtest))

        # 6. 后处理
        merged_df['pred_forward_count'] = pred_forward.round().clip(0, 100).astype(int)
        merged_df['pred_comment_count'] = pred_comment.round().clip(0, 100).astype(int)
        merged_df['pred_like_count'] = pred_like.round().clip(0, 100).astype(int)

        # 7. 保存结果
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
