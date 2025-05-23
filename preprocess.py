import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


class TrainDatasetPreprocess:
    def __init__(self):
        self.df_text_feature_1 = pd.read_csv('features/train_text_features.csv')
        self.df_text_feature_2 = pd.read_csv('features/train_text_features2.csv')
        self.df_time_feature = pd.read_csv('features/train_time_features.csv')

    def merge_mid_feature(self):
        df_mid_feature = self.df_text_feature_1.merge(self.df_text_feature_2, on=['mid', 'uid'], how='inner') \
            .merge(self.df_time_feature, on=['mid', 'uid'], how='inner')
        df_mid_feature.to_csv('features/train_mid_feature.csv', index=False)

    @staticmethod
    def mid_feature_scalar():
        df_mid_feature = pd.read_csv('features/train_mid_feature.csv')

        # 分离 uid, mid 列
        uid_mid_column = df_mid_feature[['uid', 'mid']]
        feature_columns = df_mid_feature.drop(columns=['uid', 'mid'])

        # 归一化
        scaler = MinMaxScaler()
        normalized_features = pd.DataFrame(scaler.fit_transform(feature_columns),
                                           columns=feature_columns.columns)

        # 拼接归一化结果和 mid
        normalized_df = pd.concat([uid_mid_column, normalized_features], axis=1)

        # 确保目录存在
        os.makedirs('features/', exist_ok=True)

        # 保存
        normalized_df.to_csv('features/train_mid_feature_scalar.csv',
                             index=False)

    @staticmethod
    def user_feature_scalar():
        df_user_feature = pd.read_csv('features/train_user_features.csv')

        # 分离 uid, mid 列
        uid_column = df_user_feature[['uid']]
        feature_columns = df_user_feature.drop(columns=['uid'])

        # 归一化
        scaler = MinMaxScaler()
        normalized_features = pd.DataFrame(scaler.fit_transform(feature_columns),
                                           columns=feature_columns.columns)

        # 拼接归一化结果和 uid, mid
        normalized_df = pd.concat([uid_column, normalized_features], axis=1)

        # 确保目录存在
        os.makedirs('features/', exist_ok=True)

        # 保存
        normalized_df.to_csv('features/train_user_feature_scalar.csv',
                             index=False)

    @staticmethod
    def user_mid_feature_scalar():
        df_user_feature = pd.read_csv('features/train_user_feature_scalar.csv')
        df_mid_feature = pd.read_csv('features/train_mid_feature_scalar.csv')

        df_user_mid_feature = df_mid_feature.merge(df_user_feature, on=['uid'], how='left')
        # df_user_mid_feature.fillna(value={}, inplace=True)
        user_feature_columns = df_user_feature.columns.drop('uid').tolist()
        for column in user_feature_columns:
            mean_val = df_user_feature[column].mean()
            df_user_mid_feature[column] = df_user_mid_feature[column].fillna(mean_val)
        df_user_mid_feature.to_csv('features/train_user_mid_feature_scalar.csv',
                             index=False)

class PredictDatasetPreprocess:
    def __init__(self):
        self.df_text_feature_1 = pd.read_csv('features/predict_text_features.csv')
        self.df_text_feature_2 = pd.read_csv('features/predict_text_features2.csv')
        self.df_time_feature = pd.read_csv('features/predict_time_features.csv')

    def merge_mid_feature(self):
        df_mid_feature = self.df_text_feature_1.merge(self.df_text_feature_2, on=['mid', 'uid'], how='inner') \
            .merge(self.df_time_feature, on=['mid', 'uid'], how='inner')
        df_mid_feature.to_csv('features/predict_mid_feature.csv', index=False)

    @staticmethod
    def mid_feature_scalar():
        df_mid_feature = pd.read_csv('features/predict_mid_feature.csv')

        # 分离 uid, mid 列
        uid_mid_column = df_mid_feature[['uid', 'mid']]
        feature_columns = df_mid_feature.drop(columns=['uid', 'mid'])

        # 归一化
        scaler = MinMaxScaler()
        normalized_features = pd.DataFrame(scaler.fit_transform(feature_columns),
                                           columns=feature_columns.columns)

        # 拼接归一化结果和 mid
        normalized_df = pd.concat([uid_mid_column, normalized_features], axis=1)

        # 确保目录存在
        os.makedirs('features/', exist_ok=True)

        # 保存
        normalized_df.to_csv('features/predict_mid_feature_scalar.csv',
                             index=False)

    @staticmethod
    def user_mid_feature_scalar():
        df_user_feature = pd.read_csv('features/train_user_feature_scalar.csv')
        df_mid_feature = pd.read_csv('features/predict_mid_feature_scalar.csv')

        df_user_mid_feature = df_mid_feature.merge(df_user_feature, on=['uid'], how='left')
        # df_user_mid_feature.fillna(value={}, inplace=True)
        user_feature_columns = df_user_feature.columns.drop('uid').tolist()
        for column in user_feature_columns:
            mean_val = df_user_feature[column].mean()
            df_user_mid_feature[column] = df_user_mid_feature[column].fillna(mean_val)
        df_user_mid_feature.to_csv('features/predict_user_mid_feature_scalar.csv',
                                   index=False)


if __name__ == '__main__':
    trainDatasetPreprocess = TrainDatasetPreprocess()
    # trainDatasetPreprocess.merge_mid_feature()
    # trainDatasetPreprocess.mid_feature_scalar()
    # trainDatasetPreprocess.user_feature_scalar()
    # trainDatasetPreprocess.user_mid_feature_scalar()

    predictDatasetPreprocess = PredictDatasetPreprocess()
    # predictDatasetPreprocess.merge_mid_feature()
    # predictDatasetPreprocess.mid_feature_scalar()
    predictDatasetPreprocess.user_mid_feature_scalar()
    