Microsoft Windows [版本 10.0.26100.3775]
(c) Microsoft Corporation。保留所有权利。

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/increasePCA.py
📌 第一步：拟合 IncrementalPCA
123it [11:55,  5.81s/it]
📌 第二步：降维并写入
123it [10:53,  5.32s/it]
✅ 降维完成，保存到：weibo_bert_features_train_32.csv

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/increasePCA.py
📌 第一步：拟合 IncrementalPCA
18it [01:40,  5.59s/it]
📌 第二步：降维并写入
18it [01:34,  5.27s/it]
✅ 降维完成，保存到：weibo_bert_features_predict_32.csv

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/increasePCA.py
📌 第一步：拟合 IncrementalPCA（考虑两个输入文件）
Fitting weibo_bert_features_train.csv: 123it [11:03,  5.40s/it]
Fitting weibo_bert_features_predict.csv: 18it [01:32,  5.14s/it]
📌 第二步：对每个文件进行降维并保存
Transforming weibo_bert_features_train.csv: 123it [10:34,  5.16s/it]
✅ 已完成降维并保存：weibo_bert_features_train_32.csv
Transforming weibo_bert_features_predict.csv: 18it [01:33,  5.18s/it]
✅ 已完成降维并保存：weibo_bert_features_predict_32.csv


# bert-base-chinese

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/test.py
[I 2025-05-06 17:08:59,240] A new study created in memory with name: no-name-6ef620e8-e71f-46f2-bf38-3c79d4be9271
Best trial: 38. Best value: 0.674998: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [14:50<00:00, 17.80s/it] 
[I 2025-05-06 17:25:08,402] A new study created in memory with name: no-name-d05fb42f-a762-4167-8d70-c1dc4e50a602
Best trial: 15. Best value: 0.591857: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [08:00<00:00,  9.61s/it] 
[I 2025-05-06 17:34:14,068] A new study created in memory with name: no-name-6fe91056-5676-4cd0-8167-0a6b74d81fbb
Best trial: 14. Best value: 0.627615: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [15:05<00:00, 18.11s/it] 
Training completed successfully. Results saved to ./save/

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/LightGBM_pred.py
2025-05-06 18:06:15,249 - INFO - Loading data...
2025-05-06 18:06:20,170 - INFO - Loading models...
2025-05-06 18:06:20,171 - INFO - Loading model: ./save\model_forward_count_20250506_170858.pkl
2025-05-06 18:06:21,748 - INFO - Loading model: ./save\model_comment_count_20250506_170858.pkl
2025-05-06 18:06:21,906 - INFO - Loading model: ./save\model_like_count_20250506_170858.pkl
2025-05-06 18:06:22,109 - INFO - Making predictions...
2025-05-06 18:06:32,736 - INFO - Predictions saved to ./predict_results\predictions_20250506_180632.csv

eval time: 71.1s
weighted_precision_score: 0.32468085288174403
mean_precision: 0.9044887723705354

# qwen3 0.6b

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/LightGBM.py
[I 2025-05-06 18:33:06,662] A new study created in memory with name: no-name-ba2e179d-fa66-410f-b09a-015d31d739fe
Best trial: 27. Best value: 0.791525: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [07:16<00:00,  8.73s/it]
[I 2025-05-06 18:40:32,548] A new study created in memory with name: no-name-148847e0-2524-4e9f-a35e-7248784caec0
Best trial: 25. Best value: 0.643904: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [06:11<00:00,  7.43s/it]
[I 2025-05-06 18:46:54,048] A new study created in memory with name: no-name-7fc5f1dd-cf8f-408a-a438-c2b4851467f3
Best trial: 0. Best value: 0.714487: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [04:11<00:00,  5.03s/it]
Training completed successfully. Results saved to ./save/

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/LightGBM_pred.py
2025-05-06 20:14:14,572 - INFO - Loading data...
2025-05-06 20:14:19,094 - INFO - Loading models...
2025-05-06 20:14:19,095 - INFO - Loading model: ./save\model_forward_count_20250506_183306.pkl
2025-05-06 20:14:20,466 - INFO - Loading model: ./save\model_comment_count_20250506_183306.pkl
2025-05-06 20:14:20,507 - INFO - Loading model: ./save\model_like_count_20250506_183306.pkl
2025-05-06 20:14:20,583 - INFO - Making predictions...
2025-05-06 20:14:23,293 - INFO - Predictions saved to ./save\predictions_20250506_201422.csv

eval time: 74.7s
weighted_precision_score: 0.2678484430337026
mean_precision: 0.8781056367325834

# time+content+bert, w/o user, no scalar

eval time: 87.2s
weighted_precision_score: 0.32194705161122195
mean_precision: 0.9030816356819562

# user+time+content non-scalar

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/LightGBM_basic.py  
['forward_count', 'comment_count', 'like_count']
Index(['length_all', 'length_chinese', 'english', 'non_ch', 'sharing', 'auto',
       'interaction', 'book', 'mention', 'vote', 'lottery', 'emoji', 'video',
       'http', 'stock', 'app', 'title', 'ad', 'keywords', 'time_weekday',
       'time_weekend', 'time_hour', 'panduan', 'number_in_train',
       'forward_max', 'comment_max', 'like_max', 'forward_min', 'comment_min',
       'like_min', 'forward_mean', 'comment_mean', 'like_mean',
       'forward_judge', 'comment_judge', 'like_judge'],
      dtype='object')
[I 2025-05-07 17:32:58,933] A new study created in memory with name: no-name-c0834078-8b69-431e-87f1-939e29674fca
Best trial: 4. Best value: 0.491109: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [06:45<00:00,  8.10s/it]
[I 2025-05-07 17:40:16,083] A new study created in memory with name: no-name-c9c7daf8-a114-45b3-b2e0-55faa2dd723d
Best trial: 12. Best value: 0.472877: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [09:30<00:00, 11.41s/it]
[I 2025-05-07 17:50:59,132] A new study created in memory with name: no-name-790fdae8-3318-495c-a4ff-1975e7c19fe8
Best trial: 13. Best value: 0.436873: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [09:43<00:00, 11.67s/it]
Training completed successfully. Results saved to ./save/

eval time: 76.8s
weighted_precision_score: 0.35210145307690954
mean_precision: 0.9088706368899243

# user+time+content+bert(all)

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/LightGBM_basic+bert.py
['forward_count', 'comment_count', 'like_count']
Index(['length_all', 'length_chinese', 'english', 'non_ch', 'sharing', 'auto',
       'interaction', 'book', 'mention', 'vote', 'lottery', 'emoji', 'video',
       'http', 'stock', 'app', 'title', 'ad', 'keywords', 'time_weekday',
       'time_weekend', 'time_hour', 'panduan', 'number_in_train',
       'forward_max', 'comment_max', 'like_max', 'forward_min', 'comment_min',
       'like_min', 'forward_mean', 'comment_mean', 'like_mean',
       'forward_judge', 'comment_judge', 'like_judge', 'feature_0',
       'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
       'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10',
       'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15',
       'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_20',
       'feature_21', 'feature_22', 'feature_23', 'feature_24', 'feature_25',
       'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30',
       'feature_31'],
      dtype='object')
[I 2025-05-07 19:08:32,532] A new study created in memory with name: no-name-280f7487-29ad-4b66-ad75-a36e37044c49
Best trial: 21. Best value: 0.473578: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [27:31<00:00, 33.03s/it]
[I 2025-05-07 19:37:15,885] A new study created in memory with name: no-name-de84fe45-6307-4314-ae72-1e42ae650624
Best trial: 7. Best value: 0.461471: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [11:09<00:00, 13.38s/it]
[I 2025-05-07 19:49:48,520] A new study created in memory with name: no-name-55b9b05a-8888-44e6-b640-3acc35398b22
Best trial: 22. Best value: 0.423862: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [32:09<00:00, 38.60s/it]
Training completed successfully. Results saved to ./save/

eval time: 76.8s
weighted_precision_score: 0.40538762885717095
mean_precision: 0.9216990358486769

# user+content+bert(abla-time)

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/LightGBM_basic+bert-time.py
['forward_count', 'comment_count', 'like_count']
Index(['length_all', 'length_chinese', 'english', 'non_ch', 'sharing', 'auto',
       'interaction', 'book', 'mention', 'vote', 'lottery', 'emoji', 'video',
       'http', 'stock', 'app', 'title', 'ad', 'keywords', 'number_in_train',
       'forward_max', 'comment_max', 'like_max', 'forward_min', 'comment_min',
       'like_min', 'forward_mean', 'comment_mean', 'like_mean',
       'forward_judge', 'comment_judge', 'like_judge', 'feature_0',
       'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
       'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10',
       'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15',
       'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_20',
       'feature_21', 'feature_22', 'feature_23', 'feature_24', 'feature_25',
       'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30',
       'feature_31'],
      dtype='object')
[I 2025-05-07 21:09:15,131] A new study created in memory with name: no-name-3aed92f9-ac06-47f6-9ac5-e61e617301a5
Best trial: 4. Best value: 0.474981: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [17:01<00:00, 20.43s/it]
Best trial: 0. Best value: 0.460894: 100%|█████████████████████████████████████████████████████████████| 50/50 [12:09<00:00, 14.59s/it]
[I 2025-05-07 21:43:12,718] A new study created in memory with name: no-name-dfcddd6f-8ad9-42bf-9fc4-ca79ad24ccd2
Best trial: 14. Best value: 0.425342: 100%|█████████████████████████| 50/50 [35:29<00:00, 42.59s/it]
Training completed successfully. Results saved to ./save/

88.7s
0.40150651281415944 0.9211646769237626

# time+content+bert(abla-user)

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/LightGBM_basic+bert-time.py
['forward_count', 'comment_count', 'like_count']
Index(['length_all', 'length_chinese', 'english', 'non_ch', 'sharing', 'auto',
       'interaction', 'book', 'mention', 'vote', 'lottery', 'emoji', 'video',
       'http', 'stock', 'app', 'title', 'ad', 'keywords', 'time_weekday',
       'time_weekend', 'time_hour', 'panduan', 'feature_0', 'feature_1',
       'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6',
       'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11',
 'feature_21',
       'feature_22', 'feature_23', 'feature_24', 'feature_25', 'feature_26',
       'feature_27', 'feature_28', 'feature_29', 'feature_30', 'feature_31'],
      dtype='object')
[I 2025-05-07 23:14:58,779] A new study created in memory with name: no-name-60610ed3-b219-48ec-8c90-2ee132dc9f3f
Best trial: 12. Best value: 0.655763: 100%|█████████████████████████████| 50/50 [59:21<00:00, 71.23s/it]
[I 2025-05-08 00:17:26,948] A new study created in memory with name: no-name-cbfe5f50-81ef-4da3-b55c-da854fd2b243
Best trial: 12. Best value: 0.585797: 100%|█████████████████████████████| 50/50 [22:24<00:00, 26.89s/it]
[I 2025-05-08 00:42:01,324] A new study created in memory with name: no-name-e6e12c0c-c13c-4c45-ae4a-377442d8f42f
Best trial: 31. Best value: 0.614324: 100%|█████████████████████████████| 50/50 [38:55<00:00, 46.71s/it]
Training completed successfully. Results saved to ./save/

86.0s
0.3254681151928639 0.9041434683296935

# user+time+bert(abla-content)

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/LightGBM_basic+bert-time.py
['forward_count', 'comment_count', 'like_count']
Index(['time_weekday', 'time_weekend', 'time_hour', 'panduan',
       'number_in_train', 'forward_max', 'comment_max', 'like_max',
       'forward_min', 'comment_min', 'like_min', 'forward_mean',
       'comment_mean', 'like_mean', 'forward_judge', 'comment_judge',
       'like_judge', 'feature_0', 'feature_1', 'feature_2', 'feature_3',
       'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8',
       'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13',
       'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18',
       'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23',
       'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28',
       'feature_29', 'feature_30', 'feature_31'],
      dtype='object')
[I 2025-05-08 10:13:31,586] A new study created in memory with name: no-name-674d946e-ef3d-449e-95d4-e2466d327c26
Best trial: 12. Best value: 0.477216: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [10:01<00:00, 12.03s/it]
[I 2025-05-08 10:24:27,724] A new study created in memory with name: no-name-559ed293-bf85-4821-a25f-e4a1012b64fc
Best trial: 15. Best value: 0.463639: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [10:03<00:00, 12.07s/it]
[I 2025-05-08 10:34:56,227] A new study created in memory with name: no-name-692ad63e-98a9-45f5-b3da-2ac19e7e179a
Best trial: 3. Best value: 0.425607: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [08:54<00:00, 10.68s/it]
Training completed successfully. Results saved to ./save/

78.5s
0.395526087569475 0.92000300675189

# XGBoost

(SAM) E:\School\研一下\社交网络分析（英）-冯冲郭宇航\proj>C:/Users/Highee/.conda/envs/SAM/python.exe e:/School/研一下/社交网络分析（英）-冯冲郭宇航/proj/XGB_basic+bert.py
[I 2025-05-08 15:23:13,663] A new study created in memory with name: no-name-712b6023-1f91-4002-b393-2bebec7df3a2
Best trial: 11. Best value: 0.472637: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [34:28<00:00, 41.38s/it] 
[I 2025-05-08 16:00:19,876] A new study created in memory with name: no-name-51b6f21d-cf30-4927-9e92-4d6f83232642
Best trial: 2. Best value: 0.460213: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [25:21<00:00, 30.43s/it] 
[I 2025-05-08 16:30:41,854] A new study created in memory with name: no-name-99776e81-ae8c-4604-abe3-60d1f5385b8a
Best trial: 3. Best value: 0.423795: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [1:10:48<00:00, 84.97s/it]
Training completed successfully. Results saved to ./save/

74.9s
0.4555724825591579 0.9297618394220698

# mlp

    param_list = {
        'hidden_dim': 256,
        'dropout': 0.3,
        'lr': 1e-3,
        'batch_size': 256,
        'epochs': 50
    }

    0.31515424411625237 0.9119022536244064

        param_list = {
        'hidden_dim': 512,
        'dropout': 0.3,
        'lr': 1e-3,
        'batch_size': 1024,
        'epochs': 50
    }

    0.3251283979203424 0.9112200047678414

        param_list = {
        'hidden_dim': 1024,
        'dropout': 0.3,
        'lr': 1e-4,
        'batch_size': 4096,
        'epochs': 200
    }

    0.3331129500121755 0.915836961969716

# random forest

0.3694254454730708 0.9168242881648407