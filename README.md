# Weibo Prediction Challenge

[Sina Weibo Interaction-prediction-Challenge the Baseline](https://tianchi.aliyun.com/competition/entrance/231574/information)

Improved prediction precision from **23.26%** to **30.72%**, ranking **51st**, achieving a **top 5% rank** in the competition.

Detailed method and design are available at `report.md`.

# Methodology

## Feature Engineering

1. **User Features**:
   - Historical activity patterns (posting frequency, average engagement).
   - Mean user engagement metrics for new users.
2. **Time Features**:
   - Temporal trends (hourly, daily, weekly).
3. **Text Features**:
   - **Handcrafted Features**: Basic text statistics (length, keywords).
   - **LLM-Based Features**: BERT embeddings (768 dimensions) reduced to 32 dimensions via PCA.
4. **Preprocessing**:
   - Outlier handling and Min-Max normalization.

## Model Architecture

- Combined features: `time_f + text_f + user_f + bert_f`.
- Evaluation metric: Weighted average precision score.

## Model Selection

- **XGBoost(Best)**
- LightGBM
- Random Forest
- MLP

# Run

1. **feature extraction**
   1. hand-crafted features
      1. handfeature.ipynb
   1. LLM-based features
      1. LLM-feature-extraction/bert-base-chinese-train.ipynb
      1. LLM-feature-extraction/bert-base-chinese-pred.ipynb
   1. feature dimensionality reduction (LLM-based)
      1. increasePCA.ipynb
   1. feature normalization
      1. preprocess.py
1. **model train**
   1. (*alternative*) LightGBM_all.py
1. **model predict**
   1. (*alternative*) LightGBM_pred_all.py
1. **model test (trainset)**
   1. [*optional*] eval.ipynb
1. **result format change**
   1. rst_csv2txt.py
1. **multi-model voting**
   1. [*optional*] vote.py
