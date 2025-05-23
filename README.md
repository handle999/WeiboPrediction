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
