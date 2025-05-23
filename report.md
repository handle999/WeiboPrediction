# 1. Introduction  

In the era of social media, accurately predicting user engagement metrics (e.g., likes, comments, shares) is critical for optimizing content delivery and enhancing user experience. This project aims to develop a predictive model to estimate engagement metrics for Weibo posts 24 hours after publication. Key challenges include addressing sparse/noisy data, extracting meaningful features from text and user behavior, and ensuring robustness across diverse posting patterns.  

The workflow comprises four stages:  

1. **Descriptive Analysis**: Investigating user behavior, temporal patterns, and text distribution.  
2. **Feature Extraction**: Deriving features from user profiles (historical activity, posting frequency), temporal attributes (weekdays/weekends, time periods), and text content (content classification, BERT-based embeddings).  
3. **Pre-processing**: Building a feature dictionary, handling outliers, normalizing data via Min-Max scaling, and initializing user features.  
4. **Model Development**: Evaluating baseline models (e.g., Zero Prediction) and advanced approaches such as LightGBM, XGBoost, Random Forest, MLP, and ensemble voting methods.  

---

# 2. Descriptive Statistics  

## 2.1 Dataset Overview  

- **Scope**: 1.2 million posts from 37,251 unique users.  
- **Engagement Distribution**: Highly skewed, with most posts receiving minimal interactions and a small fraction achieving viral status.  

## 2.2 Temporal Patterns  

- **Weekly activity varies significantly**: distinct engagement behaviors are observed between workdays and weekends, reflecting differing user content consumption preferences.  

## 2.3 User Activity  

- **Long-tail distribution**: A minority of highly active users generates a disproportionate volume of content, while the majority post infrequently. This pattern aligns with typical social media dynamics, where a small user subset drives platform engagement.  

# 3. Feature Extraction

## 3.1 User Feature Extraction

| Feature           | Outcome | Explanatiion                                                    |
| ----------------- | ------- | --------------------------------------------------------------- |
| Number\_in\_train | int     | Find the number of times the user appears in the training set.  |
| Forward\_max      | int     | Find the maximum of forward.                                    |
| Comment\_max      | int     | Find the maximum of comment.                                    |
| Like\_max         | int     | Find the maximum of like.                                       |
| Forward\_min      | int     | Find the minimum of forward.                                    |
| Comment\_min      | int     | Find the minimum of comment.                                    |
| Like\_min         | int     | Find the minimum of like.                                       |
| Forward\_mean     | float   | Find the mean of forward.                                       |
| Comment\_mean     | float   | Find the mean of comment.                                       |
| Like\_mean        | float   | Find the mean of like.                                          |
| Forward\_judge    | float   | Count the number of weibo posts above than the forward average. |
| Comment\_judge    | float   | Count the number of weibo posts above than the comment average. |
| Like\_judge       | float   | Count the number of weibo posts above than the like average.    |

## 3.2 Time Feature Extraction

| Feature       | Outcome   | Explanatiion                                 |
| ------------- | --------- | -------------------------------------------- |
| Time\_weekday | 1,2,...,7 | Determine the day of the week.               |
| Time\_weekend | 0,1       | Determine if the weibo post date is weekend. |
| Time\_hour    | 1,...,24  | Determine when the weibo post on the day.    |
| Period        | 1,2,3,4   | Judging Posting Period.                      |

## 3.3 Handcrafted Text Feature Extraction

| Feature         | Outcome      | Explanatiion                                                                                                                 |
| --------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Length\_all     | int          | Weibo original length.                                                                                                       |
| Length\_Chinese | int          | Length of Chinese characters in the weibo.                                                                                   |
| English         | binary (0,1) | Whether it is English content, more than half of the words are English letters, then the Weibo is English.                   |
| Non\_ch         | binary (0,1) | Whether it is non-Chinese content.                                                                                           |
| Sharing         | binary (0,1) | Whether the content is sharing content.                                                                                      |
| Auto            | binary (0,1) | Whether the text content is auto-posted in response (the text contains '我…了' and '@' or a link to the web page).             |
| Interaction     | binary (0,1) | Whether the Weibo text is interactive content (whether it contains '//', but the '//' in the web link should be considered). |
| Book            | binary (0,1) | Does the text contain the title number '《'.                                                                                  |
| Mention         | binary (0,1) | Does the text contain @.                                                                                                     |
| Vote            | binary (0,1) | Does the text contain vote.                                                                                                  |
| Lottery         | binary (0,1) | Does the text contain lottery.                                                                                               |
| Emoji           | binary (0,1) | Does the text contain emoji.                                                                                                 |
| Video           | binary (0,1) | Does the text contain video.                                                                                                 |
| Http     | binary (0,1) | Does the text contain link.                                                                                      |
| Stock    | binary (0,1) | Whether the content is a stock tweet.                                                                            |
| App      | binary (0,1) | Is there a third-party platform interactive message in the text (“我在#xxx”).                                      |
| Title    | binary (0,1) | Does the text contain 【】 or title (most likely news).                                                            |
| Ad       | binary (0,1) | Does the text contain advertise.                                                                                 |
| Keywords | binary (0,1) | Jieba word segmentation; Find high-frequency hot words, and see if each Weibo contains high-frequency hot words. |

## 3.4 LLM-based Text Feature Extraction

We use the BERT ([bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)) to extract content features with an initial dimension of 768. To make the features more tractable, we apply PCA to reduce the dimension to 32. This produces bert_f, which is a key component of our content feature set.

At the same time, we try to use QWen3 ([Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)), the results will be discussed in **Results** subsection.

Need to install `transformers`. Note that *bert* and *Qwen* need different versions of `transformers`, which can be get in there huggingface pages. And this can be **really complex**!!! So it is adviced to use bert only.

# 4. WeiBo Prediction Experiment

In this part, we conduct some experiments based on the extracted features and prediction models.

We first report the experimental designs, followed by the results and conclusion.

## 4.1 Experiment Design

Basic idea: two-step experiment
1. Feature importance: different feature combination.
1. Model importance: different model.

### 4.1.1 Evaluation metrics

Follow the [official guidelines](https://tianchi.aliyun.com/competition/entrance/231574/information), we use these metrics:
- Deviation_forward = |Forward_pred - Forward_actual| / (Forward_actual + 5)
- Deviation_comment = |Comment_pred - Comment_actual| / (Comment_actual + 3)
- Deviation_like = |Like_pred - Like_actual| / (Like_actual + 3)
- Precision_i = 1 - 0.5 × Deviation_forward - 0.25 × Deviation_comment - 0.25 × Deviation_like

All can be seen in `./eval.ipynb`

### 4.1.2 Feature list

Totally 68 dimentions.

- user(13)
    - ['number_in_train','forward_max','comment_max','like_max','forward_min','comment_min','like_min','forward_mean','comment_mean','like_mean','forward_judge','comment_judge','like_judge']
- time(4)
    - ['time_weekday','time_weekend','time_hour','panduan']
- content(19)
    - ['length_all','length_chinese','english','non_ch','sharing','auto','interaction','book','mention','vote','lottery','emoji','video','http','stock','app','title','ad','keywords']
- bert(32)
    - implementation in `./increasePCA.ipynb` for 768->32

### 4.1.3 Feature importance

1. user+time+content+bert
2. user+time+content(ablation: bert)
3. user+content+bert(ablation: time)
4. user+time+bert(ablation: text(hand))
5. time+content+bert(ablation: user)

### 4.1.4 Model importance

After feature selection, we use the best combination to test which model performs better.

Considering the huge amount of data, we do not use deep neural networks because their parameters are massive.

Instead, we choose some classic machine learning methods. We use `Optuna` library for automatic parameter tuning.

Meanwhile, we attempted to use the simplest deep network MLP. Obviously, despite its shallowest structure, the time complexity is equally unacceptable.

1. LightGBM
2. XGBoost
3. RandomForest
4. MLP
5. Multi-model vote

## 4.2 Experimental Results

The results of our experiments are shown below.

| model | features | feature dim | train precision | test precision | type |
|----------|----------|----------|-----------------|-------------|---------|
| baseline |  -  |  -  |  -  | 0.2326 | zero |
| LightGBM | bert-base-chinese | 32 | 0.32468085288174403 | 0.2450 | LLM-only |
| LightGBM | qwen3-0.6b | 32 | 0.2678484430337026 | 0.2369 | LLM-only |
| LightGBM | time+content+bert(non-nom) | 55(23+32) | 0.32194705161122195 | 0.2512 | simple attempt |
| LightGBM | user+time+content+bert | 68(36+32) | 0.40538762885717095 | 0.3072 | all |
| LightGBM | user+time+content | 36 | 0.35210145307690954 | 0.2990 | abla-LLM |
| LightGBM | user+content+bert | 64(32+32) | 0.40150651281415944 | 0.3060 | alba-time |
| LightGBM | user+time+bert | 49(17+32) | 0.395526087569475 | 0.3070 | abla-text(hand) |
| LightGBM | time+content+bert | 55(23+32) | 0.3254681151928639 | 0.2502 | abla-user |
| XGBoost  | user+time+content+bert | 68(36+32) | 0.4555724825591579 | 0.3072 |  |
| MLP      | user+time+content+bert | 68(36+32) | 0.3331129500121755 | 0.2896 |  |
| Random Forest | user+time+content+bert | 68(36+32) | 0.3694254454730708 | 0.3005 |  |
| vote | user+time+content+bert | 68(36+32) | - | 0.3051 |  |

# 5. Conclusion

## 5.1 Key Results

- **Online Performance**: Improved prediction precision from **23.26% to 30.72%**, ranking **51st**, achieving a **top 5% rank** in the competition.
- **LLM matters**: 
    - BERT-based text features outperformed handcrafted text features. 
    - For LLM, bert-base-chinese performs better then Qwen3-0.6B.
- **Ablation Studies**:
    - **Feature Impact**: The importance of feature catogery, the rank is: *"user"* > *"bert"* > *"time"* > *"text-hand"*.
    - **Model Robustness**: Ensured consistent performance across diverse user groups.
    - **Model Selection**: LightGBM and XGBoost reach the best performance. *It seems the precision of XGB is higher to the fifth decimal place*.
    - **Mechanism Matters**: Simple voting machenism cannot improve precision. Models with poor performance will lower the performance.


## 5.2 Methodology

#### 5.2.1 **Feature Engineering**

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

#### 5.2.2 **Model Architecture**

- Combined features: `time_f + text_f + user_f + bert_f`.
- Evaluation metric: Weighted average precision score.
