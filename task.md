### 赛题

新浪微博互动预测-挑战Baseline

### 简介

对于一条原创博文而言,转发、评论、赞等互动行为能够体现出用户对于博文内容的兴趣程度，也是对博文进行分发控制的重要参考指标。本届赛题的任务就是根据抽样用户的原创博文在发表一天后的转发、评论、赞总数，建立博文的互动模型，并预测用户后续博文在发表一天后的互动情况。

### 链接

[新浪微博互动预测-挑战Baseline_学习赛_天池大赛-阿里云天池的赛制](https://tianchi.aliyun.com/competition/entrance/231574)

### 数据介绍

- 训练数据（weibo_train_data(new)）2015-02-01至2015-07-31
  - 博文的全部信息都映射为一行数据。其中对用户做了一定抽样，获取了抽样用户半年的原创博文，对用户标记和博文标记做了加密，发博时间精确到天级别。 
- 预测数据（weibo_predict_data(new)）2015-08-01至2015-08-31
- 数据条目
  - uid，用户标记，字段加密
  - mid，博文标记，字段加密
  - time，发博时间，yyyy-mm-dd hh:mm:ss
  - forward_count，一周内转发量，int
  - comment_count，一周内评论数，int
  - like_count，一周内赞数，int
  - content，博文内容，string(中文)

选手提交结果文件的转、评、赞值必须为整数不接受浮点数！注意：提交格式(.txt)：uid、mid、forward_count字段以tab键分隔，forward_count、comment_count、like_count字段间以逗号分隔

### 一些想法

- 参照一些已有解题思路
  - [天池新人赛之新浪微博互动预测_天池新人赛-新浪微博互动预测-挑战baseline-CSDN博客](https://blog.csdn.net/jingyi130705008/article/details/78257350)
  - [【手把手教你玩天池新人挑战赛】新浪微博互动预测100行代码_新浪微博互动预测python代码-CSDN博客](https://blog.csdn.net/Bryan__/article/details/50220229?ops_request_misc=%7B%22request%5Fid%22%3A%22164862598716780265458165%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fblog.%22%7D&request_id=164862598716780265458165&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-2-50220229.nonecase&utm_term=100&spm=1018.2226.3001.4450)
  - [实战微博互动预测之一_问题分析_51CTO博客_新浪微博互动预测](https://blog.51cto.com/u_15794627/5682638)
- 分析思路
  - 想要准确预测，需要对任务进行明确
    - 任务是整数预测
    - 考虑用户特征、文本特征
    - 对数据进行分析（特征抽取的特征需要验证是否真的相关）
  - 用户特征
    - 之前发文
    - 之前关注（3指标）
    - 发文时间（是否会对关注度产生影响）
  - 文本特征
    - 主题
    - 情感分析
    - 内容
      - 是否包含链接？
      - 是否转发？
      - 是否对应热点话题？
    - 考虑特征向量中心性：越多被关注的越容易被关注（微博推荐算法）
    