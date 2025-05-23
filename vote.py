import os
from pathlib import Path

# 指定包含output.txt的文件夹路径列表
file_dirs = [
    "LightGBM_all4",
    # "LightGBM_abla_user",
    # "LightGBM_abla_time",
    # "LightGBM_abla_texthand",
    # "LightGBM_abla_llm",
    "MLP",
    "XGBoost_all4",
    "RF"
]

# 转为Path对象并拼接output.txt路径
output_files = [Path(d) / "output.txt" for d in file_dirs]

# 用于存储累计结果
lines_sum = []
num_files = len(output_files)

# 遍历并累计
for idx, file in enumerate(output_files):
    with open(file, 'r') as f:
        lines = f.readlines()
        if idx == 0:
            # 初始化结构
            lines_sum = [[line.strip().split('\t')[0], line.strip().split('\t')[1], [0, 0, 0]] for line in lines]

        for i, line in enumerate(lines):
            parts = line.strip().split('\t')
            values = list(map(int, parts[2].split(',')))
            for j in range(3):
                lines_sum[i][2][j] += values[j]

# 计算平均并写入
with open("output.txt", 'w') as f:
    for id1, id2, sum_vals in lines_sum:
        avg_vals = [round(v / num_files) for v in sum_vals]
        avg_str = ','.join(map(str, avg_vals))
        f.write(f"{id1}\t{id2}\t{avg_str}\n")
