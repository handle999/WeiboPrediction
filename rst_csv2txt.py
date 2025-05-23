import csv

input_file = './save/predictions_20250508_225853.csv'
output_file = './save/output.txt'
# input_file = './LightGBM_bert32_wouser/predictions_20250506_215147.csv'
# output_file = './LightGBM_bert32_wouser/output.txt'

with open(input_file, 'r', encoding='utf-8') as csv_file, open(output_file, 'w', encoding='utf-8') as txt_file:
    reader = csv.reader(csv_file)
    next(reader)  # 跳过表头
    for row in reader:
        uid, mid, forward, comment, like = row
        line = f"{uid}\t{mid}\t{forward},{comment},{like}\n"
        txt_file.write(line)
