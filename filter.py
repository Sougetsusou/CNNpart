# 定义输入和输出文件路径
input_file = "./DataSet/data.txt"
output_file = "./DataSet/filtered_green.txt"

# 定义有效车牌类型
valid_types = {"新能源小型车"}

# 打开输入文件并逐行读取
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        line = line.strip()  # 去除换行符和多余空白
        if not line:
            continue  # 跳过空行

        # 分割每行内容，假设行格式为 "图片路径 标签 车牌类型"
        parts = line.split(' ')
        if len(parts) != 3:
            print(f"无效行格式: {line}")  # 打印无效行的提示
            continue

        img_path, label, plate_type = parts

        # 检查车牌类型是否在有效类型集合中
        if plate_type in valid_types:
            # 将符合要求的行写入输出文件
            outfile.write(line + '\n')

print(f"筛选完成！结果已保存到 {output_file}")
