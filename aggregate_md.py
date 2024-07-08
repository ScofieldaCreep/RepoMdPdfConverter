import os

# 定义Markdown文件的目录
local_folder = 'LeetCode_Solutions'

# 获取所有Markdown文件
md_files = sorted([f for f in os.listdir(local_folder) if f.endswith('.md')])

# 计算每个部分应该包含的文件数量
num_files = len(md_files)
files_per_part = num_files // 8

# 创建并写入每个部分的文件
for i in range(8):
    start_index = i * files_per_part
    end_index = (i + 1) * files_per_part if i < 7 else num_files  # 确保最后一部分包含所有剩余文件
    part_files = md_files[start_index:end_index]

    part_filename = os.path.join(local_folder, f'part_{i + 1}.md')

    with open(part_filename, 'w', encoding='utf-8') as outfile:
        for md_file in part_files:
            with open(os.path.join(local_folder, md_file), 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())

print("所有Markdown文件已成功分成8个部分并合并。")