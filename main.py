import os
from fpdf import FPDF

# 定义Markdown文件的目录
local_folder = 'LeetCode_Solutions'

# 初始化PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)

# 合并所有Markdown文件到一个PDF中
for md_file in os.listdir(local_folder):
    if md_file.endswith('.md'):
        with open(os.path.join(local_folder, md_file), 'r', encoding='utf-8') as file:
            for line in file:
                # 将每行文本转换为 utf-8 编码
                encoded_line = line.encode('latin1', 'replace').decode('latin1')
                pdf.multi_cell(0, 10, encoded_line)
            pdf.add_page()

# 保存PDF文件
pdf_file = "LeetCode_Solutions.pdf"
pdf.output(pdf_file)

print(f"PDF文件已保存为 {pdf_file}")