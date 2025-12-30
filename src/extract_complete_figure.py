"""
从PDF中提取完整的Figure页面（渲染整页为图片）
"""
import fitz  # PyMuPDF
import os

# 配置
PDF_PATH = r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\data\Al2219\metals-11-00077.pdf"
OUTPUT_PATH = r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业\images\al2219_figure3_complete.png"
PAGE_NUM = 3  # Figure 3在第4页（索引从0开始，所以是3）

def extract_page_as_image(pdf_path, page_num, output_path, zoom=2.0):
    """
    将PDF页面渲染为高分辨率图片

    Args:
        pdf_path: PDF文件路径
        page_num: 页码（从0开始）
        output_path: 输出图片路径
        zoom: 缩放因子（2.0 = 144 DPI, 3.0 = 216 DPI）
    """
    print(f"正在打开PDF: {pdf_path}")
    doc = fitz.open(pdf_path)

    if page_num >= len(doc):
        print(f"错误：页码 {page_num} 超出范围（总页数：{len(doc)}）")
        return False

    print(f"正在渲染第 {page_num + 1} 页...")
    page = doc[page_num]

    # 设置缩放矩阵（提高分辨率）
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    print(f"图片尺寸: {pix.width}x{pix.height}")
    print(f"正在保存到: {output_path}")

    # 保存为PNG
    pix.save(output_path)

    doc.close()
    print("✓ 提取完成！")
    return True

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 提取Figure 3（第4页）
    success = extract_page_as_image(PDF_PATH, PAGE_NUM, OUTPUT_PATH, zoom=3.0)

    if success:
        print(f"\n完整的Figure 3已保存到:")
        print(OUTPUT_PATH)
