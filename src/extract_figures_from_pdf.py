"""
从论文PDF中自动提取应力应变曲线图表（统一版本）

支持多种材料的PDF图表提取，配置化管理提取任务

依赖库：
    pip install PyMuPDF Pillow

使用方法：
    python extract_figures_from_pdf_unified.py
"""

import fitz  # PyMuPDF
import os
from PIL import Image
import io
import shutil

# 定义项目路径
BASE_DIR = r"D:\AAA_postgraduate\FirstSemester\教材\机器学习工程应用\大作业"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "images")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 配置区：定义所有PDF提取任务 ==========

# Al7075材料提取任务
AL7075_TASKS = [
    {
        "name": "Al7075 - Frontiers 2025 Figure 3",
        "pdf_path": os.path.join(DATA_DIR, "Al7075", "fmats-12-1671753.pdf"),
        "output_file": os.path.join(OUTPUT_DIR, "al7075_frontiers_fig3.png"),
        "keyword": "Figure 3",
        "page_range": (3, 8),
        "description": "应力应变曲线"
    },
    {
        "name": "Al7075 - MDPI 2023 Figure 8",
        "pdf_path": os.path.join(DATA_DIR, "Al7075", "materials-16-07432.pdf"),
        "output_file": os.path.join(OUTPUT_DIR, "al7075_mdpi_fig8.png"),
        "keyword": "Figure 8",
        "page_range": (6, 12),
        "description": "热变形工况图"
    },
    {
        "name": "Al7075 - Springer 2020 Figure 7",
        "pdf_path": os.path.join(DATA_DIR, "Al7075", "s10033-020-00494-8.pdf"),
        "output_file": os.path.join(OUTPUT_DIR, "al7075t6_springer_fig7.png"),
        "keyword": "Figure 7",
        "page_range": (5, 10),
        "description": "True stress-strain曲线"
    }
]

# Al2024材料提取任务
AL2024_TASKS = [
    {
        "name": "Al2024 Figure 1",
        "pdf_path": os.path.join(DATA_DIR, "Al2024", "1-s2.0-S2238785423013509-main.pdf"),
        "keyword": "Fig. 1",
        "page_range": (1, 10),
        "description": "Al2024微观结构或应力应变曲线",
        "output_prefix": "al2024_fig1"
    },
    {
        "name": "Al2024 Figure 2",
        "pdf_path": os.path.join(DATA_DIR, "Al2024", "1-s2.0-S2238785423013509-main.pdf"),
        "keyword": "Fig. 2",
        "page_range": (1, 10),
        "description": "Al2024应力应变曲线",
        "output_prefix": "al2024_fig2"
    },
    {
        "name": "Al2024 Figure 3",
        "pdf_path": os.path.join(DATA_DIR, "Al2024", "1-s2.0-S2238785423013509-main.pdf"),
        "keyword": "Fig. 3",
        "page_range": (1, 10),
        "description": "Al2024应力应变曲线",
        "output_prefix": "al2024_fig3"
    },
    {
        "name": "Al2024 Figure 4",
        "pdf_path": os.path.join(DATA_DIR, "Al2024", "1-s2.0-S2238785423013509-main.pdf"),
        "keyword": "Fig. 4",
        "page_range": (1, 10),
        "description": "Al2024应力应变曲线",
        "output_prefix": "al2024_fig4"
    },
    {
        "name": "Al2024 Figure 5",
        "pdf_path": os.path.join(DATA_DIR, "Al2024", "1-s2.0-S2238785423013509-main.pdf"),
        "keyword": "Fig. 5",
        "page_range": (1, 10),
        "description": "Al2024应力应变曲线",
        "output_prefix": "al2024_fig5"
    },
    {
        "name": "Al2024 Figure 6",
        "pdf_path": os.path.join(DATA_DIR, "Al2024", "1-s2.0-S2238785423013509-main.pdf"),
        "keyword": "Fig. 6",
        "page_range": (1, 10),
        "description": "Al2024应力应变曲线",
        "output_prefix": "al2024_fig6"
    },
    {
        "name": "Al2024 Figure 7",
        "pdf_path": os.path.join(DATA_DIR, "Al2024", "1-s2.0-S2238785423013509-main.pdf"),
        "keyword": "Fig. 7",
        "page_range": (1, 10),
        "description": "Al2024应力应变曲线",
        "output_prefix": "al2024_fig7"
    },
    {
        "name": "Al2024 Figure 8",
        "pdf_path": os.path.join(DATA_DIR, "Al2024", "1-s2.0-S2238785423013509-main.pdf"),
        "keyword": "Fig. 8",
        "page_range": (1, 10),
        "description": "Al2024应力应变曲线",
        "output_prefix": "al2024_fig8"
    }
]

# Al2219材料提取任务
AL2219_TASKS = [
    {
        "name": "Al2219 Figure 3",
        "pdf_path": os.path.join(DATA_DIR, "Al2219", "metals-11-00077.pdf"),
        "keyword": "Figure 3",
        "page_range": (1, 15),
        "description": "Al2219应力应变曲线",
        "output_prefix": "al2219_fig3"
    }
]

# 合并所有任务
ALL_EXTRACTION_TASKS = AL7075_TASKS + AL2024_TASKS + AL2219_TASKS

# ========== 核心功能函数 ==========

def extract_images_from_page(page, min_width=200, min_height=200):
    """
    从PDF页面中提取所有图片

    Args:
        page: fitz.Page对象
        min_width: 最小图片宽度（像素）
        min_height: 最小图片高度（像素）

    Returns:
        list: 图片列表（PIL Image对象）
    """
    images = []
    image_list = page.get_images(full=True)

    for img_index, img in enumerate(image_list):
        xref = img[0]
        try:
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # 转换为PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))

            # 过滤太小的图片（可能是装饰图标）
            if pil_image.width >= min_width and pil_image.height >= min_height:
                images.append({
                    "image": pil_image,
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "ext": image_ext
                })
                print(f"    找到图片 {img_index+1}: {pil_image.width}x{pil_image.height} ({image_ext})")
        except Exception as e:
            print(f"    提取图片 {img_index+1} 失败: {e}")
            continue

    return images


def search_figure_by_keyword(pdf_path, keyword, page_range=None):
    """
    通过关键词搜索PDF中的Figure位置

    Args:
        pdf_path: PDF文件路径
        keyword: 搜索关键词（如"Figure 3"）
        page_range: 页码范围元组（起始页，结束页），从0开始

    Returns:
        list: 找到Figure的页码列表
    """
    doc = fitz.open(pdf_path)
    found_pages = []

    start_page = page_range[0] if page_range else 0
    end_page = page_range[1] if page_range else len(doc)

    print(f"  正在搜索关键词 '{keyword}' (页码范围: {start_page}-{end_page})...")

    for page_num in range(start_page, min(end_page, len(doc))):
        page = doc[page_num]
        text = page.get_text()

        if keyword.lower() in text.lower():
            found_pages.append(page_num)
            print(f"  [OK] 在第 {page_num+1} 页找到 '{keyword}'")

    doc.close()
    return found_pages


def extract_figure_from_pdf(task):
    """
    从PDF中提取指定Figure并保存

    Args:
        task: 提取任务字典

    Returns:
        bool: 是否成功提取
    """
    print(f"\n{'='*60}")
    print(f"任务: {task['name']}")
    print(f"PDF: {os.path.basename(task['pdf_path'])}")
    print(f"目标: {task['description']}")
    print(f"{'='*60}")

    # 检查PDF是否存在
    if not os.path.exists(task['pdf_path']):
        print(f"[ERROR] PDF文件不存在: {task['pdf_path']}")
        return False

    # 搜索Figure位置
    found_pages = search_figure_by_keyword(
        task['pdf_path'],
        task['keyword'],
        task.get('page_range')
    )

    if not found_pages:
        print(f"[WARNING] 未找到关键词 '{task['keyword']}'，将尝试提取整个页面范围的图片")
        # 如果没找到关键词，使用整个页面范围
        doc = fitz.open(task['pdf_path'])
        start, end = task.get('page_range', (0, len(doc)))
        found_pages = list(range(start, min(end, len(doc))))
        doc.close()

    # 从找到的页面中提取图片
    doc = fitz.open(task['pdf_path'])
    all_images = []

    for page_num in found_pages:
        print(f"\n  正在处理第 {page_num+1} 页...")
        page = doc[page_num]
        images = extract_images_from_page(page, min_width=300, min_height=200)

        if images:
            all_images.extend([{**img, "page": page_num+1} for img in images])

    doc.close()

    # 保存图片
    if all_images:
        # 按像素总数排序，选择最大的图片
        all_images.sort(key=lambda x: x['width'] * x['height'], reverse=True)

        # 判断是保存单个还是多个图片
        if 'output_file' in task:
            # Al7075模式：只保存最大的图片
            largest_image = all_images[0]
            print(f"\n  选择最大图片进行保存:")
            print(f"    页码: {largest_image['page']}")
            print(f"    尺寸: {largest_image['width']}x{largest_image['height']}")

            # 保存为PNG
            largest_image['image'].save(task['output_file'], "PNG", dpi=(300, 300))
            print(f"  [OK] 成功保存到: {task['output_file']}")
            return True

        elif 'output_prefix' in task:
            # Al2024模式：保存所有图片
            saved_count = 0
            for idx, img_info in enumerate(all_images):
                if idx == 0:
                    output_file = os.path.join(OUTPUT_DIR, f"{task['output_prefix']}.png")
                else:
                    output_file = os.path.join(OUTPUT_DIR, f"{task['output_prefix']}_{idx+1}.png")

                # 保存为PNG
                img_info['image'].save(output_file, "PNG", dpi=(300, 300))
                print(f"  [OK] 保存图片 {idx+1}: {output_file}")
                print(f"       尺寸: {img_info['width']}x{img_info['height']}, 页码: {img_info['page']}")
                saved_count += 1

            return True
    else:
        print(f"  [ERROR] 未找到符合条件的图片")
        return False


def copy_researchgate_image():
    """
    复制ResearchGate已有的PNG图片到images文件夹
    """
    print(f"\n{'='*60}")
    print(f"任务: 复制ResearchGate PNG图片")
    print(f"{'='*60}")

    source = os.path.join(DATA_DIR, "Al7075", "True-stress-strain-curves-of-as-cast-7075-aluminum-alloy-at-different-strain-rates-and.png")
    target = os.path.join(OUTPUT_DIR, "al7075_researchgate_fig1.png")

    if not os.path.exists(source):
        print(f"[ERROR] 源文件不存在: {source}")
        return False

    try:
        shutil.copy2(source, target)

        # 获取图片信息
        img = Image.open(target)
        print(f"  [OK] 成功复制图片")
        print(f"  源文件: {os.path.basename(source)}")
        print(f"  目标文件: {target}")
        print(f"  图片尺寸: {img.width}x{img.height}")
        return True
    except Exception as e:
        print(f"[ERROR] 复制失败: {e}")
        return False


def generate_report(results):
    """
    生成提取报告

    Args:
        results: 结果字典列表
    """
    print(f"\n{'='*60}")
    print(f"PDF图表提取总结报告")
    print(f"{'='*60}\n")

    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)

    print(f"[OK] 成功提取: {success_count}/{total_count} 个图表\n")

    # 按材料分类统计
    al2024_results = [r for r in results if 'Al2024' in r['name']]
    al7075_results = [r for r in results if 'Al7075' in r['name']]
    other_results = [r for r in results if 'Al2024' not in r['name'] and 'Al7075' not in r['name']]

    print("按材料分类:")
    if al2024_results:
        al2024_success = sum(1 for r in al2024_results if r['success'])
        print(f"  Al2024: {al2024_success}/{len(al2024_results)} 成功")

    if al7075_results:
        al7075_success = sum(1 for r in al7075_results if r['success'])
        print(f"  Al7075: {al7075_success}/{len(al7075_results)} 成功")

    if other_results:
        other_success = sum(1 for r in other_results if r['success'])
        print(f"  其他: {other_success}/{len(other_results)} 成功")

    print("\n详细清单:")
    for i, result in enumerate(results, 1):
        status = "[OK]" if result['success'] else "[FAIL]"
        print(f"{i}. {status} {result['name']}")
        if result['success'] and 'output_file' in result:
            print(f"   保存路径: {result['output_file']}")
            if 'size' in result:
                print(f"   图片尺寸: {result['size']}")
        elif not result['success']:
            print(f"   失败原因: {result.get('error', '未知')}")

    # 下一步建议
    print(f"\n{'='*60}")
    print("下一步操作建议:")
    print("1. 检查提取的图片（位于 images/ 目录）")
    print("2. 使用extract_data_points.py交互式提取数据点")
    print("   或使用WebPlotDigitizer在线工具：https://automeris.io/WebPlotDigitizer/")
    print("3. 导出为CSV格式并保存到 data/fromimage/")
    print(f"{'='*60}\n")


def main():
    """主函数"""
    print("="*60)
    print("PDF图表自动提取工具（统一版本）")
    print("="*60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"总任务数: {len(ALL_EXTRACTION_TASKS)}")
    print("="*60)

    results = []

    # 提取所有PDF图表
    for task in ALL_EXTRACTION_TASKS:
        try:
            success = extract_figure_from_pdf(task)
            result = {
                "name": task['name'],
                "success": success
            }

            # 获取输出文件路径
            if 'output_file' in task:
                result['output_file'] = task['output_file']
                if success and os.path.exists(task['output_file']):
                    img = Image.open(task['output_file'])
                    result['size'] = f"{img.width}x{img.height}"

            results.append(result)
        except Exception as e:
            print(f"[ERROR] 处理任务 '{task['name']}' 时出错: {e}")
            results.append({
                "name": task['name'],
                "success": False,
                "error": str(e)
            })

    # 复制ResearchGate图片
    try:
        success = copy_researchgate_image()
        target = os.path.join(OUTPUT_DIR, "al7075_researchgate_fig1.png")
        result = {
            "name": "Al7075 - ResearchGate Figure 1",
            "success": success,
            "output_file": target
        }

        if success and os.path.exists(target):
            img = Image.open(target)
            result['size'] = f"{img.width}x{img.height}"

        results.append(result)
    except Exception as e:
        print(f"[ERROR] 复制ResearchGate图片时出错: {e}")
        results.append({
            "name": "Al7075 - ResearchGate Figure 1",
            "success": False,
            "error": str(e)
        })

    # 生成报告
    generate_report(results)


if __name__ == "__main__":
    main()
