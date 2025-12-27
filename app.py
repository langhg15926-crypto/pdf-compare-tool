import streamlit as st
import os
import cv2
import numpy as np
import zipfile
import io
from pdf2image import convert_from_path
from PIL import Image

# === 页面配置 ===
st.set_page_config(page_title="PDF比对", layout="wide")

st.title("PDF比对")
st.markdown("### 🔍 核心功能：忽略背景，只看文字")
st.markdown("---")

# === 侧边栏设置 ===
st.sidebar.header("🔧 效果微调")
dpi_setting = st.sidebar.slider("清晰度 (DPI)", 100, 300, 150)
st.sidebar.markdown("---")
# 这是一个更直观的“文字加粗”设置
stroke_width = st.sidebar.slider("文字加粗/容错 (等级)", 1, 8, 3, help="如果你发现扫描件的字比原稿细导致对不上，请调大这个数字。")

# === 核心算法 ===

def align_images(img1_cv, img2_cv):
    """自动对齐：让扫描件尽量贴合原稿"""
    try:
        gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
        
        # 限制特征点数量，提高速度
        orb = cv2.ORB_create(3000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        # 筛选最优质的匹配点
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.2)]
        
        if len(good_matches) < 4: return img2_cv

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        h, w, _ = img1_cv.shape
        aligned_img = cv2.warpPerspective(img2_cv, M, (w, h))
        return aligned_img
    except:
        return img2_cv

def extract_text_only(cv_img):
    """
    【核心修改】文字提取模式
    使用 Otsu 二值化算法，强制把图像分为“纯黑文字”和“纯白背景”
    """
    # 1. 转灰度
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # 2. 局部自适应二值化 (对抗阴影的神器)
    # 这一步会把灰色的阴影全部变成白色，只有深色文字保留为黑色
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 15 # 这里的参数是专门调教过去除阴影的
    )
    
    return binary

def process_page(pil_img1, pil_img2, stroke_level):
    # 格式转换
    img1 = cv2.cvtColor(np.array(pil_img1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(pil_img2), cv2.COLOR_RGB2BGR)

    # 强制尺寸对齐
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 1. 自动对齐
    img2_aligned = align_images(img1, img2)

    # 2. 【关键】提取纯文字 (排除阴影干扰)
    bin1 = extract_text_only(img1)
    bin2 = extract_text_only(img2_aligned)

    # 3. 字体加粗 (形态学膨胀)
    # 扫描件的字通常会虚一点，或者位置歪一点点。
    # 我们把两个图的字都人为“变粗”，这样它们重叠的概率就大了。
    kernel = np.ones((stroke_level, stroke_level), np.uint8)
    bin1 = cv2.dilate(bin1, kernel, iterations=1)
    bin2 = cv2.dilate(bin2, kernel, iterations=1)

    # 4. 找不同 (异或运算)
    diff = cv2.bitwise_xor(bin1, bin2)

    # 5. 过滤噪点 (去除芝麻大小的差异)
    clean_kernel = np.ones((3,3), np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, clean_kernel)
    
    # 6. 画框
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_change = False
    img2_result = img2_aligned.copy()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 面积过滤：
        # 只有当变动区域大于一定像素（比如一个标点符号的大小）才算
        if w * h > 80: 
            found_change = True
            # 画粗一点的红框，方便看
            cv2.rectangle(img2_result, (x-5, y-5), (x+w+5, y+h+5), (0, 0, 255), 2)

    if not found_change:
        return None

    # 7. 拼接显示
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_result, cv2.COLOR_BGR2RGB)
    res_pil = Image.fromarray(img2_rgb)
    orig_pil = Image.fromarray(img1_rgb)
    
    combined = Image.new('RGB', (orig_pil.width + res_pil.width, max(orig_pil.height, res_pil.height)))
    combined.paste(orig_pil, (0, 0))
    combined.paste(res_pil, (orig_pil.width, 0))
    
    return combined

# === 主界面 ===
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("📂 原稿 PDF", type=["pdf"])
with col2:
    file2 = st.file_uploader("📂 扫描件/修改稿 PDF", type=["pdf"])

if file1 and file2:
    if st.button("🚀 开始文字比对", type="primary", use_container_width=True):
        
        with open("temp_v1.pdf", "wb") as f: f.write(file1.getbuffer())
        with open("temp_v2.pdf", "wb") as f: f.write(file2.getbuffer())

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            images_A = convert_from_path("temp_v1.pdf", dpi=dpi_setting)
            images_B = convert_from_path("temp_v2.pdf", dpi=dpi_setting)
            
            total_pages = min(len(images_A), len(images_B))
            results = [] 
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for i in range(total_pages):
                    progress = (i + 1) / total_pages
                    progress_bar.progress(progress)
                    status_text.text(f"正在提取第 {i+1} 页文字骨架 (等级: {stroke_width})...")
                    
                    res_img = process_page(images_A[i], images_B[i], stroke_width)
                    
                    if res_img:
                        page_name = f"page_{i+1}_diff.jpg"
                        results.append((i+1, res_img))
                        img_byte_arr = io.BytesIO()
                        res_img.save(img_byte_arr, format='JPEG')
                        zf.writestr(page_name, img_byte_arr.getvalue())

            status_text.success("✅ 对比完成")
            progress_bar.progress(100)

            if results:
                st.download_button("⬇️ 下载对比结果 (ZIP)", zip_buffer.getvalue(), "文字对比结果.zip", "application/zip", type="primary")
            else:
                st.balloons()
                st.info("太完美了！两份文件的文字内容看起来完全一致。")
            
            st.markdown("---")
            for page_num, img in results:
                st.write(f"### 第 {page_num} 页发现变动：")
                st.image(img, use_container_width=True)
                st.divider()

        except Exception as e:
            st.error(f"发生错误: {e}")
        
        if os.path.exists("temp_v1.pdf"): os.remove("temp_v1.pdf")

        if os.path.exists("temp_v2.pdf"): os.remove("temp_v2.pdf")
