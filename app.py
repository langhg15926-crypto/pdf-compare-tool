import streamlit as st
import os
import cv2
import numpy as np
import zipfile
import io
import gc
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image

# ==========================================
# 【核心配置】请确保路径与你电脑实际位置一致
# ==========================================
# 根据你之前的截图，你的 Poppler 文件在 C:\bin
POPPLER_PATH = r"C:\bin" 

# === 页面配置 ===
st.set_page_config(page_title="Nanobanana 文字校对本地版", layout="wide")
st.title("🍌 Nanobanana 文字校对专版 (本地极致优化版)")
st.markdown("---")

# === 侧边栏设置 ===
st.sidebar.header("🔧 效果微调")
dpi_setting = st.sidebar.slider("清晰度 (DPI)", 80, 200, 120, help="如果电脑卡或内存报错，请调低此值")
stroke_width = st.sidebar.slider("文字加粗/容错 (等级)", 1, 8, 3)
st.sidebar.info(f"📍 当前 Poppler 路径: {POPPLER_PATH}")

# === 核心算法库 ===

def align_images(img1_cv, img2_cv):
    """自动对齐：让扫描件尽量贴合原稿"""
    try:
        gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(2000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.15)]
        if len(good_matches) < 4: return img2_cv
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        h, w, _ = img1_cv.shape
        return cv2.warpPerspective(img2_cv, M, (w, h))
    except:
        return img2_cv

def extract_text_only(cv_img):
    """提取纯文字骨架 (抗阴影)"""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 15
    )
    return binary

def process_page(pil_img1, pil_img2, stroke_level):
    """单页对比核心逻辑"""
    img1 = cv2.cvtColor(np.array(pil_img1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(pil_img2), cv2.COLOR_RGB2BGR)
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    img2_aligned = align_images(img1, img2)
    bin1 = extract_text_only(img1)
    bin2 = extract_text_only(img2_aligned)

    kernel = np.ones((stroke_level, stroke_level), np.uint8)
    bin1 = cv2.dilate(bin1, kernel, iterations=1)
    bin2 = cv2.dilate(bin2, kernel, iterations=1)

    diff = cv2.bitwise_xor(bin1, bin2)
    clean_kernel = np.ones((3,3), np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, clean_kernel)
    
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_change = False
    img2_result = img2_aligned.copy()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 80: 
            found_change = True
            cv2.rectangle(img2_result, (x-5, y-5), (x+w+5, y+h+5), (0, 0, 255), 2)

    if not found_change:
        return None

    res_pil = Image.fromarray(cv2.cvtColor(img2_result, cv2.COLOR_BGR2RGB))
    orig_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    
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
    if st.button("🚀 开始文字比对", type="primary", width="stretch"):
        
        # --- 诊断1：检查 Poppler 路径 ---
        if not os.path.exists(os.path.join(POPPLER_PATH, "pdfinfo.exe")):
            st.error(f"❌ 找不到 Poppler 核心文件！")
            st.write(f"请检查代码第 16 行。当前设置的路径 `{POPPLER_PATH}` 目录下没找到 `pdfinfo.exe`。")
            st.stop()

        # 1. 保存到本地临时文件
        with open("temp_v1.pdf", "wb") as f: f.write(file1.getbuffer())
        with open("temp_v2.pdf", "wb") as f: f.write(file2.getbuffer())

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 2. 获取总页数
            info = pdfinfo_from_path("temp_v1.pdf", poppler_path=POPPLER_PATH)
            total_pages = info["Pages"]
            
            results = [] 
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                # 【核心优化】：按页循环处理，节省内存
                for i in range(total_pages):
                    curr_page = i + 1
                    status_text.text(f"🔍 正在对比第 {curr_page}/{total_pages} 页...")
                    
                    # 每次只加载当前页
                    img_a = convert_from_path("temp_v1.pdf", dpi=dpi_setting, first_page=curr_page, last_page=curr_page, poppler_path=POPPLER_PATH)
                    img_b = convert_from_path("temp_v2.pdf", dpi=dpi_setting, first_page=curr_page, last_page=curr_page, poppler_path=POPPLER_PATH)
                    
                    if img_a and img_b:
                        res_img = process_page(img_a[0], img_b[0], stroke_width)
                        
                        if res_img:
                            results.append((curr_page, res_img))
                            img_byte_arr = io.BytesIO()
                            res_img.save(img_byte_arr, format='JPEG', quality=85)
                            zf.writestr(f"page_{curr_page}_diff.jpg", img_byte_arr.getvalue())

                    # 强制回收内存
                    del img_a, img_b
                    gc.collect() 
                    progress_bar.progress(curr_page / total_pages)

            status_text.success(f"✅ 比对完成！共发现 {len(results)} 页差异。")

            if results:
                st.download_button("⬇️ 下载全部变动页面 (ZIP)", zip_buffer.getvalue(), "文字比对结果.zip", "application/zip")
                st.divider()
                for page_num, img in results:
                    st.write(f"### 第 {page_num} 页差异预览：")
                    st.image(img, width="stretch")
            else:
                st.balloons()
                st.info("太棒了！两份文件的文字内容看起来完全一致。")

        except Exception as e:
            st.error(f"❌ 运行过程中出错: {e}")
        
        # 3. 最后清理临时文件
        if os.path.exists("temp_v1.pdf"): os.remove("temp_v1.pdf")
        if os.path.exists("temp_v2.pdf"): os.remove("temp_v2.pdf")
