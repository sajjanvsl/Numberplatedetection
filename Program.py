import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import re
import os
import time
from io import BytesIO

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Number Plate Detection", page_icon="🚗", layout="wide")

# ----------------- HEADER -----------------
st.markdown("""
<style>
.main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem; }
.college-info { background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 5px solid #4CAF50; margin-bottom: 1rem; }
.footer { text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f1f1f1; border-radius: 10px; }
</style>
<div class="main-header"><h1>🚗 Automatic Number Plate Detection System</h1><p>AI-powered vehicle number plate recognition</p></div>
<div class="college-info"><h3>📚 Dept. of Computer Science and Application</h3><h4>🏛️ Govt. First Grade College for Women, Jamkhandi</h4><p>Project by: [Your Name] | Guided by: [Guide Name]</p></div>
""", unsafe_allow_html=True)

# ----------------- TESSERACT PATH (CHANGE IF NEEDED) -----------------
if 'tesseract_path' not in st.session_state:
    st.session_state.tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = st.session_state.tesseract_path
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("⚙️ Settings")
min_aspect = st.sidebar.slider("Min aspect ratio (width/height)", 1.5, 5.0, 2.0, 0.1)
max_aspect = st.sidebar.slider("Max aspect ratio", 4.0, 10.0, 6.0, 0.1)
min_area_ratio = st.sidebar.slider("Min plate area (fraction of image)", 0.001, 0.05, 0.003, 0.001)
psm_mode = st.sidebar.selectbox("OCR Mode", ["7 - Single line", "8 - Single word", "6 - Block"], index=0)
psm_value = psm_mode.split(" - ")[0]
whitelist = st.sidebar.text_input("Character whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

if st.sidebar.button("🔧 Test Tesseract"):
    try:
        st.sidebar.success(f"Version: {pytesseract.get_tesseract_version()}")
    except Exception as e:
        st.sidebar.error(f"Tesseract error: {e}")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** If auto detection fails, use the slider-based manual crop below.")

# ----------------- IMPROVED AUTO DETECTION -----------------
def detect_plate_auto(img):
    """Edge density + morphological gradient + contour scoring."""
    h, w = img.shape[:2]
    area_thresh = w * h * min_area_ratio

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Morphological gradient to highlight plate borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    # Adaptive threshold on original gray for text regions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 19, 9)

    # Combine gradient and text regions
    combined = cv2.bitwise_or(grad, thresh)

    # Close gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_plate = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_thresh:
            continue

        x, y, wc, hc = cv2.boundingRect(cnt)
        if wc == 0 or hc == 0:
            continue
        aspect = wc / hc
        if aspect < min_aspect or aspect > max_aspect:
            continue

        # Rectangularity and solidity
        rect_area = wc * hc
        rectangularity = area / rect_area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        score = area * rectangularity * solidity

        if score > best_score:
            best_score = score
            best_plate = (x, y, wc, hc)

    if best_plate is None:
        return None, None

    x, y, wc, hc = best_plate
    # Add 10% margin
    margin_x = int(wc * 0.1)
    margin_y = int(hc * 0.1)
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    wc = min(w - x, wc + 2 * margin_x)
    hc = min(h - y, hc + 2 * margin_y)

    plate_crop = img[y:y+hc, x:x+wc]
    cv2.rectangle(img, (x, y), (x+wc, y+hc), (0, 255, 0), 3)
    return plate_crop, img

# ----------------- OCR WITH MULTIPLE METHODS -----------------
def enhance_and_ocr(plate_img, psm, whitelist):
    if plate_img is None:
        return "", 0
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img

    h, w = gray.shape
    if w < 800:
        scale = 800 / w
        gray = cv2.resize(gray, (800, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.medianBlur(gray, 3)

    # Try different binarizations
    methods = [
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    ]

    best_text, best_conf = "", 0
    config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}'

    for thresh in methods:
        try:
            data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
            text_parts, confs = [], []
            for i, conf in enumerate(data['conf']):
                if conf != '-1' and int(conf) > 0:
                    confs.append(int(conf))
                    text_parts.append(data['text'][i])
            if confs:
                avg_conf = sum(confs) / len(confs)
                clean = re.sub(r'[^A-Z0-9]', '', ''.join(text_parts).upper())
                if len(clean) > len(best_text) and avg_conf > 20:
                    best_text = clean
                    best_conf = avg_conf
        except:
            continue

    best_text = best_text.replace('O', '0').replace('I', '1').replace('Z', '2')
    return best_text, best_conf

# ----------------- MANUAL CROP USING SLIDERS (RELIABLE) -----------------
def manual_crop_sliders(img):
    st.subheader("✂️ Manual Crop (Sliders)")
    h, w = img.shape[:2]
    col1, col2 = st.columns(2)
    with col1:
        x = st.slider("Left (X)", 0, w-1, int(w*0.3))
        y = st.slider("Top (Y)", 0, h-1, int(h*0.4))
    with col2:
        width = st.slider("Width", 50, w - x, int(w*0.4))
        height = st.slider("Height", 20, h - y, int(h*0.15))
    
    # Preview
    preview = img.copy()
    cv2.rectangle(preview, (x, y), (x+width, y+height), (0,255,0), 3)
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Selected Region", use_container_width=True)
    
    if st.button("✅ Confirm Crop"):
        crop = img[y:y+height, x:x+width]
        return crop, preview
    return None, None

# ----------------- SESSION STATE -----------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ----------------- MAIN UI -----------------
uploaded_file = st.file_uploader("📸 Upload Vehicle Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    img_np = np.array(pil_img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.subheader("📷 Uploaded Image")
    st.image(pil_img, width=600)

    # Auto detection button
    if st.button("🔍 Auto Detect Number Plate", type="primary"):
        with st.spinner("Detecting plate region..."):
            plate_crop, annotated = detect_plate_auto(img_cv.copy())
        if plate_crop is not None:
            st.success("✅ Auto detection successful!")
            st.session_state['plate_crop'] = plate_crop
            st.session_state['annotated'] = annotated
            st.session_state['crop_method'] = "Auto"
        else:
            st.error("❌ Auto detection failed. Please use manual crop.")

    st.markdown("---")
    # Always show manual crop option
    manual_crop, preview = manual_crop_sliders(img_cv)
    if manual_crop is not None:
        st.session_state['plate_crop'] = manual_crop
        st.session_state['annotated'] = preview
        st.session_state['crop_method'] = "Manual"
        st.success("Manual crop saved!")

    # Show results if a crop exists
    if 'plate_crop' in st.session_state:
        plate = st.session_state['plate_crop']
        annotated = st.session_state['annotated']

        tab1, tab2, tab3 = st.tabs(["📍 Annotated Image", "🔍 Extracted Plate", "📝 OCR Result"])

        with tab1:
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, width=600)
            buf = BytesIO()
            Image.fromarray(annotated_rgb).save(buf, format="PNG")
            st.download_button("📸 Download Annotated", data=buf.getvalue(), file_name="annotated.png")

        with tab2:
            st.image(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), width=400)
            buf2 = BytesIO()
            Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)).save(buf2, format="PNG")
            st.download_button("🖼️ Download Crop", data=buf2.getvalue(), file_name="plate_crop.png")

        with tab3:
            with st.spinner("Running OCR..."):
                ocr_text, conf = enhance_and_ocr(plate, psm_value, whitelist)
            col1, col2 = st.columns(2)
            col1.metric("Confidence", f"{conf:.1f}%")
            if conf > 60: col2.success("High")
            elif conf > 30: col2.warning("Medium")
            else: col2.error("Low")

            if ocr_text and conf > 20:
                st.success(f"**Detected Number:** `{ocr_text}`")
                st.session_state.history.append({
                    "time": time.strftime("%H:%M:%S"),
                    "plate": ocr_text,
                    "conf": conf
                })
                st.download_button("📄 Download Number", data=ocr_text.encode(), file_name="plate.txt")
            else:
                st.error("OCR could not read the plate.")
                st.info("✏️ You can manually type the number below:")
                manual_num = st.text_input("Enter plate text:", placeholder="e.g., KA01AB1234")
                if st.button("Save Manual"):
                    if manual_num:
                        clean = re.sub(r'[^A-Z0-9]', '', manual_num.upper())
                        st.success(f"Saved: {clean}")
                        st.session_state.history.append({
                            "time": time.strftime("%H:%M:%S"),
                            "plate": clean,
                            "conf": 100
                        })
                with st.expander("🔍 What Tesseract saw"):
                    if len(plate.shape)==3:
                        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = plate
                    h,w = gray.shape
                    if w < 800:
                        scale = 800/w
                        gray = cv2.resize(gray, (800, int(h*scale)))
                    gray = cv2.medianBlur(gray,3)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    st.image(thresh, caption="Preprocessed", width=400)

    # Clear crop button
    if st.button("🗑️ Clear Crop"):
        for k in ['plate_crop', 'annotated', 'crop_method']:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

else:
    st.info("👈 Upload an image to start")

# ----------------- HISTORY SIDEBAR -----------------
st.sidebar.markdown("---")
st.sidebar.subheader("📜 Recent Detections")
if st.sidebar.button("Clear History"):
    st.session_state.history = []
for entry in reversed(st.session_state.history[-5:]):
    st.sidebar.write(f"`{entry['plate']}` ({entry['conf']:.0f}%) at {entry['time']}")

# ----------------- FOOTER -----------------
st.markdown("""
<div class="footer">
    <p>Developed with ❤️ using Streamlit, OpenCV, and Tesseract OCR</p>
    <p>© 2025 Dept. of Computer Science and Application, Govt. First Grade College for Women, Jamkhandi</p>
</div>
""", unsafe_allow_html=True)