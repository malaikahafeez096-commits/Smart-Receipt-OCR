import easyocr
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Receipt OCR", layout="wide")
st.title("Receipt Field Detection & OCR")
st.write("Upload a receipt image — YOLO will detect the fields and EasyOCR will extract the text.")

MODEL_PATH = "best.pt"  # Provide the exact path if it is in a subfolder


@st.cache_resource
def load_models(model_path: str):
    detector = YOLO(model_path)
    reader = easyocr.Reader(["en"], gpu=False)
    return detector, reader


conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.30, 0.05)
padding = st.sidebar.slider("Crop Padding (px)", 0, 15, 4, 1)

uploaded_file = st.file_uploader("Upload Receipt Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Loading models and processing receipt..."):
        detector, reader = load_models(MODEL_PATH)

        pil_img = Image.open(uploaded_file).convert("RGB")
        img_rgb = np.array(pil_img)
        h, w, _ = img_rgb.shape

        results = detector(img_rgb, conf=conf_thresh)[0]
        annotated_rgb = results.plot()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected Fields")
        st.image(annotated_rgb, use_container_width=True)

    extracted_rows = []
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        # Sort boxes in top-to-bottom order
        order = np.argsort(boxes[:, 1])

        for idx in order:
            x1, y1, x2, y2 = map(int, boxes[idx])
            label = detector.names[cls_ids[idx]]
            score = float(confs[idx])

            crop = img_rgb[
                max(0, y1 - padding):min(h, y2 + padding),
                max(0, x1 - padding):min(w, x2 + padding),
            ]
            if crop.size == 0:
                continue

            ocr_text = " ".join(reader.readtext(crop, detail=0)).strip()
            extracted_rows.append({
                "Field": label,
                "Text": ocr_text,
                "Confidence": round(score, 3),
            })

    with col2:
        st.subheader("Extracted OCR Data")
        if extracted_rows:
            st.dataframe(extracted_rows, use_container_width=True)
        else:
            st.info("No fields detected. Try lowering the confidence threshold.")
