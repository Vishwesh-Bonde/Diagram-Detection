import io
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from test_detection import detect_diagrams, load_layout_model

TITLE = "AI-Based Diagram Detection"
SUBTITLE = (
    "Detect and visualize diagrams in document images using "
    "PaddleOCR PP-DocLayout-L."
)
RESULT_FILENAME = "diagram_detection_result.png"
ALLOWED_TYPES = ["jpg", "jpeg", "png"]


@st.cache_resource(show_spinner=False)
def get_model():
    return load_layout_model()


def validate_image(file_bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except (UnidentifiedImageError, OSError):
        return None
    return image.convert("RGB")


def run_detection(file_bytes, suffix):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        model = get_model()
        return detect_diagrams(tmp_path, model)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.set_page_config(page_title=TITLE, layout="centered")

st.title(TITLE)
st.write(SUBTITLE)
st.caption(
    "The first detection downloads the PP-DocLayout-L model and may take a "
    "moment. Later runs reuse the cached model."
)

uploaded = st.file_uploader(
    "Upload a document image (JPG, JPEG or PNG)",
    type=ALLOWED_TYPES,
)

if uploaded is not None:
    upload_id = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("upload_id") != upload_id:
        st.session_state["upload_id"] = upload_id
        st.session_state.pop("result", None)

    file_bytes = uploaded.getvalue()
    image = validate_image(file_bytes)

    if image is None:
        st.error(
            "The uploaded file is not a valid image. Please upload a JPG, "
            "JPEG or PNG file."
        )
    else:
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Detect Diagrams", type="primary"):
            with st.spinner("Running detection..."):
                try:
                    suffix = Path(uploaded.name).suffix or ".png"
                    annotated, detections = run_detection(file_bytes, suffix)
                    st.session_state["result"] = {
                        "png": image_to_png_bytes(annotated),
                        "detections": detections,
                    }
                except Exception as error:
                    st.session_state.pop("result", None)
                    st.error(
                        "Detection failed. Please try again with a different "
                        "image. If the problem persists, make sure the "
                        "PP-DocLayout-L model is available."
                    )
                    with st.expander("Technical details"):
                        st.write(str(error))

    result = st.session_state.get("result")
    if result is not None:
        st.subheader("Result")
        detections = result["detections"]

        if detections:
            st.success(f"Detected {len(detections)} diagram region(s).")
        else:
            st.info("No diagrams detected in this image.")

        st.image(
            result["png"],
            caption="Detected diagram regions",
            use_container_width=True,
        )

        st.download_button(
            "Download Result Image",
            data=result["png"],
            file_name=RESULT_FILENAME,
            mime="image/png",
        )

        if detections:
            with st.expander("Detection details"):
                rows = [
                    {
                        "Label": item["label"],
                        "Confidence": round(item["score"], 3),
                        "Box (x1, y1, x2, y2)": item["coordinate"],
                    }
                    for item in detections
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)
