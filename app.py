##WRITE THE BACKSIDE
%%writefile app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import json
import os
import tempfile



# --------------------
# Load Model + Labels
# --------------------

model = YOLO("/content/drive/MyDrive/Dental AI/best.pt")


with open("/content/drive/MyDrive/Dental AI/test_annotations_anonymized.json") as f:
    coco = json.load(f)


# --------------------
# Functions
# --------------------

def draw_ground_truth(image, filename):

    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Find image ID
    image_id = None

    for item in coco["images"]:
      if os.path.splitext(item["file_name"])[0] == os.path.splitext(filename)[0]:
          image_id = item["id"]
          break

    if image_id is None:
        return img


    # Draw annotations
    for ann in coco["annotations"]:

      if ann["image_id"] == image_id and ann["category_id"] == 1:

            x, y, w, h = ann["bbox"]

            draw.rectangle(
                [
                    x,
                    y,
                    x+w,
                    y+h
                ],
                outline="red",
                width=4
            )

    return img



# --------------------
# Website
# --------------------

st.set_page_config(
    page_title="Dental AI",
    page_icon="🦷"
)

st.title("🦷 Dental Cavity Detection AI")

st.write(
    "Compare YOLO predictions with dentist annotations."
)


uploaded = st.file_uploader(
    "Upload X-ray",
    type=["png","jpg","jpeg"]
)


confidence = st.slider(
    "Confidence",
    0.1,
    1.0,
    0.4
)


if uploaded:

    image = Image.open(uploaded).convert("RGB")

    filename = uploaded.name


    # --------------------
    # YOLO Prediction
    # --------------------

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )

    image.save(temp.name)


    results = model.predict(
        source=temp.name,
        conf=confidence,
        imgsz=640
    )




    prediction_image = Image.fromarray(
      results[0].plot()
    )

    # --------------------
    # AI Report
    # --------------------

    num_boxes = len(results[0].boxes)

    st.subheader("🦷 AI Cavity Report")

    st.metric(
        "Possible cavities detected",
        num_boxes
    )

    if num_boxes > 0:
        confidences = results[0].boxes.conf.cpu().numpy()

        st.metric(
            "Average confidence",
            f"{confidences.mean():.1%}"
        )

        st.metric(
            "Highest confidence",
            f"{confidences.max():.1%}"
        )

    else:
        st.write("No cavities detected above this confidence threshold.")


    # --------------------
    # Ground Truth
    # --------------------

    truth_image = draw_ground_truth(
    image,
    filename
)

    # Check whether this image actually exists in the annotations
    has_annotation = any(
        os.path.splitext(item["file_name"])[0] == os.path.splitext(filename)[0]
        for item in coco["images"]
    )

    # --------------------
    # Display
    # --------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🦷 AI Prediction")
        st.image(
            prediction_image,
            use_container_width=True
        )

    with col2:
        st.subheader("👨‍⚕️ Dentist Annotation")

        if has_annotation:
            st.image(
                truth_image,
                use_container_width=True
            )
        else:
            st.info(
                "No dentist annotation is available for this image. "
                "The AI prediction above can still be viewed."
            )
