from pathlib import Path
from PIL import Image, ImageDraw
from paddleocr import LayoutDetection

INPUT_IMAGE = Path("answer_sheet.jpg")
OUTPUT_IMAGE = Path("output_image.png")

MODEL_NAME = "PP-DocLayout-L"
DIAGRAM_LABELS = {"image", "figure", "chart", "graph"}


def load_layout_model(model_name=MODEL_NAME):
    return LayoutDetection(model_name=model_name)


def detect_diagrams(input_image, model, output_image=None):
    """Run PP-DocLayout-L and draw boxes around diagram-related regions.

    Args:
        input_image: Path to the image to run detection on.
        model: An initialized LayoutDetection predictor.
        output_image: Optional path to save the annotated image.

    Returns:
        (annotated_image, detections) where annotated_image is a PIL.Image with
        the detected boxes drawn on it and detections is a list of dicts with
        the model's label, score and coordinate for each diagram region.
    """
    input_image = Path(input_image)
    if not input_image.exists():
        raise FileNotFoundError(f"Image not found: {input_image.resolve()}")

    image = Image.open(input_image).convert("RGB")
    draw = ImageDraw.Draw(image)

    results = model.predict(
        str(input_image),
        batch_size=1,
        layout_nms=False,
    )

    detections = []

    for result in results:
        data = result.json
        if callable(data):
            data = data()

        res = data.get("res", data)
        boxes = res.get("boxes", [])

        for box in boxes:
            label = box.get("label", "").lower()
            score = float(box.get("score", 0))

            # Draw only diagram/image-related detections
            if label not in DIAGRAM_LABELS:
                continue

            x1, y1, x2, y2 = map(float, box["coordinate"])
            coords = (int(x1), int(y1), int(x2), int(y2))

            draw.rectangle(coords, outline="red", width=4)
            draw.text(
                (int(x1), max(0, int(y1) - 22)),
                f"{label} {score:.2f}",
                fill="red",
                stroke_width=1,
                stroke_fill="white",
            )
            detections.append(
                {"label": label, "score": score, "coordinate": coords}
            )

    if output_image is not None:
        image.save(output_image)

    return image, detections


def main():
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"Image not found: {INPUT_IMAGE.resolve()}")

    model = load_layout_model()
    _, detections = detect_diagrams(INPUT_IMAGE, model, OUTPUT_IMAGE)

    print(f"Detected diagram/image regions: {len(detections)}")
    print(f"Final annotated image: {OUTPUT_IMAGE.resolve()}")

    if len(detections) == 0:
        print("No diagram regions were detected. No boxes were added.")


if __name__ == "__main__":
    main()
