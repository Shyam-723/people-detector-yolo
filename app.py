from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os
import gradio as gr
from PIL import Image, ImageDraw

# Load .env file
load_dotenv()

# Read API key from environment variable
roboflow_api_key = os.getenv("ROBOFLOW_API")
if not roboflow_api_key:
    raise ValueError("ROBOFLOW_API key not found in environment variables.")

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="krlla3qTFnZHPjhqGDnM"
)

# Inference function
def detect_people(image):
    result = CLIENT.infer(image, model_id="people-detector-emgwo/1")

    image = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(image)

    for pred in result["predictions"]:
        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]

        # Calculate box coordinates
        left = x - w / 2
        top = y - h / 2
        right = x + w / 2
        bottom = y + h / 2

        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        draw.text((left, top - 10), pred["class"], fill="red")

    return image

# Gradio interface (optional)
demo = gr.Interface(
    fn=detect_people,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=gr.Image(label="Detected People"),
    title="People Detector",
    description="Detect people using Roboflow's inference SDK.",
)

if __name__ == "__main__":
    demo.launch()