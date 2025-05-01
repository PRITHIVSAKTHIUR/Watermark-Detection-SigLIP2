import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Watermark-Detection-SigLIP2"  # Update this if using a different path
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {
    "0": "No Watermark",
    "1": "Watermark"
}

def classify_watermark(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_watermark,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2, label="Watermark Detection"),
    title="Watermark-Detection-SigLIP2",
    description="Upload an image to detect whether it contains a watermark."
)

if __name__ == "__main__":
    iface.launch()
