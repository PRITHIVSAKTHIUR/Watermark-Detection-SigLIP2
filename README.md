![5.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/VXSOLkmcLM1t6XhTcYXUh.png)

# **Watermark-Detection-SigLIP2**

> **Watermark-Detection-SigLIP2** is a vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for **binary image classification**. It is trained to detect whether an image **contains a watermark or not**, using the **SiglipForImageClassification** architecture.


```py
Classification Report:
              precision    recall  f1-score   support

No Watermark     0.9290    0.9722    0.9501     12779
   Watermark     0.9622    0.9048    0.9326      9983

    accuracy                         0.9427     22762
   macro avg     0.9456    0.9385    0.9414     22762
weighted avg     0.9435    0.9427    0.9424     22762
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/_rKqtbSJbglsRiXmRF1ij.png)

---

## **Label Space: 2 Classes**

The model classifies an image as either:

```
Class 0: "No Watermark"
Class 1: "Watermark"
```

---

## **Install dependencies**

```bash
pip install -q transformers torch pillow gradio
```

---

## **Inference Code**

```python
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
```

---

## **Demo Inference**

> Watermark

![Screenshot 2025-04-29 at 18-22-52 Watermark-Detection-SigLIP2.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/CFuKWRrFM9IynRph3exwU.png)
![Screenshot 2025-04-29 at 18-27-09 Watermark-Detection-SigLIP2.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/5k0ySqH3qufhhyM4vv6fC.png)

> No Watermark

![Screenshot 2025-04-29 at 18-25-43 Watermark-Detection-SigLIP2.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Eh4gcpefV58I3M1PB93P9.png)
![Screenshot 2025-04-29 at 18-31-34 Watermark-Detection-SigLIP2.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Q1LlfE6z1VJFz58MAjEXd.png)


## **Intended Use**

**Watermark-Detection-SigLIP2** is useful in scenarios such as:

- **Content Moderation** – Automatically detect watermarked content on image sharing platforms.
- **Dataset Cleaning** – Filter out watermarked images from training datasets.
- **Copyright Enforcement** – Monitor and flag usage of watermarked media.
- **Digital Forensics** – Support analysis of tampered or protected media assets.
