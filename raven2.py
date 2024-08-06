from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch.nn.functional as F

# Load processor and model
processor = AutoImageProcessor.from_pretrained("RavenOnur/Sign-Language")
model = AutoModelForImageClassification.from_pretrained("RavenOnur/Sign-Language")

# Load and preprocess image
image_path = "image.jpg"
image = Image.open(image_path)
inputs = processor(images=image, return_tensors="pt")

# Run inference
outputs = model(**inputs)

# Get logits and probabilities
logits = outputs.logits