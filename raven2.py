import os
from transformers import AutoImageProcessor, TFAutoModelForImageClassification
from PIL import Image
from settings import IMG_DIR
#import torch.nn.functional as F

model_name = "RavenOnur/Sign-Language"
image_dir = IMG_DIR

# Load processor and model
processor = AutoImageProcessor.from_pretrained(model_name)
model = TFAutoModelForImageClassification.from_pretrained(model_name, from_pt=True)

# Load and preprocess image

image_list = [f for f in os.listdir(image_dir)]
print(image_list)
image_path = os.path.join('Images', 'A495.jpg')
print(image_path)
image = Image.open(image_path)
inputs = processor(images=image, return_tensors="pt")

# Run inference
outputs = model(**inputs)

# Get logits and probabilities
logits = outputs.logits

print(logits)

# if __name__ == '__main__':
#     pass
