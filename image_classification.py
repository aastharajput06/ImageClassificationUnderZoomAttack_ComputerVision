from PIL import Image
import torch
import numpy as np
from torchvision import transforms, models
import matplotlib.pyplot as plt

# Function to zoom into the image
def zoom_image(image, zoom_factor):
    w, h = image.size
    zoom_px = int(min(w, h) * (1 - 1/zoom_factor) / 2)
    zoomed = image.crop((zoom_px, zoom_px, w-zoom_px, h-zoom_px))
    return zoomed.resize((w, h), Image.LANCZOS)

# Load the pretrained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the original image
image_path = "panda.jpg"

original_image = Image.open(image_path)

# Define different zoom levels to test
zoom_levels = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

# Store the results for each zoom level
results = []

for zoom in zoom_levels:
    # Apply the zoom transformation
    zoomed_image = zoom_image(original_image, zoom)
    
    # Preprocess the zoomed image
    input_tensor = preprocess(zoomed_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Classify the zoomed image
    with torch.no_grad():
        output = model(input_batch)
    
    # Get the predicted class and confidence score
    _, predicted_idx = torch.max(output, 1)
    confidence = torch.nn.functional.softmax(output, dim=1)[0] * 100
    confidence_value = confidence[predicted_idx].item()
    
    results.append((zoom, zoomed_image, confidence_value))

# Display the results
plt.figure(figsize=(20, 10))

for i, (zoom, img, conf) in enumerate(results):
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(f"Zoom: {zoom:.2f}\nConfidence: {conf:.2f}%")
    plt.axis('off')

plt.tight_layout()
plt.show()
