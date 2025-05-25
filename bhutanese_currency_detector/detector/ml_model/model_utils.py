import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
# import models
# import nn
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import os

# Dictionary mapping class indices to currency values
CURRENCY_CLASSES = {
    0: {'name': 'Nu. 1', 'audio': 'Nu. 1.mp3', 'description': 'One Ngultrum note of Bhutan'},
    1: {'name': 'Nu. 5', 'audio': 'Nu. 5.mp3', 'description': 'Five Ngultrum note of Bhutan'},
    2: {'name': 'Nu. 10', 'audio': 'Nu.10.mp3', 'description': 'Ten Ngultrum note of Bhutan'},
    3: {'name': 'Nu. 20', 'audio': 'Nu.20.mp3', 'description': 'Twenty Ngultrum note of Bhutan'},
    4: {'name': ' Nu. 50', 'audio': 'Nu. 50.mp3', 'description': 'Fifty Ngultrum note of Bhutan'},
    5: {'name': 'Nu. 100', 'audio': 'Nu. 100.mp3', 'description': 'One Hundred Ngultrum note of Bhutan'},
    6: {'name': 'Nu. 500', 'audio': 'Nu. 500.mp3', 'description': 'Five Hundred Ngultrum note of Bhutan'},
    7: {'name': 'Nu. 1000', 'audio': 'Nu. 1000.mp3', 'description': 'One Thousand Ngultrum note of Bhutan'},
    8: {'name': 'Unknown', 'audio': 'unknown.mp3', 'description': 'This is not a Currency'},
}

# Get the path to the model file
def get_model_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'currency_model.pth')
    return model_path

# Load the trained model
# def load_model():
#     model_path = get_model_path()
#     # Check if CUDA is available
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Load the model
#     model = torch.load(model_path, map_location=device)
#     model.eval()  # Set the model to evaluation mode
#     return model
def load_model():
    model_path = get_model_path()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model architecture
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CURRENCY_CLASSES))  # 8 classes for 8 currency types

    # Load state_dict instead of the full model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
# def load_model():
#     model_path = get_model_path()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Define the same model architecture used during training
#     model = models.resnet18(pretrained=False)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, len(CURRENCY_CLASSES))  # 8 classes for 8 currency types

#     # Load state_dict
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)

#     model.to(device)
#     model.eval()
#     return model

# Preprocess image for model input
def preprocess_image(image):
    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Detect currency from PIL Image
def detect_currency_from_image(image):
    model = load_model()
    
    # Preprocess the image
    image_tensor = preprocess_image(image)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
    
    # Get currency details
    if class_idx in CURRENCY_CLASSES:
        return CURRENCY_CLASSES[class_idx]
    else:
        return {'name': 'Unknown', 'audio': None, 'description': 'Could not detect valid Bhutanese currency'}

# Process base64 image data
def process_base64_image(base64_data):
    # Remove the data URL prefix if present
    if ',' in base64_data:
        base64_data = base64_data.split(',')[1]
    
    # Decode base64 data to binary
    image_data = base64.b64decode(base64_data)
    
    # Convert binary to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Decode numpy array as image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB (PIL uses RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(img_rgb)
    
    return pil_img