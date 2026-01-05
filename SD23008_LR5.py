import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# Step 1 & 2: Configure Page and Layout [cite: 72, 73]
st.set_page_config(page_title="Computer Vision Classifier", layout="centered")
st.title("AI Image Classification with ResNet18")

# Step 3: Configure CPU settings [cite: 74]
device = torch.device('cpu')

# Step 4: Load pre-trained ResNet18 and set to evaluation mode [cite: 75]
@st.cache_resource
def load_model():
    model = models.resnet18(weights='DEFAULT')
    model.to(device)
    model.eval()
    return model

model = load_model()

# Step 5: Image preprocessing transformations [cite: 76]
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet labels for output display
@st.cache_data
def load_labels():
    import requests
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    return response.text.splitlines()

labels = load_labels()

# Step 6: User Interface for file upload [cite: 77]
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Step 7: Convert to tensor and perform inference (no gradient) [cite: 78]
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    # Step 8: Apply Softmax and get Top-5 predictions [cite: 79]
    probabilities = F.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    results = []
    for i in range(top5_prob.size(0)):
        results.append({"Class": labels[top5_catid[i]], "Probability": top5_prob[i].item()})

    # Step 9: Visualize with Bar Chart [cite: 80]
    df = pd.DataFrame(results)
    st.write("### Top 5 Predictions")
    st.bar_chart(df.set_index("Class"))
    st.table(df)

# Step 10: Discussion Placeholder [cite: 81, 82]
st.info("Discussion: The model follows a path from pixel input -> preprocessing -> "
        "feature extraction via ResNet18 layers -> Softmax classification.")