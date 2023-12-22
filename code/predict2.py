############测试预测 测试集整个文件#################
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from net import vgg16
import os  # To handle file paths

# Load and process all images from a folder
def load_images_from_folder(folder, transform):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path)
            img = transform(img)
            images.append(img)
        except IOError:
            pass  # You can handle errors here
    return images

# Function to display images
def display_images(images, predictions, classes):
    num_images = len(images)
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(num_images//3, 3, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title("Class: {}".format(classes[predictions[i]], predictions[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Modify these paths according to your setup
folder_path = 'E:\VGG\VGGnet_cat\project1\picture_test'  # Folder containing images
model_path = 'E:\VGG\VGGnet_cat\project1\AKR_B4_12_1915.pth'  # Model path

# Image transformation
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

# Load images
images = load_images_from_folder(folder_path, transform)

# Load and prepare the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = vgg16()
model = torch.load(model_path, map_location=device)
net.load_state_dict(model)
net.eval()

# Predict and store results
predictions = []
for image in images:
    image = torch.reshape(image, (1, 3, 224, 224))
    with torch.no_grad():
        out = net(image)
    out = F.softmax(out, dim=1)
    out = out.data.cpu().numpy()
    pred_class = int(out.argmax(1))
    predictions.append(pred_class)

# Display results
classes = ['0','1'] #['1', '0']
display_images(images, predictions, classes)
