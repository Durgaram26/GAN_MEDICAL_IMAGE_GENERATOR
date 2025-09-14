import os
import time
import threading
import random
import numpy as np
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# --------------------------
# Configuration
# --------------------------
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --------------------------
# Flask Application Setup
# --------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key_here"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# --------------------------
# Model Definitions (Streamlit Logic)
# --------------------------

# Basic GAN Generator (for 64x64 images)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.net(input)

# Basic GAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_maps=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.net(input).view(-1, 1).squeeze(1)

# Simplified StyleGAN Generator
class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64):
        super(StyleGANGenerator, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(True),
            nn.Linear(latent_dim, latent_dim)
        )
        self.synthesis = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        if input.dim() > 2:
            input = input.view(input.size(0), -1)
        style = self.mapping(input)
        style = style.unsqueeze(2).unsqueeze(3)
        return self.synthesis(style)

# Simplified StyleGAN2 ADA Generator
class StyleGAN2ADAGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64):
        super(StyleGAN2ADAGenerator, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim)
        )
        self.synthesis = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        if input.dim() > 2:
            input = input.view(input.size(0), -1)
        style = self.mapping(input)
        style = style.unsqueeze(2).unsqueeze(3)
        return self.synthesis(style)

# For super-resolution, we use a simplified ESRGAN Generator and Discriminator
class ESRGANGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, upscale_factor=2):
        super(ESRGANGenerator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out = self.res_block(out1) + out1
        out = self.upsample(out)
        out = self.conv2(out)
        return self.tanh(out)

class ESRGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_features=64):
        super(ESRGANDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features * 2, 3, 2, 1),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, 3, 2, 1),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features * 4, 1, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x).view(-1))

# For SRGAN we use the same architecture as ESRGAN here
class SRGANGenerator(ESRGANGenerator):
    pass

class SRGANDiscriminator(ESRGANDiscriminator):
    pass

# --------------------------
# Dataset Definition
# --------------------------
class UploadedImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.image_files = [os.path.join(image_folder, f)
                            for f in os.listdir(image_folder)
                            if f.split('.')[-1].lower() in ALLOWED_EXTENSIONS]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# --------------------------
# Training and Generation Functions
# --------------------------

def train_model(model_type, dataset, epochs, batch_size, lr, latent_dim=100, upscale_factor=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models based on type
    if model_type in ["GAN", "StyleGAN", "StyleGAN2 ADA"]:
        if model_type == "GAN":
            netG = Generator(latent_dim=latent_dim).to(device)
        elif model_type == "StyleGAN":
            netG = StyleGANGenerator(latent_dim=latent_dim).to(device)
        else:  # StyleGAN2 ADA
            netG = StyleGAN2ADAGenerator(latent_dim=latent_dim).to(device)
        netD = Discriminator().to(device)
        criterion = nn.BCELoss()
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    elif model_type in ["ESRGAN", "SRGAN"]:
        if model_type == "ESRGAN":
            netG = ESRGANGenerator(upscale_factor=upscale_factor).to(device)
            netD = ESRGANDiscriminator().to(device)
        else:  # SRGAN
            netG = SRGANGenerator(upscale_factor=upscale_factor).to(device)
            netD = SRGANDiscriminator().to(device)
        criterion_adv = nn.BCELoss()
        criterion_pixel = nn.L1Loss()
        optimizerD = optim.Adam(netD.parameters(), lr=lr)
        optimizerG = optim.Adam(netG.parameters(), lr=lr)
    else:
        raise ValueError("Unknown model type")

    real_label = 1.0
    fake_label = 0.0
    G_losses, D_losses = [], []

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            current_batch = data.size(0)
            ############################
            # Update Discriminator
            ############################
            netD.zero_grad()
            if model_type in ["GAN", "StyleGAN", "StyleGAN2 ADA"]:
                real_images = data.to(device)
                label = torch.full((current_batch,), real_label, dtype=torch.float, device=device)
                output = netD(real_images)
                errD_real = criterion(output, label)
                errD_real.backward()

                noise = torch.randn(current_batch, latent_dim, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # Update Generator
                ############################
                netG.zero_grad()
                label.fill_(real_label)
                output = netD(fake)
                errG = criterion(output, label)
                errG.backward()
                optimizerG.step()
            else:
                # For ESRGAN/SRGAN training: update discriminator first
                hr_images = data.to(device)
                label = torch.full((current_batch,), real_label, dtype=torch.float, device=device)
                output_real = netD(hr_images)
                errD_real = criterion_adv(output_real, label)
                errD_real.backward()

                # Generate fake high-res images from low-res version
                # (For simplicity, we use hr_images as dummy low-res input)
                fake_hr = netG(hr_images)
                label.fill_(fake_label)
                output_fake = netD(fake_hr.detach())
                errD_fake = criterion_adv(output_fake, label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizerD.step()

                # Update Generator with pixel and adversarial loss
                netG.zero_grad()
                label.fill_(real_label)
                output_fake = netD(fake_hr)
                loss_pixel = criterion_pixel(fake_hr, hr_images)
                loss_adv = criterion_adv(output_fake, label)
                errG = loss_pixel + 1e-3 * loss_adv
                errG.backward()
                optimizerG.step()

            G_losses.append(errG.item())
            D_losses.append(errD.item())

        print(f"Epoch [{epoch+1}/{epochs}] - Loss_G: {np.mean(G_losses):.4f}, Loss_D: {np.mean(D_losses):.4f}")

    # Save models
    model_name = f"{model_type}_{int(time.time())}.pth"
    torch.save(netG.state_dict(), os.path.join(MODELS_FOLDER, "generator_" + model_name))
    torch.save(netD.state_dict(), os.path.join(MODELS_FOLDER, "discriminator_" + model_name))
    return model_name, G_losses, D_losses

def generate_synthetic_images(model_name, model_type, num_images=5, latent_dim=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the generator based on the model type
    if model_type == "GAN":
        netG = Generator(latent_dim=latent_dim).to(device)
    elif model_type == "StyleGAN":
        netG = StyleGANGenerator(latent_dim=latent_dim).to(device)
    elif model_type == "StyleGAN2 ADA":
        netG = StyleGAN2ADAGenerator(latent_dim=latent_dim).to(device)
    else:
        flash("Generation only supported for GAN-based models.")
        return None

    # Load saved weights
    model_path = os.path.join(MODELS_FOLDER, "generator_" + model_name)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()

    output_folder = os.path.join(RESULTS_FOLDER, f"generated_{int(time.time())}")
    os.makedirs(output_folder, exist_ok=True)
    for i in range(num_images):
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        with torch.no_grad():
            fake = netG(noise)
        img_tensor = (fake.cpu().squeeze() + 1) / 2  # scale to [0, 1]
        np_img = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img = Image.fromarray(np_img)
        img.save(os.path.join(output_folder, f"generated_{i+1}.png"))
    return output_folder

# --------------------------
# Utility Functions
# --------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------------------
# Flask Routes
# --------------------------

@app.route('/')
def index():
    # List uploaded images (if any) and available models
    uploaded = os.listdir(UPLOAD_FOLDER)
    models = os.listdir(MODELS_FOLDER)
    return render_template('index.html', uploaded=uploaded, models=models)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file provided')
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)
            flash(f"Uploaded {filename}")
            return redirect(url_for('index'))
        else:
            flash("Invalid file type.")
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        model_type = request.form.get('model_type')
        epochs = int(request.form.get('epochs', 10))
        batch_size = int(request.form.get('batch_size', 8))
        lr = float(request.form.get('learning_rate', 0.0002))
        latent_dim = int(request.form.get('latent_dim', 100))
        upscale_factor = int(request.form.get('upscale_factor', 2))
        # For simplicity, we use the UPLOAD_FOLDER as the dataset folder
        dataset = UploadedImageDataset(UPLOAD_FOLDER)
        flash("Training started. Check the console for progress.")

        def training_task():
            model_name, G_losses, D_losses = train_model(model_type, dataset, epochs, batch_size, lr, latent_dim, upscale_factor)
            print("Training complete. Model saved as:", model_name)

        thread = threading.Thread(target=training_task)
        thread.daemon = True
        thread.start()
        return redirect(url_for('index'))
    return render_template('train.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        model_name = request.form.get('model_name')
        model_type = request.form.get('model_type')
        num_images = int(request.form.get('num_images', 5))
        latent_dim = int(request.form.get('latent_dim', 100))
        output_folder = generate_synthetic_images(model_name, model_type, num_images, latent_dim)
        if output_folder:
            flash("Image generation complete.")
            return redirect(url_for('view_generated', folder=os.path.basename(output_folder)))
        else:
            flash("Image generation failed.")
            return redirect(request.url)
    # List only GAN-based models from MODELS_FOLDER for generation
    available_models = [f.replace("generator_", "") for f in os.listdir(MODELS_FOLDER) if f.startswith("generator_")]
    return render_template('generate.html', models=available_models)

@app.route('/generated/<folder>')
def view_generated(folder):
    folder_path = os.path.join(RESULTS_FOLDER, folder)
    images = [f for f in os.listdir(folder_path) if allowed_file(f)]
    return render_template('view_generated.html', folder=folder, images=images)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/models/<filename>')
def model_file(filename):
    return send_from_directory(MODELS_FOLDER, filename)

# --------------------------
# Run the App
# --------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
