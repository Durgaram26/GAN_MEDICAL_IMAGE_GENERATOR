import os
import time
import io
import json
import base64
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from flask import Flask, request, redirect, url_for, flash, render_template, send_from_directory, jsonify
import matplotlib.pyplot as plt
import zipfile
from datetime import datetime

# Create necessary folders
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("generated_images", exist_ok=True)

# File to store trained models metadata
TRAINED_MODELS_FILE = "models/trained_models.json"

app = Flask(__name__)
app.secret_key = "your_secret_key_here"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config['MAX_CONTENT_LENGTH'] = 3500 * 1080 * 1080  # ~1000 MB

# Global state for uploaded images and trained models
uploaded_images = []
trained_models = {}
training_progress = {}

##############################################
# Utility Modules for Improved GAN
##############################################

class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(EqualizedLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.scale = math.sqrt(2 / in_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, input):
        return F.linear(input, self.weight * self.scale, self.bias)

class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(EqualizedConv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = math.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, stride=self.stride, padding=self.padding)

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True, stride=1, padding=1):
        super(ModulatedConv2d, self).__init__()
        self.eps = 1e-8
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.demodulate = demodulate
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.modulation = EqualizedLinear(style_dim, in_channels)

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt((weight ** 2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)
        weight = weight.view(batch * self.out_channels, in_channel, self.kernel_size, self.kernel_size)
        x = x.view(1, batch * in_channel, height, width)
        out = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=batch)
        out = out.view(batch, self.out_channels, out.shape[-2], out.shape[-1])
        return out

class StyledConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim):
        super(StyledConvBlock, self).__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size, style_dim, padding=kernel_size//2)
        self.noise = NoiseInjection(out_channels)
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise=None):
        out = self.conv(x, style)
        out = self.noise(out, noise)
        return self.activate(out)

##############################################
# Model Definitions: Generators & Discriminators
##############################################

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
            # Use a 2x2 kernel to collapse 2x2 input to 1x1 output
            nn.Conv2d(feature_maps * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.net(input).view(-1, 1).squeeze(1)

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

class ImprovedStyleGAN2Generator(nn.Module):
    def __init__(self, latent_dim=100, style_dim=100, channels=3, base_channels=64, resolution=64):
        super(ImprovedStyleGAN2Generator, self).__init__()
        self.resolution = resolution
        self.style_dim = style_dim
        self.mapping = nn.Sequential(
            EqualizedLinear(latent_dim, style_dim),
            nn.LeakyReLU(0.2),
            EqualizedLinear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            EqualizedLinear(style_dim, style_dim)
        )
        self.constant = nn.Parameter(torch.randn(1, base_channels * 8, 4, 4))
        self.conv1 = StyledConvBlock(base_channels * 8, base_channels * 8, 3, style_dim)
        self.to_rgb1 = ModulatedConv2d(base_channels * 8, channels, 1, style_dim, demodulate=False, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = StyledConvBlock(base_channels * 8, base_channels * 4, 3, style_dim)
        self.to_rgb2 = ModulatedConv2d(base_channels * 4, channels, 1, style_dim, demodulate=False, padding=0)
        self.conv3 = StyledConvBlock(base_channels * 4, base_channels * 2, 3, style_dim)
        self.to_rgb3 = ModulatedConv2d(base_channels * 2, channels, 1, style_dim, demodulate=False, padding=0)
        self.conv4 = StyledConvBlock(base_channels * 2, base_channels, 3, style_dim)
        self.to_rgb4 = ModulatedConv2d(base_channels, channels, 1, style_dim, demodulate=False, padding=0)

    def forward(self, z):
        style = self.mapping(z.view(z.size(0), -1))
        batch = z.size(0)
        out = self.constant.repeat(batch, 1, 1, 1)
        out = self.conv1(out, style)
        rgb = self.to_rgb1(out, style)
        out = self.upsample(out)
        out = self.conv2(out, style)
        rgb = self.to_rgb2(out, style)
        out = self.upsample(out)
        out = self.conv3(out, style)
        rgb = self.to_rgb3(out, style)
        out = self.upsample(out)
        out = self.conv4(out, style)
        rgb = self.to_rgb4(out, style)
        return torch.tanh(rgb)

# Super-resolution models (ESRGAN/SRGAN)
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

class SRGANGenerator(ESRGANGenerator):
    pass

class SRGANDiscriminator(ESRGANDiscriminator):
    pass

##############################################
# Dataset Definitions
##############################################

class UploadedImageDataset(Dataset):
    def __init__(self, image_paths, base_dir=app.config["UPLOAD_FOLDER"], transform=None):
        self.image_paths = image_paths
        self.base_dir = base_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        full_path = os.path.join(self.base_dir, self.image_paths[idx])
        image = Image.open(full_path).convert("RGB")
        return self.transform(image)

class SuperResDataset(Dataset):
    def __init__(self, image_paths, upscale_factor=2, hr_size=(64, 64)):
        self.image_paths = image_paths
        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((hr_size[0]//upscale_factor, hr_size[1]//upscale_factor)),
            transforms.Resize(hr_size, interpolation=Image.BICUBIC)
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        full_path = os.path.join(app.config["UPLOAD_FOLDER"], self.image_paths[idx])
        hr_image = Image.open(full_path).convert("RGB")
        hr_image = self.hr_transform(hr_image)
        lr_image = transforms.ToPILImage()(hr_image)
        lr_image = self.lr_transform(lr_image)
        lr_image = transforms.ToTensor()(lr_image)
        lr_image = (lr_image - 0.5) / 0.5
        return lr_image, hr_image

##############################################
# Utility Functions
##############################################

def default_upscale_image(image, scale_factor=2):
    width, height = image.size
    new_size = (width * scale_factor, height * scale_factor)
    return image.resize(new_size, Image.LANCZOS)

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

##############################################
# Training Functions (Without Early Stopping)
##############################################

def train_model_no_early_stopping(model_type, image_paths, epochs, batch_size, lr, latent_dim=100, upscale_factor=2,
                                  feature_maps=64, loss_function="BCE"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_losses, D_losses = [], []
    global training_progress
    training_progress = {"epoch": 0, "total_epochs": epochs, "batch": 0, "total_batches": 0, "loss_D": 0, "loss_G": 0}

    if model_type in ["GAN", "StyleGAN", "StyleGAN2 ADA"]:
        dataset = UploadedImageDataset(image_paths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if model_type == "GAN":
            netG = Generator(latent_dim=latent_dim, feature_maps=feature_maps).to(device)
        elif model_type == "StyleGAN":
            netG = StyleGANGenerator(latent_dim=latent_dim, feature_maps=feature_maps).to(device)
        else:
            netG = ImprovedStyleGAN2Generator(latent_dim=latent_dim, style_dim=latent_dim, channels=3,
                                               base_channels=feature_maps, resolution=64).to(device)
        netD = Discriminator(feature_maps=feature_maps).to(device)

        if loss_function == "BCE":
            criterion = nn.BCELoss()
        elif loss_function == "Wasserstein":
            def wasserstein_loss(output, target):
                return -torch.mean(output * target)
            criterion = wasserstein_loss
        else:
            criterion = nn.BCELoss()

        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
        real_label = 1.0
        fake_label = 0.0

        for epoch in range(epochs):
            training_progress["epoch"] = epoch + 1
            for i, data in enumerate(dataloader, 0):
                training_progress["batch"] = i + 1
                training_progress["total_batches"] = len(dataloader)
                netD.zero_grad()
                real_cpu = data.to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                output = netD(real_cpu)
                errD_real = criterion(output, label)
                errD_real.backward()

                noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizerD.step()

                netG.zero_grad()
                label.fill_(real_label)
                output = netD(fake)
                errG = criterion(output, label)
                errG.backward()
                optimizerG.step()

                G_losses.append(errG.item())
                D_losses.append(errD.item())
                training_progress["loss_D"] = errD.item()
                training_progress["loss_G"] = errG.item()
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{len(dataloader)}] | Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f}")

    elif model_type in ["ESRGAN", "SRGAN"]:
        sr_dataset = SuperResDataset(image_paths, upscale_factor=upscale_factor, hr_size=(64,64))
        dataloader = DataLoader(sr_dataset, batch_size=batch_size, shuffle=True)
        if model_type == "ESRGAN":
            netG = ESRGANGenerator(upscale_factor=upscale_factor).to(device)
            netD = ESRGANDiscriminator().to(device)
        else:
            netG = SRGANGenerator(upscale_factor=upscale_factor).to(device)
            netD = SRGANDiscriminator().to(device)
        
        criterion_pixel = nn.L1Loss()
        criterion_adv = nn.BCELoss()
        optimizerG = optim.Adam(netG.parameters(), lr=lr)
        optimizerD = optim.Adam(netD.parameters(), lr=lr)
        real_label = 1.0
        fake_label = 0.0

        for epoch in range(epochs):
            training_progress["epoch"] = epoch + 1
            for i, (lr_imgs, hr_imgs) in enumerate(dataloader, 0):
                training_progress["batch"] = i + 1
                training_progress["total_batches"] = len(dataloader)
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                b_size = lr_imgs.size(0)
                netD.zero_grad()
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                output_real = netD(hr_imgs)
                errD_real = criterion_adv(output_real, label)
                errD_real.backward()

                fake_hr = netG(lr_imgs)
                label.fill_(fake_label)
                output_fake = netD(fake_hr.detach())
                errD_fake = criterion_adv(output_fake, label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizerD.step()

                netG.zero_grad()
                label.fill_(real_label)
                output_fake = netD(fake_hr)
                loss_pixel = criterion_pixel(fake_hr, hr_imgs)
                loss_adv = criterion_adv(output_fake, label)
                errG = loss_pixel + 1e-3 * loss_adv
                errG.backward()
                optimizerG.step()

                G_losses.append(errG.item())
                D_losses.append(errD.item())
                training_progress["loss_D"] = errD.item()
                training_progress["loss_G"] = errG.item()
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{len(dataloader)}] | Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f}")
    else:
        print("Unknown model type.")
        return None, None, None

    return netG, G_losses, D_losses

##############################################
# Generation & Analysis Functions
##############################################

def generate_synthetic_images(netG, latent_dim, num_images=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG.to(device)
    netG.eval()
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        fake_images = netG(noise).cpu()
    images = []
    for img_tensor in fake_images:
        img_tensor = (img_tensor + 1) / 2
        img_np = img_tensor.permute(1, 2, 0).numpy() * 255
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        images.append(img_pil)
    return images

def analyze_model(G_losses, D_losses):
    analysis = {
        "Final Generator Loss": G_losses[-1] if G_losses else None,
        "Final Discriminator Loss": D_losses[-1] if D_losses else None,
        "Average Generator Loss": np.mean(G_losses) if G_losses else None,
        "Average Discriminator Loss": np.mean(D_losses) if D_losses else None,
    }
    if len(G_losses) > 1:
        improvement = (G_losses[0] - G_losses[-1]) / G_losses[0] * 100
        analysis["Generator Improvement (%)"] = improvement
    else:
        analysis["Generator Improvement (%)"] = 0
    return analysis

##############################################
# Persistence Functions for Trained Models
##############################################

def get_generator_instance(model_type, latent_dim):
    if model_type == "GAN":
        return Generator(latent_dim=latent_dim)
    elif model_type == "StyleGAN":
        return StyleGANGenerator(latent_dim=latent_dim)
    elif model_type == "StyleGAN2 ADA":
        return ImprovedStyleGAN2Generator(latent_dim=latent_dim, style_dim=latent_dim, channels=3, base_channels=64, resolution=64)
    elif model_type == "ESRGAN":
        return ESRGANGenerator()
    elif model_type == "SRGAN":
        return SRGANGenerator()
    else:
        return None

def save_trained_models():
    metadata = {}
    for name, data in trained_models.items():
        metadata[name] = {
            "model_type": data["model_type"],
            "latent_dim": data["latent_dim"],
            "G_losses": data["G_losses"],
            "D_losses": data["D_losses"],
            "generator_path": data["generator_path"]
        }
    with open(TRAINED_MODELS_FILE, "w") as f:
        json.dump(metadata, f)

def load_trained_models():
    if os.path.exists(TRAINED_MODELS_FILE):
        with open(TRAINED_MODELS_FILE, "r") as f:
            metadata = json.load(f)
        for name, data in metadata.items():
            generator = get_generator_instance(data["model_type"], data["latent_dim"])
            if generator:
                generator_path = data["generator_path"]
                if os.path.exists(generator_path):
                    state = torch.load(generator_path, map_location=torch.device("cuda"))
                    generator.load_state_dict(state)
                    trained_models[name] = {
                        "model_type": data["model_type"],
                        "latent_dim": data["latent_dim"],
                        "G_losses": data["G_losses"],
                        "D_losses": data["D_losses"],
                        "generator_path": generator_path,
                        "netG": generator
                    }
                    print(f"Loaded trained model: {name}")
                else:
                    print(f"Generator file not found for model {name}")

load_trained_models()

##############################################
# Routes
##############################################

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data_manager", methods=["GET", "POST"])
def data_manager():
    global uploaded_images
    if request.method == "POST":
        files = request.files.getlist("files")
        batch_size = 100
        total_uploaded = 0

        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            for file in batch:
                if file and file.filename != "":
                    filename = file.filename
                    if filename.lower().endswith(".zip"):
                        zip_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                        file.save(zip_path)
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            extract_dir = os.path.join(app.config["UPLOAD_FOLDER"], os.path.splitext(filename)[0])
                            os.makedirs(extract_dir, exist_ok=True)
                            zip_ref.extractall(extract_dir)
                        os.remove(zip_path)
                        for root, _, files_in_zip in os.walk(extract_dir):
                            for f in files_in_zip:
                                rel_path = os.path.relpath(os.path.join(root, f), app.config["UPLOAD_FOLDER"])
                                rel_path = rel_path.replace("\\", "/")
                                if rel_path not in uploaded_images:
                                    uploaded_images.append(rel_path)
                                    total_uploaded += 1
                    else:
                        subdirectory = os.path.dirname(filename)
                        save_dir = os.path.join(app.config["UPLOAD_FOLDER"], subdirectory)
                        os.makedirs(save_dir, exist_ok=True)
                        filepath = os.path.join(save_dir, os.path.basename(filename))
                        file.save(filepath)
                        relative_path = os.path.relpath(filepath, app.config["UPLOAD_FOLDER"])
                        relative_path = relative_path.replace("\\", "/")
                        if relative_path not in uploaded_images:
                            uploaded_images.append(relative_path)
                            total_uploaded += 1

            flash(f"Batch {i//batch_size + 1}: Uploaded {len(batch)} files successfully.")
        flash(f"Total uploaded files: {total_uploaded}")
        return redirect(url_for("data_manager"))

    image_files = uploaded_images[:5]
    return render_template("data_manager.html", images=image_files)

@app.route("/train_model", methods=["GET", "POST"])
def train_model_route():
    global trained_models, uploaded_images
    if request.method == "POST":
        custom_name = request.form.get("custom_name")
        model_type = request.form.get("model_type")
        epochs = int(request.form.get("epochs"))
        batch_size = int(request.form.get("batch_size"))
        lr = float(request.form.get("lr"))
        latent_dim = int(request.form.get("latent_dim"))
        upscale_factor = int(request.form.get("upscale_factor"))
        loss_function = request.form.get("loss_function")
        feature_maps = int(request.form.get("feature_maps"))
        if not uploaded_images:
            flash("No images uploaded. Please use the Data Manager first.")
            return redirect(url_for("train_model_route"))
        netG, G_losses, D_losses = train_model_no_early_stopping(
            model_type, uploaded_images, epochs, batch_size, lr,
            latent_dim, upscale_factor, feature_maps, loss_function
        )
        if netG:
            generator_filename = os.path.join("models", f"generator_{custom_name}.pth")
            torch.save(netG.state_dict(), generator_filename)
            trained_models[custom_name] = {
                "model_type": model_type,
                "latent_dim": latent_dim,
                "G_losses": G_losses,
                "D_losses": D_losses,
                "generator_path": generator_filename,
                "netG": netG
            }
            save_trained_models()
            flash(f"Training completed for {custom_name} ({model_type}).")
        return redirect(url_for("train_model_route"))
    return render_template("train_model.html", models=trained_models)

@app.route("/progress")
def progress():
    return jsonify(training_progress)

@app.route("/generate_images", methods=["GET", "POST"])
def generate_images():
    generated_images = None
    zip_download_link = None
    if request.method == "POST":
        model_name = request.form.get("model_name")
        num_images = int(request.form.get("num_images"))
        confirm_save = request.form.get("confirm_save")
        if model_name not in trained_models:
            flash("Selected model not found.")
            return redirect(url_for("generate_images"))
        model_data = trained_models[model_name]
        netG = model_data["netG"]
        latent_dim = model_data["latent_dim"]
        imgs = generate_synthetic_images(netG, latent_dim, num_images)
        generated_images = [pil_to_base64(img) for img in imgs]
        if confirm_save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_folder = os.path.join("generated_images", f"{model_name}_{timestamp}")
            os.makedirs(save_folder, exist_ok=True)
            for idx, img in enumerate(imgs):
                img.save(os.path.join(save_folder, f"generated_{idx+1}.png"))
            zip_filename = f"{model_name}_{timestamp}.zip"
            zip_filepath = os.path.join("generated_images", zip_filename)
            with zipfile.ZipFile(zip_filepath, "w") as zipf:
                for root, _, files in os.walk(save_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, arcname=os.path.relpath(file_path, save_folder))
            zip_download_link = url_for("static_download", filename=zip_filename)
            flash("Generated images saved and zipped successfully.")
    return render_template("generate_images.html", models=trained_models, generated_images=generated_images, zip_link=zip_download_link)

@app.route("/download/<path:filename>")
def static_download(filename):
    return send_from_directory("generated_images", filename, as_attachment=True)

@app.route("/upscale", methods=["GET", "POST"])
def upscale():
    result_image = None
    if request.method == "POST":
        if "up_image" not in request.files:
            flash("No image uploaded")
            return redirect(request.url)
        file = request.files["up_image"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        image = Image.open(file).convert("RGB")
        scale_factor = int(request.form.get("scale_factor"))
        model_name = request.form.get("model_name")
        if model_name and model_name in trained_models and trained_models[model_name]["model_type"] in ["ESRGAN", "SRGAN"]:
            model_data = trained_models[model_name]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            netG = model_data["netG"].to(device)
            netG.eval()
            preprocess = transforms.Compose([
                transforms.Resize((64//scale_factor, 64//scale_factor)),
                transforms.Resize((64, 64), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            input_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output_tensor = netG(input_tensor).cpu()
            output_tensor = (output_tensor + 1) / 2
            img_np = output_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255
            result = Image.fromarray(img_np.astype(np.uint8))
            result_image = pil_to_base64(result)
        else:
            result = default_upscale_image(image, scale_factor)
            result_image = pil_to_base64(result)
    return render_template("upscale.html", models=trained_models, result_image=result_image)

@app.route("/model_analysis", methods=["GET", "POST"])
def model_analysis():
    analysis = None
    graph_base64 = None
    if request.method == "POST":
        model_name = request.form.get("model_name")
        if model_name not in trained_models:
            flash("Selected model not found.")
            return redirect(url_for("model_analysis"))
        model_data = trained_models[model_name]
        analysis = analyze_model(model_data["G_losses"], model_data["D_losses"])
        plt.figure(figsize=(6,4))
        plt.plot(model_data["G_losses"], label="Generator Loss")
        plt.plot(model_data["D_losses"], label="Discriminator Loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()
    return render_template("model_analysis.html", models=trained_models, analysis=analysis, graph=graph_base64)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
