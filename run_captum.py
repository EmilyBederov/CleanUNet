from captum_cleanunet_analyzer import analyze_cleanunet_with_captum
from network import CleanUNet
import json
import torch
import os

# Load config exactly like denoise.py
with open('configs/wind-denoise.json') as f:
    data = f.read()
config = json.loads(data)

network_config = config["network_config"]
train_config = config["train_config"]

# Create model exactly like denoise.py
net = CleanUNet(**network_config).cuda()

# Load checkpoint exactly like denoise.py
exp_path = train_config["exp_path"]  # This gets "DNS-large-full"
ckpt_directory = os.path.join(train_config["log"]["directory"], exp_path, 'checkpoint')
ckpt_iter = 'pretrained'
model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))

print(f"Loading model from: {model_path}")

checkpoint = torch.load(model_path, map_location='cpu')
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

print("Model loaded successfully!")

# Now run Captum analysis
results, attributions = analyze_cleanunet_with_captum(
    model=net,  # Use 'net' like in denoise.py
    clean_path="../data/clean_data/Sample32.wav",
    noisy_path="../data/distorted_signals/mixed_Sample100.wav", 
    windy_path="../data/wind_samples/WinD131.wav"
)


