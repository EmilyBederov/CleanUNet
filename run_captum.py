from working_activation_analyzer import analyze_cleanunet_working
from network import CleanUNet
import json
import torch
import os

# Load model (same as before)
with open('configs/wind-denoise.json') as f:
    data = f.read()
config = json.loads(data)

network_config = config["network_config"]
train_config = config["train_config"]

net = CleanUNet(**network_config).cuda()

exp_path = train_config["exp_path"]
ckpt_directory = os.path.join(train_config["log"]["directory"], exp_path, 'checkpoint')
model_path = os.path.join(ckpt_directory, 'pretrained.pkl')

checkpoint = torch.load(model_path, map_location='cpu')
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

print("✅ Model loaded successfully!")

# Run analysis with your audio files (UPDATE THESE PATHS)
results = analyze_cleanunet_simple(
    model=net,  # Use 'net' like in denoise.py
    clean_path="../data/clean_samples/Sample32.wav",
    noisy_path="../data/distorted_signals/mixed_Sample100.wav", 
    windy_path="../data/wind_samples/WinD131.wav"
)

