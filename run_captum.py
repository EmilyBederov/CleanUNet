from captum_cleanunet_analyzer import analyze_cleanunet_with_captum
from network import CleanUNet
import json
import torch


# Load your model (using your existing setup)
with open('configs/DNS-large-full.json') as f:
    config = json.load(f)

model = CleanUNet(**config['network_config'])
checkpoint = torch.load('/Users/emilybederov/github/CleanUNet/exp/DNS-large-full/checkpoint/pretrained.pkl', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.cuda().eval()

# Run analysis with your audio files
results, attributions = analyze_cleanunet_with_captum(
    model=model,
    clean_path="../data/clean_data/Sample32.wav",
    noisy_path="../data/distorted_signals/mixed_Sample100.wav", 
    windy_path="../data/wind_samples/WinD131.wav"
)


