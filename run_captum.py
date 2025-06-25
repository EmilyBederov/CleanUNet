# from working_activation_analyzer import analyze_cleanunet_working
# from network import CleanUNet
# import json
# import torch
# import os

# # Load model (same as before)
# with open('configs/wind-denoise.json') as f:
#     data = f.read()
# config = json.loads(data)

# network_config = config["network_config"]
# train_config = config["train_config"]

# net = CleanUNet(**network_config).cuda()

# exp_path = train_config["exp_path"]
# ckpt_directory = os.path.join(train_config["log"]["directory"], exp_path, 'checkpoint')
# model_path = os.path.join(ckpt_directory, 'pretrained.pkl')

# checkpoint = torch.load(model_path, map_location='cpu')
# net.load_state_dict(checkpoint['model_state_dict'])
# net.eval()

# print("✅ Model loaded successfully!")

# # Run analysis with your audio files (UPDATE THESE PATHS)
# results = analyze_cleanunet_working(
#     model=net,  # Use 'net' like in denoise.py
#     clean_path="../data/clean_samples/Sample32.wav",
#     noisy_path="../data/distorted_signals/mixed_Sample100.wav", 
#     windy_path="../data/wind_samples/WinD131.wav"
# )

from working_activation_analyzer import WorkingCleanUNetAnalyzer, load_audio_sample
from network import CleanUNet
import json
import torch
import os
import glob

# Load your model (same as before)
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

# Analyze multiple windy samples from directory
windy_dir = "../data/wind_samples"  # UPDATE THIS PATH

# Find all wav files
windy_files = glob.glob(os.path.join(windy_dir, "*.wav"))

print(f"Found {len(windy_files)} windy samples to analyze...")

# Initialize analyzer
analyzer = WorkingCleanUNetAnalyzer(net)

# Analyze each windy sample
all_windy_results = []

for i, windy_file in enumerate(windy_files[:10]):  # Analyze first 10 files
    print(f"\nAnalyzing windy sample {i+1}: {os.path.basename(windy_file)}")
    
    try:
        windy_audio = load_audio_sample(windy_file)
        result = analyzer.analyze_sample(windy_audio, f"windy_{i}")
        all_windy_results.append(result)
    except Exception as e:
        print(f"Error with {windy_file}: {e}")

# Print summary of which layers are most active across all windy samples
if all_windy_results:
    print(f"\n SUMMARY ACROSS {len(all_windy_results)} WINDY SAMPLES:")
    print("="*50)
    
    # Average activity scores across all samples
    layer_averages = {}
    for result in all_windy_results:
        for layer_name, stats in result['layer_stats'].items():
            if layer_name not in layer_averages:
                layer_averages[layer_name] = []
            layer_averages[layer_name].append(stats['activity_score'])
    
    # Compute and sort averages
    for layer_name in layer_averages:
        layer_averages[layer_name] = sum(layer_averages[layer_name]) / len(layer_averages[layer_name])
    
    sorted_layers = sorted(layer_averages.items(), key=lambda x: x[1], reverse=True)
    
    print("Most active layers for wind noise:")
    for layer_name, avg_activity in sorted_layers[:8]:
        print(f"  {layer_name}: {avg_activity:.4f}")

# Clean up
analyzer.cleanup()
