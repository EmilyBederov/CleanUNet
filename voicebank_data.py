from datasets import load_dataset
import os
from scipy.io.wavfile import write as wavwrite

# Load the dataset
ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

# Create DNS-style directory structure
os.makedirs("voicebank_dns_format/training_set/clean", exist_ok=True)
os.makedirs("voicebank_dns_format/training_set/noisy", exist_ok=True)
os.makedirs("voicebank_dns_format/datasets/test_set/synthetic/no_reverb/clean", exist_ok=True)
os.makedirs("voicebank_dns_format/datasets/test_set/synthetic/no_reverb/noisy", exist_ok=True)

# Save training data
for i, example in enumerate(ds['train']):
    # Save clean audio
    wavwrite(f"voicebank_dns_format/training_set/clean/fileid_{i}.wav", 
             16000, example['clean']['array'])
    # Save noisy audio  
    wavwrite(f"voicebank_dns_format/training_set/noisy/fileid_{i}.wav", 
             16000, example['noisy']['array'])

# Save test data
for i, example in enumerate(ds['test']):
    # Save clean audio
    wavwrite(f"voicebank_dns_format/datasets/test_set/synthetic/no_reverb/clean/clean_fileid_{i}.wav", 
             16000, example['clean']['array'])
    # Save noisy audio
    wavwrite(f"voicebank_dns_format/datasets/test_set/synthetic/no_reverb/noisy/noisy_fileid_{i}.wav", 
             16000, example['noisy']['array'])

print("Dataset saved in DNS format!")