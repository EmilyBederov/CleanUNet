import os
import argparse
import json
from tqdm import tqdm
import glob
import random

import numpy as np
import torch
import torch.nn as nn

from scipy.io.wavfile import write as wavwrite
from scipy.io.wavfile import read as wavread
import soundfile as sf

from util import rescale, find_max_epoch, print_size, sampling
from network import CleanUNet

# Set random seeds for reproducibility
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def load_audio_file(file_path, target_sr=16000):
    """Load audio file and ensure it's at target sample rate"""
    try:
        # Try soundfile first (better format support)
        audio, sr = sf.read(file_path, dtype='float32')
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if necessary
        if sr != target_sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                print(f"Warning: librosa not available for resampling. Using original {sr}Hz")
        
        return audio, target_sr
        
    except Exception as e:
        # Fallback to scipy
        sr, audio = wavread(file_path)
        audio = audio.astype(np.float32) / 32768.0  # Convert to float32 [-1, 1]
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        return audio, sr


def denoise_custom_dataset(
    checkpoint_path, 
    config_path,
    test_set_path, 
    output_directory, 
    num_samples=3,
    ckpt_iter='500000'
):
    """
    Denoise audio using original CleanUNet method on your custom dataset
    
    Parameters:
    checkpoint_path (str):    Path to the checkpoint file
    config_path (str):        Path to config.json
    test_set_path (str):      Path to testing_set (contains clean/ and noisy/)
    output_directory (str):   Where to save enhanced audio
    num_samples (int):        Number of random samples to process
    ckpt_iter (str):          Checkpoint iteration identifier
    """
    
    print("="*60)
    print("CleanUNet Original Denoising (Authors' Method)")
    print("="*60)
    
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    network_config = config["network_config"]
    trainset_config = config.get("trainset_config", {})
    sample_rate = trainset_config.get("sample_rate", 16000)
    
    print(f"Using sample rate: {sample_rate}Hz")
    print(f"Network config: {network_config}")
    
    # Setup paths
    clean_dir = os.path.join(test_set_path, "clean")
    noisy_dir = os.path.join(test_set_path, "noisy")
    
    if not os.path.exists(clean_dir) or not os.path.exists(noisy_dir):
        print(f"Error: Could not find clean/ and noisy/ folders in {test_set_path}")
        return
    
    # Get all audio files and find matching pairs
    clean_files = glob.glob(os.path.join(clean_dir, "*.wav"))
    noisy_files = glob.glob(os.path.join(noisy_dir, "*.wav"))
    
    clean_basenames = {os.path.basename(f) for f in clean_files}
    noisy_basenames = {os.path.basename(f) for f in noisy_files}
    matching_files = list(clean_basenames & noisy_basenames)
    
    print(f"Found {len(clean_files)} clean files and {len(noisy_files)} noisy files")
    print(f"Found {len(matching_files)} matching pairs")
    
    if len(matching_files) == 0:
        print("Error: No matching file pairs found")
        return
    
    # Randomly select samples
    selected_files = random.sample(matching_files, min(num_samples, len(matching_files)))
    print(f"\nSelected {len(selected_files)} random samples:")
    for f in selected_files:
        print(f"  {f}")
    
    # Load and initialize model
    print(f"\nLoading CleanUNet model from {checkpoint_path}")
    net = CleanUNet(**network_config).cuda()
    print_size(net)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print("Model loaded successfully")
    
    # Create output directory
    output_dir = os.path.join(output_directory, f"original_denoising_{ckpt_iter}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Process each selected file
    all_results = []
    
    print(f"\nProcessing {len(selected_files)} files...")
    for i, filename in enumerate(tqdm(selected_files, desc="Denoising")):
        
        # Load clean and noisy audio
        clean_path = os.path.join(clean_dir, filename)
        noisy_path = os.path.join(noisy_dir, filename)
        
        try:
            clean_audio, _ = load_audio_file(clean_path, sample_rate)
            noisy_audio, _ = load_audio_file(noisy_path, sample_rate)
            
            print(f"\nProcessing {filename}")
            print(f"  Audio length: {len(noisy_audio)/sample_rate:.2f}s")
            print(f"  Clean RMS: {np.sqrt(np.mean(clean_audio**2)):.4f}")
            print(f"  Noisy RMS: {np.sqrt(np.mean(noisy_audio**2)):.4f}")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
        
        # Prepare noisy audio for inference (following authors' approach)
        noisy_tensor = torch.from_numpy(noisy_audio).unsqueeze(0).cuda()  # (1, L)
        LENGTH = len(noisy_audio)
        
        # Use authors' sampling function for inference
        print(f"  Running inference...")
        inference_start = torch.cuda.Event(enable_timing=True)
        inference_end = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            inference_start.record()
            generated_audio = sampling(net, noisy_tensor.unsqueeze(0))  # (1, 1, L)
            inference_end.record()
            torch.cuda.synchronize()
            
        inference_time = inference_start.elapsed_time(inference_end)  # milliseconds
        enhanced_audio = generated_audio[0].squeeze().cpu().numpy()
        
        print(f"  Inference time: {inference_time:.2f}ms")
        print(f"  Enhanced RMS: {np.sqrt(np.mean(enhanced_audio**2)):.4f}")
        
        # Save all three versions (like authors' format)
        base_name = filename.replace('.wav', '')
        
        # Save using authors' naming convention
        enhanced_filename = f"enhanced_{filename}"
        clean_filename = f"clean_{filename}"
        noisy_filename = f"noisy_{filename}"
        
        # Save as WAV files (authors use wavwrite)
        enhanced_path = os.path.join(output_dir, enhanced_filename)
        clean_path_out = os.path.join(output_dir, clean_filename)
        noisy_path_out = os.path.join(output_dir, noisy_filename)
        
        # Save using scipy (like authors)
        enhanced_int16 = (enhanced_audio * 32767).astype(np.int16)
        clean_int16 = (clean_audio * 32767).astype(np.int16)
        noisy_int16 = (noisy_audio * 32767).astype(np.int16)
        
        wavwrite(enhanced_path, sample_rate, enhanced_int16)
        wavwrite(clean_path_out, sample_rate, clean_int16)
        wavwrite(noisy_path_out, sample_rate, noisy_int16)
        
        # Calculate metrics
        # Ensure same length for comparison
        min_length = min(len(enhanced_audio), len(clean_audio), len(noisy_audio))
        enhanced_audio = enhanced_audio[:min_length]
        clean_audio = clean_audio[:min_length]
        noisy_audio = noisy_audio[:min_length]
        
        # RMS values
        noisy_rms = np.sqrt(np.mean(noisy_audio**2))
        clean_rms = np.sqrt(np.mean(clean_audio**2))
        enhanced_rms = np.sqrt(np.mean(enhanced_audio**2))
        
        # SNR calculation
        noise_original = noisy_audio - clean_audio
        noise_original_rms = np.sqrt(np.mean(noise_original**2))
        
        noise_enhanced = enhanced_audio - clean_audio
        noise_enhanced_rms = np.sqrt(np.mean(noise_enhanced**2))
        
        if noise_original_rms > 1e-6 and noise_enhanced_rms > 1e-6:
            snr_before = 20 * np.log10(clean_rms / noise_original_rms)
            snr_after = 20 * np.log10(clean_rms / noise_enhanced_rms)
            snr_improvement = snr_after - snr_before
        else:
            snr_before = snr_after = snr_improvement = 0.0
        
        # Store results
        result = {
            'filename': filename,
            'audio_length_sec': len(noisy_audio) / sample_rate,
            'inference_time_ms': inference_time,
            'noisy_rms': noisy_rms,
            'clean_rms': clean_rms,
            'enhanced_rms': enhanced_rms,
            'snr_before': snr_before,
            'snr_after': snr_after,
            'snr_improvement': snr_improvement
        }
        all_results.append(result)
        
        print(f"  Results saved to: {enhanced_filename}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("DENOISING RESULTS SUMMARY")
    print(f"{'='*60}")
    
    if all_results:
        avg_inference_time = np.mean([r['inference_time_ms'] for r in all_results])
        avg_snr_improvement = np.mean([r['snr_improvement'] for r in all_results])
        
        print(f"Processed {len(all_results)} files successfully")
        print(f"Average inference time: {avg_inference_time:.2f}ms")
        print(f"Average SNR improvement: {avg_snr_improvement:.2f}dB")
        print(f"\nDetailed results:")
        
        for result in all_results:
            print(f"  {result['filename']}:")
            print(f"    Length: {result['audio_length_sec']:.2f}s")
            print(f"    Inference: {result['inference_time_ms']:.2f}ms")
            print(f"    SNR: {result['snr_before']:.2f} -> {result['snr_after']:.2f}dB "
                  f"(+{result['snr_improvement']:.2f}dB)")
    
    print(f"\nAll files saved to: {output_dir}")
    print(f"Files follow authors' naming convention:")
    print(f"  enhanced_*.wav - Denoised audio")
    print(f"  clean_*.wav    - Ground truth")
    print(f"  noisy_*.wav    - Original noisy input")
    
    return all_results


if __name__ == "__main__":
    # Your paths
    checkpoint_path = os.path.expanduser("~/CleanUNet/exp/VOICEBANK-large-full/checkpoint/500000.pkl")
    config_path = os.path.expanduser("~/CleanUNet/configs/VOICEBANK-large-full.json")
    test_set_path = "/home/emilybederov/CleanUNet2/voicebank_dns_format/testing_set"
    output_directory = "./original_denoising_results"
    
    # Run original denoising on your data
    results = denoise_custom_dataset(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        test_set_path=test_set_path,
        output_directory=output_directory,
        num_samples=3,
        ckpt_iter='500000'
    )
