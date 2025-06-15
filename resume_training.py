import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import json
import os
import numpy as np
import random
from tqdm import tqdm
from models.cleanunet import CleanUNet  # Your CleanUNet model
from stft_loss import CleanUnetLoss, MultiResolutionSTFTLoss

class CosineAnnealingWithWarmup:
    """Learning rate scheduler that can resume from a checkpoint"""
    def __init__(self, optimizer, max_lr, total_steps, warmup_ratio=0.05, current_step=0):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.current_step = current_step  # Resume from this step
    
    def step(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.max_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1
        return lr

class WaveformDataset(Dataset):
    """Dataset for CleanUNet waveform training"""
    def __init__(self, csv_path, sample_rate=16000, crop_length_sec=5, add_noise=True):
        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.crop_length = int(crop_length_sec * sample_rate)
        self.add_noise = add_noise
    
    def _load_and_process_audio(self, path):
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(path)
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Normalize to [-1, 1]
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform.squeeze(0)
    
    def _random_crop(self, waveform):
        """Randomly crop audio to specified length"""
        if len(waveform) <= self.crop_length:
            pad_length = self.crop_length - len(waveform)
            waveform = F.pad(waveform, (0, pad_length), mode='constant', value=0)
        else:
            start_idx = random.randint(0, len(waveform) - self.crop_length)
            waveform = waveform[start_idx:start_idx + self.crop_length]
        
        return waveform
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        clean_waveform = self._load_and_process_audio(row['clean'])
        noisy_waveform = self._load_and_process_audio(row['noisy'])
        
        # Align lengths and crop
        min_len = min(len(clean_waveform), len(noisy_waveform))
        clean_waveform = clean_waveform[:min_len]
        noisy_waveform = noisy_waveform[:min_len]
        
        clean_waveform = self._random_crop(clean_waveform)
        noisy_waveform = self._random_crop(noisy_waveform)
        
        # Add channel dimension
        clean_waveform = clean_waveform.unsqueeze(0)
        noisy_waveform = noisy_waveform.unsqueeze(0)
        
        return noisy_waveform, clean_waveform
    
    def __len__(self):
        return len(self.df)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint and return training state"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Get starting iteration
    start_iteration = checkpoint.get('iteration', 0)
    
    print(f"Loaded checkpoint from iteration {start_iteration}")
    
    return start_iteration

def resume_cleanunet_training():
    """Resume CleanUNet training from checkpoint with smaller batch size"""
    
    parser = argparse.ArgumentParser(description='Resume CleanUNet training')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to model configuration')
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, required=True,
                       help='Path to validation CSV')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (reduced for limited resources)')
    parser.add_argument('--additional_iters', type=int, default=400000,
                       help='Additional iterations to train')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create datasets with smaller batch size considerations
    print("Loading datasets...")
    train_dataset = WaveformDataset(args.train_csv, crop_length_sec=5)  # Slightly smaller crops for memory
    val_dataset = WaveformDataset(args.val_csv, crop_length_sec=5, add_noise=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,  # Reduced batch size
        shuffle=True, 
        num_workers=2,  # Reduced workers to save memory
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    network_config = config.get("network_config", {})
    model = CleanUNet(**network_config).to(device)
    
    # Adjusted learning rate for smaller batch size
    # Rule of thumb: lr scales with sqrt(batch_size_ratio)
    original_batch_size = 32
    batch_size_ratio = args.batch_size / original_batch_size
    adjusted_lr = 2e-4 * np.sqrt(batch_size_ratio)  # Scale down LR
    
    print(f"Adjusted learning rate: {adjusted_lr:.2e} (original: 2e-4)")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=adjusted_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5
    )
    
    # Load checkpoint
    start_iteration = load_checkpoint(args.checkpoint_path, model, optimizer)
    
    # Create loss function
    mrstftloss = MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 512], 
        hop_sizes=[120, 240, 50], 
        win_lengths=[600, 1200, 240],
        window="hann_window",
        band="full"  # or "high" as in some paper configurations
    )
    
    criterion = CleanUnetLoss(
        ell_p=1,  # L1 loss
        ell_p_lambda=1.0,
        stft_lambda=1.0,
        mrstftloss=mrstftloss
    )
    
    # Create scheduler starting from current iteration
    total_iterations = start_iteration + args.additional_iters
    scheduler = CosineAnnealingWithWarmup(
        optimizer, 
        max_lr=adjusted_lr, 
        total_steps=total_iterations,
        warmup_ratio=0.05,
        current_step=start_iteration  # Resume from checkpoint iteration
    )
    
    # Training setup
    os.makedirs('checkpoints/cleanunet_resumed', exist_ok=True)
    
    checkpoint_interval = 10000  # Save more frequently
    validation_interval = 2000   # Validate more frequently
    
    # Training loop
    print(f"Resuming training from iteration {start_iteration} for {args.additional_iters} more iterations...")
    print(f"Target iteration: {total_iterations}")
    
    model.train()
    best_val_loss = float('inf')
    
    # Infinite data loader
    train_iter = iter(train_loader)
    
    progress_bar = tqdm(range(start_iteration, total_iterations), desc="Training")
    
    for iteration in progress_bar:
        try:
            noisy_waveform, clean_waveform = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            noisy_waveform, clean_waveform = next(train_iter)
        
        noisy_waveform = noisy_waveform.to(device)
        clean_waveform = clean_waveform.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        denoised_waveform = model(noisy_waveform)
        
        # Compute loss
        loss, loss_dict = criterion(clean_waveform, denoised_waveform)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (important for smaller batch sizes)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        current_lr = scheduler.step()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'LR': f'{current_lr:.2e}',
            'Iter': f'{iteration+1}/{total_iterations}'
        })
        
        # Validation
        if (iteration + 1) % validation_interval == 0:
            model.eval()
            val_loss = 0.0
            val_count = 0
            
            with torch.no_grad():
                for val_noisy, val_clean in val_loader:
                    val_noisy = val_noisy.to(device)
                    val_clean = val_clean.to(device)
                    
                    val_pred = model(val_noisy)
                    val_loss_batch, _ = criterion(val_clean, val_pred)
                    val_loss += val_loss_batch.item()
                    val_count += 1
                    
                    if val_count >= 25:  # Limit validation for speed
                        break
            
            avg_val_loss = val_loss / val_count
            print(f"\nIteration {iteration + 1}: Validation Loss = {avg_val_loss:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'iteration': iteration + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': config,
                    'batch_size': args.batch_size,
                    'lr': current_lr
                }, 'checkpoints/cleanunet_resumed/best_model.pkl')
                print(f"Saved new best model with validation loss: {avg_val_loss:.6f}")
            
            model.train()
        
        # Save checkpoint
        if (iteration + 1) % checkpoint_interval == 0:
            torch.save({
                'iteration': iteration + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'batch_size': args.batch_size,
                'lr': current_lr
            }, f'checkpoints/cleanunet_resumed/checkpoint_iter_{iteration + 1}.pkl')
            print(f"\nSaved checkpoint at iteration {iteration + 1}")
    
    # Save final model
    torch.save({
        'iteration': total_iterations,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'batch_size': args.batch_size,
        'lr': current_lr
    }, 'checkpoints/cleanunet_resumed/final_model.pkl')
    
    print("Training completed!")
    print(f"Final iteration: {total_iterations}")

if __name__ == '__main__':
    resume_cleanunet_training()