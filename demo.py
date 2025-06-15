import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import threading
import time
import os
from typing import Optional, Callable

class CircularBuffer:
    """Efficient circular buffer for audio streaming"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = np.zeros(max_size, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        self.lock = threading.Lock()
    
    def write(self, data: np.ndarray):
        """Write data to buffer"""
        with self.lock:
            data_len = len(data)
            
            # Handle wraparound
            if self.write_pos + data_len <= self.max_size:
                self.buffer[self.write_pos:self.write_pos + data_len] = data
            else:
                # Split write across buffer boundary
                first_part = self.max_size - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:data_len - first_part] = data[first_part:]
            
            self.write_pos = (self.write_pos + data_len) % self.max_size
            self.size = min(self.size + data_len, self.max_size)
    
    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read data from buffer"""
        with self.lock:
            if self.size < num_samples:
                return None
            
            data = np.zeros(num_samples, dtype=np.float32)
            
            # Handle wraparound
            if self.read_pos + num_samples <= self.max_size:
                data = self.buffer[self.read_pos:self.read_pos + num_samples].copy()
            else:
                # Split read across buffer boundary
                first_part = self.max_size - self.read_pos
                data[:first_part] = self.buffer[self.read_pos:]
                data[first_part:] = self.buffer[:num_samples - first_part]
            
            self.read_pos = (self.read_pos + num_samples) % self.max_size
            self.size -= num_samples
            
            return data
    
    def peek(self, num_samples: int, offset: int = 0) -> Optional[np.ndarray]:
        """Peek at data without consuming it"""
        with self.lock:
            if self.size < num_samples + offset:
                return None
            
            start_pos = (self.read_pos + offset) % self.max_size
            data = np.zeros(num_samples, dtype=np.float32)
            
            if start_pos + num_samples <= self.max_size:
                data = self.buffer[start_pos:start_pos + num_samples].copy()
            else:
                first_part = self.max_size - start_pos
                data[:first_part] = self.buffer[start_pos:]
                data[first_part:] = self.buffer[:num_samples - first_part]
            
            return data


def padding(x, D, K, S):
    """CleanUNet's exact padding function for proper audio length"""
    L = x.shape[-1]
    for _ in range(D):
        if L < K:
            L = 1
        else:
            L = 1 + np.ceil((L - K) / S)

    for _ in range(D):
        L = (L - 1) * S + K
    
    L = int(L)
    x = F.pad(x, (0, L - x.shape[-1]))
    return x


class CleanUNetStreaming:
    """
    Streaming inference wrapper for CleanUNet with exact model compatibility.
    
    This implementation matches the exact preprocessing, normalization, and 
    inference pipeline used in your CleanUNet training.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        sample_rate: int = 16000,
        chunk_size: int = 2048,  # Larger chunks work better with CleanUNet
        overlap_size: int = 512,  # 25% overlap
        buffer_size_seconds: float = 3.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_causal_attention: bool = True  # CleanUNet supports causal attention
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.use_causal_attention = use_causal_attention
        
        # Get model architecture parameters for padding calculation
        self.encoder_n_layers = getattr(model, 'encoder_n_layers', 8)
        self.kernel_size = getattr(model, 'kernel_size', 4)
        self.stride = getattr(model, 'stride', 2)
        
        # Calculate buffer size
        buffer_size = int(buffer_size_seconds * sample_rate)
        self.input_buffer = CircularBuffer(buffer_size)
        self.output_buffer = CircularBuffer(buffer_size)
        
        # For overlap-add processing
        self.overlap_buffer = np.zeros(overlap_size, dtype=np.float32)
        
        # For std calculation across chunks (essential for CleanUNet)
        self.std_window_size = chunk_size * 2  # Use larger window for stable std
        self.std_buffer = CircularBuffer(self.std_window_size)
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        
        # Callbacks
        self.audio_callback: Optional[Callable] = None
        
        # Statistics for monitoring
        self.stats = {
            'chunks_processed': 0,
            'avg_std': 0.0,
            'processing_time_ms': 0.0
        }
    
    def set_audio_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback function for processed audio output"""
        self.audio_callback = callback
    
    def add_audio(self, audio_data: np.ndarray):
        """Add new audio data to input buffer"""
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Add to both input buffer and std calculation buffer
        self.input_buffer.write(audio_data)
        self.std_buffer.write(audio_data)
    
    def calculate_streaming_std(self, audio_chunk: np.ndarray) -> float:
        """
        Calculate standard deviation for normalization.
        Uses a larger context window for stable std estimation.
        """
        # Get more context for std calculation if available
        context_data = self.std_buffer.peek(self.std_window_size)
        
        if context_data is not None and len(context_data) >= self.chunk_size:
            # Use the larger context for std calculation
            std = np.std(context_data) + 1e-3
        else:
            # Fallback to chunk-only std
            std = np.std(audio_chunk) + 1e-3
        
        # Update stats
        self.stats['avg_std'] = 0.95 * self.stats['avg_std'] + 0.05 * std
        
        return std
    
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process a single audio chunk through CleanUNet with exact model preprocessing"""
        start_time = time.time()
        
        with torch.no_grad():
            # Calculate std for normalization (CleanUNet's exact approach)
            std = self.calculate_streaming_std(audio_chunk)
            
            # Normalize (CleanUNet's exact normalization)
            normalized_audio = audio_chunk / std
            
            # Convert to tensor and add batch/channel dimensions
            # CleanUNet expects (B, C, L) format
            audio_tensor = torch.from_numpy(normalized_audio).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Apply CleanUNet's padding
            padded_audio = padding(audio_tensor, self.encoder_n_layers, self.kernel_size, self.stride)
            
            # Get original length for output cropping
            original_length = audio_tensor.shape[-1]
            
            # Process through model
            enhanced = self.model(padded_audio)
            
            # Crop to original length and denormalize (CleanUNet's exact approach)
            enhanced = enhanced[:, :, :original_length] * std
            
            # Convert back to numpy
            enhanced_audio = enhanced.squeeze(0).squeeze(0).cpu().numpy()
            
            # Update processing time stats
            processing_time = (time.time() - start_time) * 1000
            self.stats['processing_time_ms'] = 0.9 * self.stats['processing_time_ms'] + 0.1 * processing_time
            
            return enhanced_audio
    
    def apply_overlap_add(self, chunk: np.ndarray) -> np.ndarray:
        """Apply overlap-add processing to maintain continuity"""
        if len(chunk) < self.overlap_size:
            return chunk
        
        output = chunk.copy()
        
        if self.overlap_size > 0 and len(self.overlap_buffer) > 0:
            # Apply triangular window for smooth overlap-add
            overlap_len = min(len(self.overlap_buffer), len(output), self.overlap_size)
            
            # Create triangular fade windows
            fade_out = np.linspace(1, 0, overlap_len)
            fade_in = np.linspace(0, 1, overlap_len)
            
            # Apply overlap-add with windowing
            output[:overlap_len] = (
                output[:overlap_len] * fade_in + 
                self.overlap_buffer[:overlap_len] * fade_out
            )
            
            # Store overlap for next chunk
            self.overlap_buffer = chunk[-self.overlap_size:].copy()
        else:
            # Initialize overlap buffer
            self.overlap_buffer = chunk[-self.overlap_size:].copy()
        
        return output
    
    def processing_loop(self):
        """Main processing loop optimized for CleanUNet"""
        while self.is_processing:
            try:
                # Try to read a chunk from input buffer
                audio_chunk = self.input_buffer.read(self.chunk_size)
                
                if audio_chunk is not None:
                    # Process the chunk
                    enhanced_chunk = self.process_chunk(audio_chunk)
                    
                    # Apply overlap-add for smooth transitions
                    if self.overlap_size > 0:
                        enhanced_chunk = self.apply_overlap_add(enhanced_chunk)
                    
                    # Write to output buffer
                    self.output_buffer.write(enhanced_chunk)
                    
                    # Update stats
                    self.stats['chunks_processed'] += 1
                    
                    # Call audio callback if set
                    if self.audio_callback:
                        self.audio_callback(enhanced_chunk)
                
                else:
                    # No data available, sleep briefly
                    time.sleep(0.001)
                    
            except Exception as e:
                print(f"Error in processing loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)
    
    def start_streaming(self):
        """Start streaming processing"""
        if not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self.processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            print("Started CleanUNet streaming processing")
            print(f"Chunk size: {self.chunk_size} samples ({self.chunk_size/self.sample_rate*1000:.1f}ms)")
            print(f"Overlap: {self.overlap_size} samples ({self.overlap_size/self.chunk_size*100:.1f}%)")
    
    def stop_streaming(self):
        """Stop streaming processing"""
        if self.is_processing:
            self.is_processing = False
            if self.processing_thread:
                self.processing_thread.join(timeout=2.0)
            print("Stopped streaming processing")
            print(f"Processed {self.stats['chunks_processed']} chunks")
            print(f"Average processing time: {self.stats['processing_time_ms']:.2f}ms")
    
    def get_processed_audio(self, num_samples: int) -> Optional[np.ndarray]:
        """Get processed audio from output buffer"""
        return self.output_buffer.read(num_samples)
    
    def get_latency_ms(self) -> float:
        """Calculate approximate latency in milliseconds"""
        algorithmic_latency = self.chunk_size - self.overlap_size
        return (algorithmic_latency / self.sample_rate) * 1000
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        return self.stats.copy()


# Optimized loading function for CleanUNet
def load_cleanunet_streaming(checkpoint_path: str, config_path: str = None, device: str = None):
    """
    Load CleanUNet model for streaming with proper configuration.
    
    Args:
        checkpoint_path: Path to your trained model checkpoint (.pkl file)
        config_path: Optional path to config.json file
        device: Device to load model on
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model configuration if provided
    if config_path:
        import json
        with open(config_path) as f:
            config = json.load(f)
        network_config = config.get("network_config", {})
    else:
        # Default CleanUNet configuration (adjust to match your training)
        network_config = {
            "channels_input": 1,
            "channels_output": 1,
            "channels_H": 64,
            "max_H": 768,
            "encoder_n_layers": 8,
            "kernel_size": 4,
            "stride": 2,
            "tsfm_n_layers": 3,
            "tsfm_n_head": 8,
            "tsfm_d_model": 512,
            "tsfm_d_inner": 2048
        }
    
    # Import and initialize your CleanUNet model
    from network import CleanUNet  # Adjust import based on your file structure
    model = CleanUNet(**network_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded CleanUNet model from {checkpoint_path}")
    print(f"Model configuration: {network_config}")
    
    return model


# Example usage
def demo_cleanunet_streaming(checkpoint_path: str, config_path: str = None):
    """Demonstrate CleanUNet streaming inference"""
    
    # Load your trained model
    model = load_cleanunet_streaming(checkpoint_path, config_path)
    
    # Create streaming processor with optimized settings for CleanUNet
    streamer = CleanUNetStreaming(
        model=model,
        sample_rate=16000,
        chunk_size=2048,  # ~128ms chunks work well with CleanUNet
        overlap_size=512,  # 25% overlap for smooth transitions
        buffer_size_seconds=2.0
    )
    
    # Set up audio callback for real-time output
    processed_chunks = []
    def audio_callback(chunk):
        processed_chunks.append(chunk.copy())
        print(f"Enhanced chunk: {len(chunk)} samples, "
              f"RMS: {np.sqrt(np.mean(chunk**2)):.4f}")
    
    streamer.set_audio_callback(audio_callback)
    
    # Generate test audio (replace with your audio source)
    print("Generating test audio...")
    duration = 5  # seconds
    t = np.linspace(0, duration, duration * 16000)
    # Simulate noisy audio
    clean_signal = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    noise = 0.05 * np.random.randn(len(clean_signal))
    noisy_audio = (clean_signal + noise).astype(np.float32)
    
    # Start streaming
    streamer.start_streaming()
    
    try:
        print("Starting CleanUNet streaming demo...")
        
        # Stream audio in real-time chunks
        chunk_size = 1024
        for i in range(0, len(noisy_audio), chunk_size):
            chunk = noisy_audio[i:i+chunk_size]
            
            # Add to streamer
            streamer.add_audio(chunk)
            
            # Simulate real-time by sleeping
            time.sleep(chunk_size / 16000 * 0.8)  # Slightly faster than real-time
        
        # Wait for processing to complete
        time.sleep(2.0)
        
        # Print statistics
        stats = streamer.get_stats()
        print(f"\nProcessing Statistics:")
        print(f"Chunks processed: {stats['chunks_processed']}")
        print(f"Average std: {stats['avg_std']:.4f}")
        print(f"Average processing time: {stats['processing_time_ms']:.2f}ms")
        print(f"Theoretical latency: {streamer.get_latency_ms():.2f}ms")
        
        if processed_chunks:
            total_enhanced = np.concatenate(processed_chunks)
            print(f"Enhanced {len(total_enhanced)} samples total")
            
            # Save enhanced audio for listening
            try:
                import soundfile as sf
                output_file = "demo_enhanced_output.wav"
                sf.write(output_file, total_enhanced, 16000, subtype='PCM_16')
                print(f"Enhanced audio saved to: {output_file}")
                
                # Also save original for comparison
                original_file = "demo_original_noisy.wav"
                sf.write(original_file, noisy_audio[:len(total_enhanced)], 16000, subtype='PCM_16')
                print(f"Original noisy audio saved to: {original_file}")
                
            except ImportError:
                print("Install soundfile to save audio: pip install soundfile")
                print("Processed audio is in memory but not saved")
            
            # Calculate enhancement metrics
            original_rms = np.sqrt(np.mean(noisy_audio[:len(total_enhanced)]**2))
            enhanced_rms = np.sqrt(np.mean(total_enhanced**2))
            print(f"RMS change: {original_rms:.4f} -> {enhanced_rms:.4f}")
        
    finally:
        streamer.stop_streaming()


if __name__ == "__main__":
    # Example usage - replace with your actual checkpoint path
    checkpoint_path = os.path.expanduser("~/CleanUNet/exp/VOICEBANK-large-full/checkpoint/500000.pkl")
    config_path = os.path.expanduser("~/CleanUNet/configs/VOICEBANK-large-full.json")
    
    print("CleanUNet Streaming Inference Demo")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using config: {config_path}")
    
    # Run demo:
    demo_cleanunet_streaming(checkpoint_path, config_path)
