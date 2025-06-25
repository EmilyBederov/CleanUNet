#!/usr/bin/env python3
"""
CleanUNet Attention Map Analyzer for Wind Noise

Analyzes where different layers focus their attention during wind noise processing
vs clean speech processing to identify specialization patterns.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
from scipy.io.wavfile import read as wavread
import librosa
from scipy.signal import find_peaks
import json
from network import CleanUNet
import cv2

class AttentionMapAnalyzer:
    """
    Analyze attention maps to understand where layers focus during wind vs clean processing
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.eval()
        self.device = device
        
        # Storage for activations and attention maps
        self.activations = {}
        self.attention_maps = {}
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture activations for attention analysis"""
        
        # Hook encoder layers
        for i, encoder_block in enumerate(self.model.encoder):
            hook = encoder_block.register_forward_hook(
                lambda module, input, output, layer_idx=i: 
                self._save_activation(f'encoder_{layer_idx}', output)
            )
            self.hooks.append(hook)
        
        # Hook transformer components
        self.hooks.append(
            self.model.tsfm_conv1.register_forward_hook(
                lambda module, input, output: self._save_activation('tsfm_conv1', output)
            )
        )
        
        self.hooks.append(
            self.model.tsfm_encoder.register_forward_hook(
                lambda module, input, output: self._save_activation('tsfm_encoder', output)
            )
        )
        
        self.hooks.append(
            self.model.tsfm_conv2.register_forward_hook(
                lambda module, input, output: self._save_activation('tsfm_conv2', output)
            )
        )
        
        # Hook decoder layers
        for i, decoder_block in enumerate(self.model.decoder):
            hook = decoder_block.register_forward_hook(
                lambda module, input, output, layer_idx=i: 
                self._save_activation(f'decoder_{layer_idx}', output)
            )
            self.hooks.append(hook)
    
    def _save_activation(self, name, activation):
        """Save activation and compute attention map"""
        if isinstance(activation, tuple):
            activation = activation[0]
        
        self.activations[name] = activation.detach().cpu()
        
        # Compute attention map (spatial attention across time)
        # Average across channels to get time-based attention
        attention = activation.mean(dim=1)  # (batch, time)
        
        # Normalize to create attention weights
        attention = torch.abs(attention)
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)
        
        self.attention_maps[name] = attention.detach().cpu()
    
    def analyze_attention_patterns(self, clean_audio, windy_audio, audio_labels=None):
        """
        Analyze attention patterns for clean vs windy audio
        
        Args:
            clean_audio: clean speech sample
            windy_audio: windy speech sample  
            audio_labels: optional labels for the samples
        """
        
        if audio_labels is None:
            audio_labels = ['Clean Speech', 'Windy Speech']
        
        print("🔍 Analyzing attention patterns...")
        
        # Analyze clean audio
        print("  Processing clean audio...")
        clean_results = self._process_sample(clean_audio, 'clean')
        
        # Analyze windy audio
        print("  Processing windy audio...")
        windy_results = self._process_sample(windy_audio, 'windy')
        
        # Compare attention patterns
        attention_analysis = self._compare_attention_patterns(
            clean_results, windy_results, audio_labels
        )
        
        # Create comprehensive visualizations
        self._create_attention_visualizations(
            clean_results, windy_results, attention_analysis, audio_labels
        )
        
        # Print insights
        self._print_attention_insights(attention_analysis)
        
        return clean_results, windy_results, attention_analysis
    
    def _process_sample(self, audio_input, sample_type):
        """Process single sample and capture attention maps"""
        
        self.activations.clear()
        self.attention_maps.clear()
        
        audio_input = audio_input.to(self.device)
        
        # Forward pass (captures activations via hooks)
        with torch.no_grad():
            output = self.model(audio_input)
        
        return {
            'audio_input': audio_input.cpu(),
            'audio_output': output.cpu(), 
            'activations': dict(self.activations),
            'attention_maps': dict(self.attention_maps),
            'sample_type': sample_type
        }
    
    def _compare_attention_patterns(self, clean_results, windy_results, labels):
        """Compare attention patterns between clean and windy audio"""
        
        analysis = {
            'attention_differences': {},
            'specialization_scores': {},
            'temporal_focus': {},
            'wind_vs_speech_regions': {}
        }
        
        # Compare each layer's attention
        for layer_name in clean_results['attention_maps'].keys():
            clean_attention = clean_results['attention_maps'][layer_name][0]  # First sample
            windy_attention = windy_results['attention_maps'][layer_name][0]
            
            # Ensure same length (might differ due to downsampling)
            min_len = min(len(clean_attention), len(windy_attention))
            clean_attention = clean_attention[:min_len]
            windy_attention = windy_attention[:min_len]
            
            # Compute attention difference
            attention_diff = windy_attention - clean_attention
            
            # Specialization score: how different are the attention patterns?
            specialization = torch.norm(attention_diff).item()
            
            # Temporal focus: where does each layer focus most?
            clean_peak_idx = torch.argmax(clean_attention).item()
            windy_peak_idx = torch.argmax(windy_attention).item()
            
            # Focus shift: how much does focus location change?
            focus_shift = abs(windy_peak_idx - clean_peak_idx) / len(clean_attention)
            
            analysis['attention_differences'][layer_name] = attention_diff
            analysis['specialization_scores'][layer_name] = specialization
            analysis['temporal_focus'][layer_name] = {
                'clean_peak': clean_peak_idx,
                'windy_peak': windy_peak_idx,
                'focus_shift': focus_shift
            }
            
            # Identify wind vs speech regions
            wind_regions = self._identify_wind_regions(attention_diff)
            analysis['wind_vs_speech_regions'][layer_name] = wind_regions
        
        # Rank layers by specialization
        specialization_ranking = sorted(
            analysis['specialization_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        analysis['most_specialized_layers'] = specialization_ranking[:5]
        
        return analysis
    
    def _identify_wind_regions(self, attention_diff, threshold=0.1):
        """Identify regions where layer pays more attention to wind vs speech"""
        
        # Positive values = more attention to windy audio
        # Negative values = more attention to clean speech
        
        wind_focused = attention_diff > threshold
        speech_focused = attention_diff < -threshold
        
        return {
            'wind_regions': wind_focused.nonzero().squeeze().tolist() if wind_focused.any() else [],
            'speech_regions': speech_focused.nonzero().squeeze().tolist() if speech_focused.any() else [],
            'wind_percentage': wind_focused.float().mean().item(),
            'speech_percentage': speech_focused.float().mean().item()
        }
    
    def _create_attention_visualizations(self, clean_results, windy_results, analysis, labels):
        """Create comprehensive attention visualizations"""
        
        # Main attention heatmap plot
        self._plot_attention_heatmaps(clean_results, windy_results, labels)
        
        # Attention difference analysis
        self._plot_attention_differences(analysis)
        
        # Temporal focus analysis
        self._plot_temporal_focus(analysis, labels)
        
        # Layer specialization summary
        self._plot_specialization_summary(analysis)
    
    def _plot_attention_heatmaps(self, clean_results, windy_results, labels):
        """Plot attention heatmaps for all layers"""
        
        layers = list(clean_results['attention_maps'].keys())
        n_layers = len(layers)
        
        # Create large figure for all attention maps
        fig, axes = plt.subplots(n_layers, 3, figsize=(18, 3 * n_layers))
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Layer Attention Maps: Clean vs Windy Speech', fontsize=16, fontweight='bold')
        
        for i, layer_name in enumerate(layers):
            clean_attention = clean_results['attention_maps'][layer_name][0].numpy()
            windy_attention = windy_results['attention_maps'][layer_name][0].numpy()
            
            # Ensure same length
            min_len = min(len(clean_attention), len(windy_attention))
            clean_attention = clean_attention[:min_len]
            windy_attention = windy_attention[:min_len]
            
            # Time axis
            time_axis = np.arange(len(clean_attention))
            
            # Plot clean attention
            axes[i, 0].plot(time_axis, clean_attention, color='green', alpha=0.8, linewidth=2)
            axes[i, 0].fill_between(time_axis, clean_attention, alpha=0.3, color='green')
            axes[i, 0].set_title(f'{layer_name} - {labels[0]}')
            axes[i, 0].set_ylabel('Attention Weight')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot windy attention
            axes[i, 1].plot(time_axis, windy_attention, color='red', alpha=0.8, linewidth=2)
            axes[i, 1].fill_between(time_axis, windy_attention, alpha=0.3, color='red')
            axes[i, 1].set_title(f'{layer_name} - {labels[1]}')
            axes[i, 1].set_ylabel('Attention Weight')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Plot difference
            attention_diff = windy_attention - clean_attention
            colors = ['red' if x > 0 else 'blue' for x in attention_diff]
            
            axes[i, 2].bar(time_axis, attention_diff, color=colors, alpha=0.7, width=1.0)
            axes[i, 2].set_title(f'{layer_name} - Attention Difference')
            axes[i, 2].set_ylabel('Windy - Clean Attention')
            axes[i, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[i, 2].grid(True, alpha=0.3)
            
            if i == n_layers - 1:  # Last row
                axes[i, 0].set_xlabel('Time Steps')
                axes[i, 1].set_xlabel('Time Steps') 
                axes[i, 2].set_xlabel('Time Steps')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = 'attention_heatmaps_analysis.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Attention heatmaps saved as: {plot_filename}")
    
    def _plot_attention_differences(self, analysis):
        """Plot attention difference analysis"""
        
        layers = list(analysis['attention_differences'].keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Attention Pattern Analysis: Wind vs Speech Specialization', 
                     fontsize=16, fontweight='bold')
        
        # 1. Specialization scores
        specialization_scores = [analysis['specialization_scores'][layer] for layer in layers]
        
        axes[0, 0].bar(layers, specialization_scores, color='purple', alpha=0.7)
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Specialization Score')
        axes[0, 0].set_title('Layer Specialization (Higher = More Different Attention)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Focus shift analysis
        focus_shifts = [analysis['temporal_focus'][layer]['focus_shift'] for layer in layers]
        
        axes[0, 1].bar(layers, focus_shifts, color='orange', alpha=0.7)
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Focus Shift (0-1)')
        axes[0, 1].set_title('Temporal Focus Shift (Clean → Windy)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Wind vs speech region analysis
        wind_percentages = [analysis['wind_vs_speech_regions'][layer]['wind_percentage'] 
                           for layer in layers]
        speech_percentages = [analysis['wind_vs_speech_regions'][layer]['speech_percentage']
                             for layer in layers]
        
        x = np.arange(len(layers))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, wind_percentages, width, label='Wind Focus %', 
                      alpha=0.8, color='red')
        axes[1, 0].bar(x + width/2, speech_percentages, width, label='Speech Focus %', 
                      alpha=0.8, color='blue')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Percentage of Time Steps')
        axes[1, 0].set_title('Wind vs Speech Focus Distribution')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(layers, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Top specialized layers
        top_layers = analysis['most_specialized_layers']
        top_names = [item[0] for item in top_layers]
        top_scores = [item[1] for item in top_layers]
        
        axes[1, 1].barh(top_names, top_scores, color='green', alpha=0.7)
        axes[1, 1].set_xlabel('Specialization Score')
        axes[1, 1].set_title('Most Specialized Layers (Top 5)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = 'attention_difference_analysis.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Attention analysis saved as: {plot_filename}")
    
    def _plot_temporal_focus(self, analysis, labels):
        """Plot temporal focus patterns"""
        
        layers = list(analysis['temporal_focus'].keys())
        
        clean_peaks = [analysis['temporal_focus'][layer]['clean_peak'] for layer in layers]
        windy_peaks = [analysis['temporal_focus'][layer]['windy_peak'] for layer in layers]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Temporal Focus Analysis: Where Layers Pay Attention', 
                     fontsize=16, fontweight='bold')
        
        # 1. Peak attention locations
        x = np.arange(len(layers))
        width = 0.35
        
        axes[0].bar(x - width/2, clean_peaks, width, label=labels[0], alpha=0.8, color='green')
        axes[0].bar(x + width/2, windy_peaks, width, label=labels[1], alpha=0.8, color='red')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Peak Attention Time Step')
        axes[0].set_title('Peak Attention Locations')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(layers, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Focus shift magnitude
        focus_shifts = [analysis['temporal_focus'][layer]['focus_shift'] for layer in layers]
        colors = plt.cm.viridis(np.array(focus_shifts) / max(focus_shifts))
        
        bars = axes[1].bar(layers, focus_shifts, color=colors, alpha=0.8)
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Focus Shift (Normalized)')
        axes[1].set_title('Attention Focus Shift Magnitude')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                  norm=plt.Normalize(vmin=0, vmax=max(focus_shifts)))
        sm.set_array([])
        plt.colorbar(sm, ax=axes[1], label='Focus Shift')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = 'temporal_focus_analysis.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Temporal focus analysis saved as: {plot_filename}")
    
    def _plot_specialization_summary(self, analysis):
        """Plot summary of layer specialization"""
        
        layers = list(analysis['specialization_scores'].keys())
        
        # Prepare data for summary
        specialization_data = []
        layer_types = []
        
        for layer in layers:
            spec_score = analysis['specialization_scores'][layer]
            wind_pct = analysis['wind_vs_speech_regions'][layer]['wind_percentage']
            speech_pct = analysis['wind_vs_speech_regions'][layer]['speech_percentage']
            
            specialization_data.append([spec_score, wind_pct, speech_pct])
            
            if 'encoder' in layer:
                layer_types.append('Encoder')
            elif 'decoder' in layer:
                layer_types.append('Decoder')
            else:
                layer_types.append('Transformer')
        
        specialization_data = np.array(specialization_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Layer Specialization Summary', fontsize=16, fontweight='bold')
        
        # 1. Specialization vs Wind Focus scatter
        colors = {'Encoder': 'red', 'Decoder': 'blue', 'Transformer': 'green'}
        
        for layer_type in set(layer_types):
            mask = [lt == layer_type for lt in layer_types]
            axes[0].scatter(specialization_data[mask, 0], specialization_data[mask, 1], 
                           c=colors[layer_type], label=layer_type, alpha=0.7, s=100)
        
        axes[0].set_xlabel('Specialization Score')
        axes[0].set_ylabel('Wind Focus Percentage')
        axes[0].set_title('Specialization vs Wind Focus')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add layer labels
        for i, layer in enumerate(layers):
            axes[0].annotate(layer.replace('encoder_', 'E').replace('decoder_', 'D').replace('tsfm_', 'T'), 
                           (specialization_data[i, 0], specialization_data[i, 1]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 2. Architecture component analysis
        encoder_indices = [i for i, lt in enumerate(layer_types) if lt == 'Encoder']
        decoder_indices = [i for i, lt in enumerate(layer_types) if lt == 'Decoder']
        transformer_indices = [i for i, lt in enumerate(layer_types) if lt == 'Transformer']
        
        component_data = []
        component_labels = []
        
        if encoder_indices:
            component_data.append(np.mean(specialization_data[encoder_indices, 0]))
            component_labels.append(f'Encoders ({len(encoder_indices)})')
        
        if decoder_indices:
            component_data.append(np.mean(specialization_data[decoder_indices, 0]))
            component_labels.append(f'Decoders ({len(decoder_indices)})')
            
        if transformer_indices:
            component_data.append(np.mean(specialization_data[transformer_indices, 0]))
            component_labels.append(f'Transformer ({len(transformer_indices)})')
        
        axes[1].pie(component_data, labels=component_labels, autopct='%1.1f%%',
                   colors=['red', 'blue', 'green'][:len(component_data)], startangle=90)
        axes[1].set_title('Average Specialization by Architecture Component')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = 'specialization_summary.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Specialization summary saved as: {plot_filename}")
    
    def _print_attention_insights(self, analysis):
        """Print key insights from attention analysis"""
        
        print("\n" + "="*60)
        print("🎯 ATTENTION MAP ANALYSIS INSIGHTS")
        print("="*60)
        
        # Most specialized layers
        print("\n🔍 MOST SPECIALIZED LAYERS (Different attention for wind vs speech):")
        for layer_name, spec_score in analysis['most_specialized_layers']:
            wind_regions = analysis['wind_vs_speech_regions'][layer_name]
            temporal_focus = analysis['temporal_focus'][layer_name]
            
            print(f"  • {layer_name}:")
            print(f"    Specialization score: {spec_score:.4f}")
            print(f"    Wind focus: {wind_regions['wind_percentage']*100:.1f}% of time")
            print(f"    Speech focus: {wind_regions['speech_percentage']*100:.1f}% of time")
            print(f"    Focus shift: {temporal_focus['focus_shift']:.3f}")
        
        # Architecture analysis
        encoder_specs = [score for layer, score in analysis['specialization_scores'].items() 
                        if 'encoder' in layer]
        decoder_specs = [score for layer, score in analysis['specialization_scores'].items() 
                        if 'decoder' in layer]
        transformer_specs = [score for layer, score in analysis['specialization_scores'].items() 
                            if 'tsfm' in layer]
        
        print(f"\n🏗️ ATTENTION SPECIALIZATION BY ARCHITECTURE:")
        if encoder_specs:
            print(f"  • Encoder layers avg specialization: {np.mean(encoder_specs):.4f}")
        if decoder_specs:
            print(f"  • Decoder layers avg specialization: {np.mean(decoder_specs):.4f}")
        if transformer_specs:
            print(f"  • Transformer avg specialization: {np.mean(transformer_specs):.4f}")
        
        # Find wind vs speech specialists
        wind_specialists = []
        speech_specialists = []
        
        for layer_name, regions in analysis['wind_vs_speech_regions'].items():
            if regions['wind_percentage'] > 0.3:  # >30% focus on wind
                wind_specialists.append((layer_name, regions['wind_percentage']))
            elif regions['speech_percentage'] > 0.3:  # >30% focus on speech
                speech_specialists.append((layer_name, regions['speech_percentage']))
        
        if wind_specialists:
            print(f"\n🌪️ WIND SPECIALIST LAYERS:")
            for layer, wind_pct in sorted(wind_specialists, key=lambda x: x[1], reverse=True):
                print(f"  • {layer}: {wind_pct*100:.1f}% wind focus")
        
        if speech_specialists:
            print(f"\n🗣️ SPEECH PRESERVATION LAYERS:")
            for layer, speech_pct in sorted(speech_specialists, key=lambda x: x[1], reverse=True):
                print(f"  • {layer}: {speech_pct*100:.1f}% speech focus")
        
        # Temporal focus insights
        large_shifts = [(layer, data['focus_shift']) for layer, data in analysis['temporal_focus'].items() 
                       if data['focus_shift'] > 0.2]
        
        if large_shifts:
            print(f"\n⏰ LAYERS WITH MAJOR TEMPORAL FOCUS SHIFTS:")
            for layer, shift in sorted(large_shifts, key=lambda x: x[1], reverse=True):
                print(f"  • {layer}: {shift:.3f} focus shift (attention moves significantly)")
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def load_audio_sample(path, duration=4.0, sample_rate=16000):
    """Load audio sample for attention analysis"""
    
    if path.endswith('.wav'):
        try:
            sr, audio = wavread(path)
            audio = audio.astype(np.float32) / 32768.0
        except:
            audio, sr = librosa.load(path, sr=sample_rate)
    else:
        audio, sr = librosa.load(path, sr=sample_rate)
    
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    
    target_length = int(duration * sample_rate)
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)))
    
    return torch.FloatTensor(audio).unsqueeze(0)


def analyze_attention_maps(model, clean_path, windy_path, labels=None):
    """
    Complete attention map analysis for CleanUNet wind vs speech processing
    
    Args:
        model: CleanUNet model
        clean_path: path to clean speech sample
        windy_path: path to windy speech sample
        labels: optional custom labels for the samples
    """
    
    print("🔍 CleanUNet Attention Map Analysis")
    print("="*50)
    
    # Load audio samples
    print("Loading audio samples...")
    clean_audio = load_audio_sample(clean_path)
    windy_audio = load_audio_sample(windy_path)
    
    print(f"✅ Loaded samples:")
    print(f"  Clean: {clean_audio.shape}")
    print(f"  Windy: {windy_audio.shape}")
    
    # Initialize analyzer
    analyzer = AttentionMapAnalyzer(model)
    
    try:
        # Run attention analysis
        clean_results, windy_results, attention_analysis = analyzer.analyze_attention_patterns(
            clean_audio, windy_audio, labels
        )
        
        print("\n✅ Attention analysis complete!")
        print("📊 Generated plots:")
        print("  - attention_heatmaps_analysis.png")
        print("  - attention_difference_analysis.png")
        print("  - temporal_focus_analysis.png")
        print("  - specialization_summary.png")
        
        return clean_results, windy_results, attention_analysis
        
    finally:
        analyzer.cleanup()


def analyze_attention_maps(model, clean_path, windy_path, labels=None):
    """
    Complete attention map analysis for CleanUNet wind vs speech processing
    
    Args:
        model: CleanUNet model
        clean_path: path to clean speech sample
        windy_path: path to windy speech sample
        labels: optional custom labels for the samples
    """
    
    print("🔍 CleanUNet Attention Map Analysis")
    print("="*50)
    
    # Load audio samples
    print("Loading audio samples...")
    clean_audio = load_audio_sample(clean_path)
    windy_audio = load_audio_sample(windy_path)
    
    print(f"  Loaded samples:")
    print(f"  Clean: {clean_audio.shape}")
    print(f"  Windy: {windy_audio.shape}")
    
    # Initialize analyzer
    analyzer = AttentionMapAnalyzer(model)
    
    try:
        # Run attention analysis
        clean_results, windy_results, attention_analysis = analyzer.analyze_attention_patterns(
            clean_audio, windy_audio, labels
        )
        
        print("\n Attention analysis complete!")
        print(" Generated plots:")
        print("  - attention_heatmaps_analysis.png")
        print("  - attention_difference_analysis.png")
        print("  - temporal_focus_analysis.png")
        print("  - specialization_summary.png")
        
        return clean_results, windy_results, attention_analysis
        
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    print("CleanUNet Attention Map Analyzer")
    print("Analyzes where different layers focus during wind vs speech processing")
    print("Use analyze_attention_maps() function to run analysis")
    
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

    # Run attention analysis
    clean_results, windy_results, attention_analysis = analyze_attention_maps(
        model=net,
        clean_path="../data/clean_samples/Sample32.wav",    # UPDATE
        windy_path="../data/wind_samples/WinD131.wav",    # UPDATE
        labels=['Clean Speech', 'Windy Speech']
    )
        
    print("CleanUNet Feature Analysis Toolkit Ready!")
    print("Use analyze_cleanunet_features() function to run complete analysis.")

