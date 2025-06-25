#!/usr/bin/env python3
"""
Simple CleanUNet Attribution Analyzer that works with Conv1d layers
Uses LayerGradientXActivation instead of LRP which has built-in Conv1d support
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients
from scipy.io.wavfile import read as wavread
import librosa

class SimpleCleanUNetAnalyzer:
    """
    Simple attribution analyzer that works with CleanUNet's Conv1d layers
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.eval()
        self.device = device
        
        # Get target layers
        self.target_layers = self._get_target_layers()
        
        # Use GradientXActivation (works with Conv1d)
        self.analyzers = {}
        for layer_name, layer_module in self.target_layers.items():
            self.analyzers[layer_name] = LayerGradientXActivation(self.model, layer_module)
    
    def _get_target_layers(self):
        """Get key layers for analysis"""
        layers = {}
        
        # Encoder layers
        for i in range(len(self.model.encoder)):
            layers[f'encoder_{i}'] = self.model.encoder[i]
        
        # Transformer components  
        layers['tsfm_conv1'] = self.model.tsfm_conv1
        layers['tsfm_encoder'] = self.model.tsfm_encoder
        layers['tsfm_conv2'] = self.model.tsfm_conv2
        
        # Decoder layers
        for i in range(len(self.model.decoder)):
            layers[f'decoder_{i}'] = self.model.decoder[i]
        
        return layers
    
    def analyze_sample(self, audio_input, sample_type="unknown"):
        """
        Analyze single audio sample and get layer contributions
        """
        
        audio_input = audio_input.to(self.device).requires_grad_(True)
        
        # Target function: energy reduction (proxy for denoising)
        def target_func(x):
            output = self.model(x)
            input_energy = (x ** 2).mean()
            output_energy = (output ** 2).mean()
            return input_energy - output_energy  # How much energy was reduced
        
        print(f"Analyzing {sample_type} sample...")
        
        contributions = {}
        attributions = {}
        
        for layer_name, analyzer in self.analyzers.items():
            try:
                print(f"  Processing {layer_name}...")
                
                # Get attribution
                attribution = analyzer.attribute(
                    audio_input,
                    target=target_func
                )
                
                # Compute contribution score
                contribution_score = torch.abs(attribution).mean().item()
                
                contributions[layer_name] = contribution_score
                attributions[layer_name] = attribution.detach().cpu()
                
                print(f"    {layer_name}: {contribution_score:.4f}")
                
            except Exception as e:
                print(f"    Error with {layer_name}: {e}")
                continue
        
        return {
            'contributions': contributions,
            'attributions': attributions,
            'sample_type': sample_type
        }
    
    def compare_samples(self, clean_audio, noisy_audio, windy_audio=None):
        """
        Compare layer contributions across different sample types
        """
        
        print("🔬 Running comparative layer analysis...")
        
        # Analyze each sample
        clean_results = self.analyze_sample(clean_audio, "clean")
        noisy_results = self.analyze_sample(noisy_audio, "noisy")
        
        windy_results = None
        if windy_audio is not None:
            windy_results = self.analyze_sample(windy_audio, "windy")
        
        # Create comparison plots
        self._plot_comparison(clean_results, noisy_results, windy_results)
        
        # Print insights
        self._print_insights(clean_results, noisy_results, windy_results)
        
        return clean_results, noisy_results, windy_results
    
    def _plot_comparison(self, clean_results, noisy_results, windy_results=None):
        """Create comparison plots"""
        
        # Get common layers
        layers = list(clean_results['contributions'].keys())
        
        clean_scores = [clean_results['contributions'][layer] for layer in layers]
        noisy_scores = [noisy_results['contributions'][layer] for layer in layers]
        windy_scores = [windy_results['contributions'][layer] for layer in layers] if windy_results else None
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CleanUNet Layer Contribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Side-by-side comparison
        x = np.arange(len(layers))
        width = 0.25
        
        axes[0,0].bar(x - width, clean_scores, width, label='Clean', alpha=0.8, color='green')
        axes[0,0].bar(x, noisy_scores, width, label='Noisy', alpha=0.8, color='red')
        if windy_scores:
            axes[0,0].bar(x + width, windy_scores, width, label='Windy', alpha=0.8, color='purple')
        
        axes[0,0].set_xlabel('Layer')
        axes[0,0].set_ylabel('Contribution Score')
        axes[0,0].set_title('Layer Contributions by Sample Type')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(layers, rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Noise specificity (noisy - clean)
        noise_specificity = [noisy - clean for noisy, clean in zip(noisy_scores, clean_scores)]
        colors = ['red' if x > 0 else 'blue' for x in noise_specificity]
        
        axes[0,1].bar(layers, noise_specificity, color=colors, alpha=0.7)
        axes[0,1].set_xlabel('Layer')
        axes[0,1].set_ylabel('Noise Specificity (Noisy - Clean)')
        axes[0,1].set_title('Layers Most Active for Noise Processing')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 3. Top contributing layers for noisy samples
        noisy_contrib_sorted = sorted(zip(layers, noisy_scores), key=lambda x: x[1], reverse=True)
        top_layers = noisy_contrib_sorted[:8]  # Top 8
        
        top_names = [item[0] for item in top_layers]
        top_scores = [item[1] for item in top_layers]
        
        axes[1,0].barh(top_names, top_scores, color='orange', alpha=0.7)
        axes[1,0].set_xlabel('Contribution Score')
        axes[1,0].set_title('Top Contributing Layers (Noisy Audio)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Wind specificity (if available)
        if windy_scores:
            wind_specificity = [windy - noisy for windy, noisy in zip(windy_scores, noisy_scores)]
            colors = ['purple' if x > 0 else 'orange' for x in wind_specificity]
            
            axes[1,1].bar(layers, wind_specificity, color=colors, alpha=0.7)
            axes[1,1].set_xlabel('Layer')
            axes[1,1].set_ylabel('Wind Specificity (Windy - General Noisy)')
            axes[1,1].set_title('Wind-Specific Layer Responses')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        else:
            # If no windy data, show encoder vs decoder breakdown
            encoder_scores = [score for layer, score in zip(layers, noisy_scores) if 'encoder' in layer]
            decoder_scores = [score for layer, score in zip(layers, noisy_scores) if 'decoder' in layer]
            transformer_scores = [score for layer, score in zip(layers, noisy_scores) if 'tsfm' in layer]
            
            breakdown_data = []
            breakdown_labels = []
            
            if encoder_scores:
                breakdown_data.append(np.mean(encoder_scores))
                breakdown_labels.append('Encoders')
            
            if decoder_scores:
                breakdown_data.append(np.mean(decoder_scores))
                breakdown_labels.append('Decoders')
                
            if transformer_scores:
                breakdown_data.append(np.mean(transformer_scores))
                breakdown_labels.append('Transformer')
            
            axes[1,1].pie(breakdown_data, labels=breakdown_labels, autopct='%1.2f%%', 
                         colors=['red', 'blue', 'green'], startangle=90)
            axes[1,1].set_title('Average Contribution by Layer Type')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _print_insights(self, clean_results, noisy_results, windy_results=None):
        """Print key insights from the analysis"""
        
        print("\n" + "="*60)
        print("🎯 KEY INSIGHTS FROM LAYER ANALYSIS")
        print("="*60)
        
        # Get contributions
        clean_contrib = clean_results['contributions']
        noisy_contrib = noisy_results['contributions']
        
        # Find most noise-sensitive layers
        noise_specificity = {layer: noisy_contrib[layer] - clean_contrib[layer] 
                           for layer in clean_contrib.keys()}
        
        sorted_noise_spec = sorted(noise_specificity.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🔴 LAYERS MOST ACTIVE FOR NOISE PROCESSING:")
        for layer, specificity in sorted_noise_spec[:5]:
            print(f"  • {layer}: +{specificity:.4f} (higher activation for noisy vs clean)")
        
        print("\n🔵 LAYERS THAT PRESERVE CLEAN SPEECH:")
        for layer, specificity in sorted_noise_spec[-3:]:
            if specificity < 0:
                print(f"  • {layer}: {specificity:.4f} (less active for noise, preserves clean)")
        
        # Encoder vs Decoder analysis
        encoder_noise_spec = [spec for layer, spec in noise_specificity.items() if 'encoder' in layer]
        decoder_noise_spec = [spec for layer, spec in noise_specificity.items() if 'decoder' in layer]
        
        if encoder_noise_spec and decoder_noise_spec:
            encoder_avg = np.mean(encoder_noise_spec)
            decoder_avg = np.mean(decoder_noise_spec)
            
            print(f"\n🏗️ ARCHITECTURE ANALYSIS:")
            print(f"  • Encoder layers avg noise specificity: {encoder_avg:.4f}")
            print(f"  • Decoder layers avg noise specificity: {decoder_avg:.4f}")
            
            if encoder_avg > decoder_avg * 1.2:
                print("  → Encoders primarily handle noise removal")
                print("  → Decoders focus on clean speech reconstruction")
            elif decoder_avg > encoder_avg * 1.2:
                print("  → Decoders also significantly process noise")
                print("  → More distributed noise handling")
            else:
                print("  → Balanced noise processing across encoder-decoder")
        
        # Wind-specific analysis
        if windy_results:
            windy_contrib = windy_results['contributions']
            wind_specificity = {layer: windy_contrib[layer] - noisy_contrib[layer]
                              for layer in noisy_contrib.keys()}
            
            sorted_wind_spec = sorted(wind_specificity.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print(f"\n🌪️ WIND-SPECIFIC LAYER RESPONSES:")
            for layer, specificity in sorted_wind_spec[:5]:
                direction = "MORE active for wind" if specificity > 0 else "LESS active for wind"
                print(f"  • {layer}: {specificity:+.4f} ({direction} than general noise)")
        
        # Find potential wind specialist layers
        if windy_results:
            wind_specialists = [(layer, abs(spec)) for layer, spec in wind_specificity.items() 
                              if abs(spec) > 0.01]  # Threshold for significant difference
            
            if wind_specialists:
                wind_specialists.sort(key=lambda x: x[1], reverse=True)
                print(f"\n⭐ POTENTIAL WIND SPECIALIST LAYERS:")
                for layer, abs_spec in wind_specialists[:3]:
                    print(f"  • {layer}: Shows strong wind-specific response")


def load_audio_sample(path, duration=4.0, sample_rate=16000):
    """Load a single audio sample"""
    
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


def analyze_cleanunet_simple(model, clean_path, noisy_path, windy_path=None):
    """
    Simple CleanUNet analysis that works with Conv1d layers
    """
    
    print("🚀 Simple CleanUNet Layer Analysis")
    print("="*50)
    
    # Load audio samples
    print("Loading audio samples...")
    clean_audio = load_audio_sample(clean_path)
    noisy_audio = load_audio_sample(noisy_path)
    windy_audio = load_audio_sample(windy_path) if windy_path else None
    
    print(f"✅ Loaded samples:")
    print(f"  Clean: {clean_audio.shape}")
    print(f"  Noisy: {noisy_audio.shape}")
    if windy_audio is not None:
        print(f"  Windy: {windy_audio.shape}")
    
    # Initialize analyzer
    analyzer = SimpleCleanUNetAnalyzer(model)
    
    # Run comparative analysis
    results = analyzer.compare_samples(clean_audio, noisy_audio, windy_audio)
    
    print("\n✅ Analysis complete! Check the plots above for visual insights.")
    
    return results


if __name__ == "__main__":
    print("Simple CleanUNet Attribution Analyzer")
    print("Works with Conv1d layers using GradientXActivation")
    print("Use analyze_cleanunet_simple() function to run analysis")