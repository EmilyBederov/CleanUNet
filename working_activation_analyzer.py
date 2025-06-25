#!/usr/bin/env python3
"""
Working CleanUNet Activation Analyzer
Uses activation magnitude analysis instead of gradients to avoid in-place operation issues
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
from scipy.io.wavfile import read as wavread
import librosa

class WorkingCleanUNetAnalyzer:
    """
    Activation-based analyzer that actually works with CleanUNet
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.eval()
        self.device = device
        
        # Storage for activations
        self.activations = {}
        self.hooks = []
        
        # Register hooks to capture activations
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations"""
        
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
        """Save activation for analysis"""
        if isinstance(activation, tuple):
            activation = activation[0]
        self.activations[name] = activation.detach().cpu()
    
    def analyze_sample(self, audio_input, sample_type="unknown"):
        """
        Analyze single audio sample using activation patterns
        """
        
        self.activations.clear()
        
        audio_input = audio_input.to(self.device)
        
        print(f"Analyzing {sample_type} sample...")
        
        # Forward pass (captures activations via hooks)
        with torch.no_grad():
            output = self.model(audio_input)
        
        # Compute activation statistics
        layer_stats = {}
        
        for layer_name, activation in self.activations.items():
            # Compute various activation metrics
            act_mean = activation.mean().item()
            act_std = activation.std().item()
            act_max = activation.max().item()
            act_energy = (activation ** 2).mean().item()
            
            # Sparsity (fraction of near-zero activations)
            sparsity = (torch.abs(activation) < 0.01).float().mean().item()
            
            layer_stats[layer_name] = {
                'mean': act_mean,
                'std': act_std,
                'max': act_max,
                'energy': act_energy,
                'sparsity': sparsity,
                'activity_score': act_energy * (1 - sparsity)  # High energy, low sparsity = active
            }
            
            print(f"  {layer_name}: energy={act_energy:.4f}, activity={layer_stats[layer_name]['activity_score']:.4f}")
        
        return {
            'layer_stats': layer_stats,
            'sample_type': sample_type,
            'raw_activations': dict(self.activations)
        }
    
    def compare_samples(self, clean_audio, noisy_audio, windy_audio=None):
        """
        Compare activation patterns across different sample types
        """
        
        print("🔬 Running activation-based layer analysis...")
        
        # Analyze each sample
        clean_results = self.analyze_sample(clean_audio, "clean")
        noisy_results = self.analyze_sample(noisy_audio, "noisy")
        
        windy_results = None
        if windy_audio is not None:
            windy_results = self.analyze_sample(windy_audio, "windy")
        
        # Compute differences and insights
        insights = self._compute_insights(clean_results, noisy_results, windy_results)
        
        # Create visualizations
        self._create_visualizations(clean_results, noisy_results, windy_results, insights)
        
        # Print insights
        self._print_insights(insights)
        
        return clean_results, noisy_results, windy_results, insights
    
    def _compute_insights(self, clean_results, noisy_results, windy_results=None):
        """Compute insights from activation differences"""
        
        clean_stats = clean_results['layer_stats']
        noisy_stats = noisy_results['layer_stats']
        
        insights = {
            'noise_sensitivity': {},
            'activity_changes': {},
            'layer_rankings': {}
        }
        
        # Compute noise sensitivity for each layer
        for layer_name in clean_stats.keys():
            clean_activity = clean_stats[layer_name]['activity_score']
            noisy_activity = noisy_stats[layer_name]['activity_score']
            
            # Noise sensitivity = how much activity changes with noise
            noise_sensitivity = noisy_activity - clean_activity
            activity_ratio = noisy_activity / (clean_activity + 1e-8)
            
            insights['noise_sensitivity'][layer_name] = noise_sensitivity
            insights['activity_changes'][layer_name] = {
                'clean_activity': clean_activity,
                'noisy_activity': noisy_activity,
                'activity_ratio': activity_ratio,
                'noise_sensitivity': noise_sensitivity
            }
        
        # Rank layers by noise sensitivity
        noise_ranking = sorted(
            insights['noise_sensitivity'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        insights['layer_rankings']['most_noise_sensitive'] = noise_ranking[:5]
        insights['layer_rankings']['least_noise_sensitive'] = noise_ranking[-5:]
        
        # Wind-specific analysis
        if windy_results:
            windy_stats = windy_results['layer_stats']
            insights['wind_specificity'] = {}
            
            for layer_name in clean_stats.keys():
                windy_activity = windy_stats[layer_name]['activity_score']
                noisy_activity = noisy_stats[layer_name]['activity_score']
                
                # Wind specificity = how wind differs from general noise
                wind_specificity = windy_activity - noisy_activity
                insights['wind_specificity'][layer_name] = wind_specificity
            
            # Rank layers by wind specificity
            wind_ranking = sorted(
                insights['wind_specificity'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            insights['layer_rankings']['most_wind_specific'] = wind_ranking[:5]
        
        return insights
    
    def _create_visualizations(self, clean_results, noisy_results, windy_results, insights):
        """Create comprehensive visualizations"""
        
        # Get data for plotting
        layers = list(clean_results['layer_stats'].keys())
        
        clean_activities = [clean_results['layer_stats'][layer]['activity_score'] for layer in layers]
        noisy_activities = [noisy_results['layer_stats'][layer]['activity_score'] for layer in layers]
        windy_activities = [windy_results['layer_stats'][layer]['activity_score'] for layer in layers] if windy_results else None
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CleanUNet Layer Activation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Activity comparison
        x = np.arange(len(layers))
        width = 0.25
        
        axes[0,0].bar(x - width, clean_activities, width, label='Clean', alpha=0.8, color='green')
        axes[0,0].bar(x, noisy_activities, width, label='Noisy', alpha=0.8, color='red')
        if windy_activities:
            axes[0,0].bar(x + width, windy_activities, width, label='Windy', alpha=0.8, color='purple')
        
        axes[0,0].set_xlabel('Layer')
        axes[0,0].set_ylabel('Activity Score')
        axes[0,0].set_title('Layer Activity Scores by Sample Type')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(layers, rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Noise sensitivity
        noise_sensitivities = [insights['noise_sensitivity'][layer] for layer in layers]
        colors = ['red' if x > 0 else 'blue' for x in noise_sensitivities]
        
        axes[0,1].bar(layers, noise_sensitivities, color=colors, alpha=0.7)
        axes[0,1].set_xlabel('Layer')
        axes[0,1].set_ylabel('Noise Sensitivity (Noisy - Clean Activity)')
        axes[0,1].set_title('Layer Noise Sensitivity')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 3. Top noise-sensitive layers
        top_noise_layers = insights['layer_rankings']['most_noise_sensitive']
        top_names = [item[0] for item in top_noise_layers]
        top_scores = [item[1] for item in top_noise_layers]
        
        axes[1,0].barh(top_names, top_scores, color='orange', alpha=0.7)
        axes[1,0].set_xlabel('Noise Sensitivity Score')
        axes[1,0].set_title('Most Noise-Sensitive Layers')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Wind specificity or architecture breakdown
        if windy_results and 'wind_specificity' in insights:
            wind_specificities = [insights['wind_specificity'][layer] for layer in layers]
            colors = ['purple' if x > 0 else 'orange' for x in wind_specificities]
            
            axes[1,1].bar(layers, wind_specificities, color=colors, alpha=0.7)
            axes[1,1].set_xlabel('Layer')
            axes[1,1].set_ylabel('Wind Specificity (Windy - General Noisy)')
            axes[1,1].set_title('Wind-Specific Layer Responses')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        else:
            # Architecture breakdown
            encoder_activities = [act for layer, act in zip(layers, noisy_activities) if 'encoder' in layer]
            decoder_activities = [act for layer, act in zip(layers, noisy_activities) if 'decoder' in layer]
            transformer_activities = [act for layer, act in zip(layers, noisy_activities) if 'tsfm' in layer]
            
            breakdown_data = []
            breakdown_labels = []
            
            if encoder_activities:
                breakdown_data.append(np.mean(encoder_activities))
                breakdown_labels.append(f'Encoders ({len(encoder_activities)})')
            
            if decoder_activities:
                breakdown_data.append(np.mean(decoder_activities))
                breakdown_labels.append(f'Decoders ({len(decoder_activities)})')
                
            if transformer_activities:
                breakdown_data.append(np.mean(transformer_activities))
                breakdown_labels.append(f'Transformer ({len(transformer_activities)})')
            
            if breakdown_data:
                wedges, texts, autotexts = axes[1,1].pie(
                    breakdown_data, 
                    labels=breakdown_labels, 
                    autopct='%1.1f%%',
                    colors=['red', 'blue', 'green'], 
                    startangle=90
                )
                axes[1,1].set_title('Average Activity by Architecture Component')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _print_insights(self, insights):
        """Print key insights"""
        
        print("\n" + "="*60)
        print("🎯 KEY INSIGHTS FROM ACTIVATION ANALYSIS")
        print("="*60)
        
        # Most noise-sensitive layers
        print("\n🔴 LAYERS MOST ACTIVE FOR NOISE PROCESSING:")
        for layer_name, sensitivity in insights['layer_rankings']['most_noise_sensitive']:
            activity_info = insights['activity_changes'][layer_name]
            print(f"  • {layer_name}: +{sensitivity:.4f} sensitivity")
            print(f"    Clean→Noisy activity: {activity_info['clean_activity']:.3f} → {activity_info['noisy_activity']:.3f}")
        
        # Least noise-sensitive layers
        print("\n🔵 LAYERS THAT PRESERVE PATTERNS (low noise sensitivity):")
        for layer_name, sensitivity in insights['layer_rankings']['least_noise_sensitive']:
            activity_info = insights['activity_changes'][layer_name]
            print(f"  • {layer_name}: {sensitivity:.4f} sensitivity")
            print(f"    Activity ratio (noisy/clean): {activity_info['activity_ratio']:.3f}")
        
        # Wind-specific analysis
        if 'most_wind_specific' in insights['layer_rankings']:
            print("\n🌪️ LAYERS WITH STRONGEST WIND-SPECIFIC RESPONSES:")
            for layer_name, wind_spec in insights['layer_rankings']['most_wind_specific']:
                direction = "MORE active" if wind_spec > 0 else "LESS active"
                print(f"  • {layer_name}: {wind_spec:+.4f} ({direction} for wind vs general noise)")
        
        # Architecture analysis
        encoder_sensitivities = [sens for layer, sens in insights['noise_sensitivity'].items() if 'encoder' in layer]
        decoder_sensitivities = [sens for layer, sens in insights['noise_sensitivity'].items() if 'decoder' in layer]
        
        if encoder_sensitivities and decoder_sensitivities:
            encoder_avg = np.mean(encoder_sensitivities)
            decoder_avg = np.mean(decoder_sensitivities)
            
            print(f"\n🏗️ ARCHITECTURE ANALYSIS:")
            print(f"  • Encoder layers avg noise sensitivity: {encoder_avg:.4f}")
            print(f"  • Decoder layers avg noise sensitivity: {decoder_avg:.4f}")
            
            if encoder_avg > decoder_avg * 1.2:
                print("  → Encoders are primary noise processors")
                print("  → Decoders focus on reconstruction")
            elif decoder_avg > encoder_avg * 1.2:
                print("  → Decoders also heavily process noise")
                print("  → Distributed noise handling architecture")
            else:
                print("  → Balanced noise processing across encoder-decoder")
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


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


def analyze_cleanunet_working(model, clean_path, noisy_path, windy_path=None):
    """
    Working CleanUNet analysis using activation patterns
    """
    
    print("🚀 Working CleanUNet Activation Analysis")
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
    analyzer = WorkingCleanUNetAnalyzer(model)
    
    try:
        # Run comparative analysis
        results = analyzer.compare_samples(clean_audio, noisy_audio, windy_audio)
        
        print("\n✅ Analysis complete! Check the plots and insights above.")
        
        return results
        
    finally:
        # Clean up hooks
        analyzer.cleanup()


if __name__ == "__main__":
    print("Working CleanUNet Activation Analyzer")
    print("Uses activation magnitude analysis instead of gradients")
    print("Use analyze_cleanunet_working() function to run analysis")