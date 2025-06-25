#!/usr/bin/env python3
"""
Real-time CleanUNet Layer Contribution Visualization

This shows exactly what plots and insights you'll get when passing samples through inference
with Captum LayerLRP analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

# Captum imports
from captum.attr import LayerLRP, LayerGradientXActivation
from captum.attr import visualization as viz

class InferenceLayerContributionVisualizer:
    """
    Real-time visualization of layer contributions during CleanUNet inference
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.eval()
        self.device = device
        
        # Get target layers for analysis
        self.target_layers = self._get_key_layers()
        
        # Initialize LRP analyzers for each layer
        self.lrp_analyzers = {}
        for layer_name, layer_module in self.target_layers.items():
            self.lrp_analyzers[layer_name] = LayerLRP(self.model, layer_module)
        
        # Storage for results
        self.contribution_history = defaultdict(list)
        self.sample_counter = 0
    
    def _get_key_layers(self):
        """Get the most important layers to monitor"""
        layers = {}
        
        # Key encoder layers (early, middle, late processing)
        layers['encoder_0'] = self.model.encoder[0]  # Early noise detection
        layers['encoder_3'] = self.model.encoder[3]  # Middle processing  
        layers['encoder_6'] = self.model.encoder[6]  # Late encoding
        
        # Transformer (temporal modeling)
        layers['transformer'] = self.model.tsfm_encoder
        
        # Key decoder layers
        layers['decoder_0'] = self.model.decoder[0]  # Early reconstruction
        layers['decoder_3'] = self.model.decoder[3]  # Middle reconstruction
        layers['decoder_6'] = self.model.decoder[6]  # Final output
        
        return layers
    
    def analyze_single_sample(self, audio_input, sample_type="unknown"):
        """
        Analyze a single audio sample and return layer contributions
        
        Args:
            audio_input: input audio tensor (1, length)
            sample_type: type of sample ("clean", "noisy", "windy", etc.)
        
        Returns:
            dict: layer contribution scores and visualizations
        """
        
        audio_input = audio_input.to(self.device).requires_grad_(True)
        
        # Define target function (noise reduction capability)
        def noise_reduction_target(x):
            output = self.model(x)
            # Measure how much the model changed the input (proxy for denoising)
            input_energy = (x ** 2).mean()
            output_energy = (output ** 2).mean()
            return input_energy - output_energy  # Positive = noise was removed
        
        # Compute layer contributions
        layer_contributions = {}
        layer_attributions = {}
        
        print(f"Analyzing {sample_type} sample...")
        
        for layer_name, lrp_analyzer in self.lrp_analyzers.items():
            try:
                # Get layer attribution
                attribution = lrp_analyzer.attribute(
                    audio_input,
                    target=noise_reduction_target
                )
                
                # Compute contribution score (mean absolute attribution)
                contribution_score = torch.abs(attribution).mean().item()
                
                layer_contributions[layer_name] = contribution_score
                layer_attributions[layer_name] = attribution.detach().cpu()
                
                # Store in history
                self.contribution_history[layer_name].append({
                    'sample_id': self.sample_counter,
                    'sample_type': sample_type,
                    'contribution': contribution_score,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                print(f"Error analyzing {layer_name}: {e}")
                continue
        
        self.sample_counter += 1
        
        return {
            'layer_contributions': layer_contributions,
            'layer_attributions': layer_attributions,
            'sample_type': sample_type,
            'sample_id': self.sample_counter - 1
        }
    
    def create_realtime_contribution_plot(self, results):
        """
        Create real-time plot showing layer contributions for current sample
        
        Returns:
            matplotlib figure showing layer contributions
        """
        
        contributions = results['layer_contributions']
        sample_type = results['sample_type']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Layer Contributions - {sample_type.upper()} Sample #{results["sample_id"]}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Bar plot of current contributions
        layers = list(contributions.keys())
        scores = list(contributions.values())
        
        # Color-code layers by type
        colors = []
        for layer in layers:
            if 'encoder' in layer:
                colors.append('#FF6B6B')  # Red for encoders
            elif 'transformer' in layer:
                colors.append('#4ECDC4')  # Teal for transformer
            elif 'decoder' in layer:
                colors.append('#45B7D1')  # Blue for decoders
            else:
                colors.append('#96CEB4')  # Green for others
        
        bars = axes[0,0].bar(layers, scores, color=colors, alpha=0.8)
        axes[0,0].set_xlabel('Layer')
        axes[0,0].set_ylabel('Contribution Score')
        axes[0,0].set_title('Current Sample - Layer Contributions')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                          f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Top contributing layers (sorted)
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_layers = sorted_contributions[:5]
        
        top_names = [item[0] for item in top_layers]
        top_scores = [item[1] for item in top_layers]
        
        axes[0,1].barh(top_names, top_scores, color='orange', alpha=0.7)
        axes[0,1].set_xlabel('Contribution Score')
        axes[0,1].set_title('Top 5 Contributing Layers')
        axes[0,1].grid(True, alpha=0.3)
        
        # Add score labels
        for i, score in enumerate(top_scores):
            axes[0,1].text(score + 0.001, i, f'{score:.3f}', 
                          va='center', fontsize=9)
        
        # 3. Attribution heatmap for top layer
        if top_layers:
            top_layer_name = top_layers[0][0]
            top_attribution = results['layer_attributions'][top_layer_name][0]  # First sample
            
            # Average across channels if multi-dimensional
            if len(top_attribution.shape) > 1:
                top_attribution = top_attribution.mean(dim=0)
            
            # Create heatmap
            attribution_2d = top_attribution.unsqueeze(0).numpy()
            im = axes[1,0].imshow(attribution_2d, cmap='RdBu_r', aspect='auto')
            axes[1,0].set_xlabel('Time Steps')
            axes[1,0].set_ylabel('Channel')
            axes[1,0].set_title(f'Attribution Heatmap - {top_layer_name}')
            plt.colorbar(im, ax=axes[1,0])
        
        # 4. Layer type analysis (encoder vs decoder vs transformer)
        layer_type_contributions = defaultdict(list)
        for layer, score in contributions.items():
            if 'encoder' in layer:
                layer_type_contributions['Encoder'].append(score)
            elif 'decoder' in layer:
                layer_type_contributions['Decoder'].append(score)
            elif 'transformer' in layer:
                layer_type_contributions['Transformer'].append(score)
        
        type_avg_scores = {k: np.mean(v) for k, v in layer_type_contributions.items()}
        
        wedges, texts, autotexts = axes[1,1].pie(
            type_avg_scores.values(), 
            labels=type_avg_scores.keys(),
            autopct='%1.2f',
            colors=['#FF6B6B', '#45B7D1', '#4ECDC4'],
            startangle=90
        )
        axes[1,1].set_title('Contribution by Layer Type')
        
        plt.tight_layout()
        return fig
    
    def create_comparative_analysis_plot(self, clean_results, noisy_results, windy_results=None):
        """
        Create comparative plot showing how contributions differ across sample types
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparative Layer Contribution Analysis', fontsize=16, fontweight='bold')
        
        # Get common layers
        layers = list(clean_results['layer_contributions'].keys())
        
        clean_scores = [clean_results['layer_contributions'][layer] for layer in layers]
        noisy_scores = [noisy_results['layer_contributions'][layer] for layer in layers]
        windy_scores = [windy_results['layer_contributions'][layer] for layer in layers] if windy_results else None
        
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
        axes[0,1].set_title('Noise-Specific Layer Contributions')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 3. Wind specificity (if available)
        if windy_scores:
            wind_specificity = [windy - noisy for windy, noisy in zip(windy_scores, noisy_scores)]
            colors = ['purple' if x > 0 else 'orange' for x in wind_specificity]
            
            axes[1,0].bar(layers, wind_specificity, color=colors, alpha=0.7)
            axes[1,0].set_xlabel('Layer')
            axes[1,0].set_ylabel('Wind Specificity (Windy - General Noisy)')
            axes[1,0].set_title('Wind-Specific Layer Contributions')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 4. Correlation heatmap
        correlation_data = np.array([clean_scores, noisy_scores])
        if windy_scores:
            correlation_data = np.vstack([correlation_data, windy_scores])
        
        correlation_matrix = np.corrcoef(correlation_data)
        labels = ['Clean', 'Noisy', 'Windy'] if windy_scores else ['Clean', 'Noisy']
        
        im = axes[1,1].imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1,1].set_xticks(range(len(labels)))
        axes[1,1].set_yticks(range(len(labels)))
        axes[1,1].set_xticklabels(labels)
        axes[1,1].set_yticklabels(labels)
        axes[1,1].set_title('Sample Type Correlation')
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = axes[1,1].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=axes[1,1])
        plt.tight_layout()
        return fig
    
    def analyze_and_visualize_sample(self, audio_input, sample_type="unknown", show_plots=True):
        """
        Complete analysis and visualization for a single sample
        
        Args:
            audio_input: input audio tensor
            sample_type: type of sample for labeling
            show_plots: whether to display plots immediately
        
        Returns:
            analysis results and matplotlib figure
        """
        
        # Analyze the sample
        results = self.analyze_single_sample(audio_input, sample_type)
        
        # Create visualization
        fig = self.create_realtime_contribution_plot(results)
        
        if show_plots:
            plt.show()
        
        # Print key insights
        contributions = results['layer_contributions']
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🎯 KEY INSIGHTS for {sample_type.upper()} sample:")
        print("="*50)
        print(f"Top 3 contributing layers:")
        for i, (layer, score) in enumerate(sorted_contributions[:3]):
            print(f"  {i+1}. {layer}: {score:.4f}")
        
        # Identify layer roles
        encoder_avg = np.mean([score for layer, score in contributions.items() if 'encoder' in layer])
        decoder_avg = np.mean([score for layer, score in contributions.items() if 'decoder' in layer])
        
        print(f"\nLayer type analysis:")
        print(f"  Encoder avg contribution: {encoder_avg:.4f}")
        print(f"  Decoder avg contribution: {decoder_avg:.4f}")
        
        if encoder_avg > decoder_avg * 1.2:
            print("  → Encoders dominate (noise removal focus)")
        elif decoder_avg > encoder_avg * 1.2:
            print("  → Decoders dominate (reconstruction focus)")
        else:
            print("  → Balanced encoder-decoder processing")
        
        return results, fig
    
    def batch_analysis_with_comparison(self, clean_sample, noisy_sample, windy_sample=None):
        """
        Analyze multiple samples and create comparative visualizations
        """
        
        print("🔬 Running batch analysis with comparisons...")
        
        # Analyze each sample
        clean_results = self.analyze_single_sample(clean_sample, "clean")
        noisy_results = self.analyze_single_sample(noisy_sample, "noisy")
        windy_results = self.analyze_single_sample(windy_sample, "windy") if windy_sample is not None else None
        
        # Create comparative plot
        comparison_fig = self.create_comparative_analysis_plot(
            clean_results, noisy_results, windy_results
        )
        
        plt.show()
        
        # Print comparative insights
        self._print_comparative_insights(clean_results, noisy_results, windy_results)
        
        return clean_results, noisy_results, windy_results, comparison_fig
    
    def _print_comparative_insights(self, clean_results, noisy_results, windy_results=None):
        """Print key comparative insights"""
        
        print(f"\n🔍 COMPARATIVE ANALYSIS INSIGHTS:")
        print("="*50)
        
        # Find layers that contribute most to noise removal
        clean_contrib = clean_results['layer_contributions']
        noisy_contrib = noisy_results['layer_contributions']
        
        noise_specificity = {layer: noisy_contrib[layer] - clean_contrib[layer] 
                           for layer in clean_contrib.keys()}
        
        sorted_noise_spec = sorted(noise_specificity.items(), key=lambda x: x[1], reverse=True)
        
        print("Layers most responsible for noise removal:")
        for layer, specificity in sorted_noise_spec[:3]:
            print(f"  • {layer}: +{specificity:.4f} (noise - clean)")
        
        print("\nLayers that preserve clean speech:")
        for layer, specificity in sorted_noise_spec[-3:]:
            if specificity < 0:
                print(f"  • {layer}: {specificity:.4f} (maintains clean patterns)")
        
        # Wind-specific analysis
        if windy_results:
            windy_contrib = windy_results['layer_contributions']
            wind_specificity = {layer: windy_contrib[layer] - noisy_contrib[layer]
                              for layer in noisy_contrib.keys()}
            
            sorted_wind_spec = sorted(wind_specificity.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print(f"\nWind-specific layer responses:")
            for layer, specificity in sorted_wind_spec[:3]:
                direction = "handles more wind" if specificity > 0 else "handles less wind"
                print(f"  • {layer}: {specificity:+.4f} ({direction} than general noise)")


# Example usage function
def demo_inference_analysis():
    """
    Demo showing exactly what you'll see during inference analysis
    """
    
    print("🚀 CleanUNet Inference Layer Contribution Demo")
    print("="*60)
    
    # Load your model (you'll replace this with your actual model loading)
    # from network import CleanUNet
    # model = CleanUNet(**your_config)
    # model.load_state_dict(torch.load('your_checkpoint.pkl')['model_state_dict'])
    # model.eval()
    
    # For demo purposes, we'll simulate what you would see
    print("📊 What you'll see when analyzing samples:")
    print("\n1. REAL-TIME LAYER CONTRIBUTIONS:")
    print("   - Bar chart showing which layers contribute most to current sample")
    print("   - Color-coded by layer type (encoder=red, transformer=teal, decoder=blue)")
    print("   - Numerical contribution scores for each layer")
    
    print("\n2. TOP CONTRIBUTING LAYERS:")
    print("   - Ranked list of most important layers for current sample")
    print("   - Quantitative scores (e.g., encoder_3: 0.847)")
    
    print("\n3. ATTRIBUTION HEATMAP:")
    print("   - Time-series visualization of how the top layer contributes")
    print("   - Shows when in the audio the layer is most active")
    
    print("\n4. LAYER TYPE ANALYSIS:")
    print("   - Pie chart showing encoder vs decoder vs transformer contributions")
    print("   - Reveals the overall processing strategy")
    
    print("\n5. COMPARATIVE ANALYSIS (multiple samples):")
    print("   - Side-by-side comparison of clean vs noisy vs windy")
    print("   - Noise specificity scores showing wind-handling layers")
    print("   - Correlation analysis between different sample types")
    
    print("\n✨ EXAMPLE OUTPUT:")
    print("🎯 KEY INSIGHTS for WINDY sample:")
    print("="*30)
    print("Top 3 contributing layers:")
    print("  1. encoder_3: 0.847")
    print("  2. encoder_4: 0.692") 
    print("  3. transformer: 0.534")
    print("")
    print("Layer type analysis:")
    print("  Encoder avg contribution: 0.723")
    print("  Decoder avg contribution: 0.421")
    print("  → Encoders dominate (noise removal focus)")
    print("")
    print("Wind-specific responses:")
    print("  • encoder_3: +0.234 (handles more wind than general noise)")
    print("  • encoder_4: +0.187 (wind specialist layer)")
    print("  • decoder_1: -0.089 (preserves speech during wind removal)")


if __name__ == "__main__":
    demo_inference_analysis()
    
    print("\n" + "="*60)
    print("TO USE WITH YOUR MODEL:")
    print("="*60)
    print("""
# 1. Initialize the visualizer with your model
visualizer = InferenceLayerContributionVisualizer(your_model)

# 2. Analyze a single sample (gets immediate plots)
results, fig = visualizer.analyze_and_visualize_sample(
    audio_input=your_audio_tensor,
    sample_type="windy",
    show_plots=True
)

# 3. Compare multiple samples
clean_results, noisy_results, windy_results, comparison_fig = visualizer.batch_analysis_with_comparison(
    clean_sample=clean_audio,
    noisy_sample=noisy_audio, 
    windy_sample=windy_audio
)

# You'll get:
# - Real-time contribution scores
# - Layer importance rankings  
# - Attribution heatmaps
# - Comparative analysis plots
# - Quantitative insights about which layers handle wind noise
""")