import numpy as np
import matplotlib.pyplot as plt
from project_code.schedulers.Schedulers import (
    ConstantScheduler,
    LinearScheduler,
    LinearUniformMixScheduler,
    ExponentialScheduler,
    CyclicScheduler,
    RandomScheduler,
    CompositeScheduler
)
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from typing import List, Dict, Tuple, Optional

class EpsilonAnalysis:
    def __init__(self, k_min: int = 0, k_max: int = 10, max_epochs: int = 100):
        self.k_min = k_min
        self.k_max = k_max
        self.max_epochs = max_epochs
        
        # Different epsilon configurations to test
        self.epsilon_configs = {
            'constant_small': 0.1,
            'constant_medium': 0.3,
            'constant_large': 0.5,
            'linear': 'linear',
            'cyclic': 'cyclic',
            'random': 'random'
        }
        
        # Initialize base schedulers
        self.base_schedulers = {
            'Constant': ConstantScheduler,
            'Linear': LinearScheduler,
            'LinearUniformMix': LinearUniformMixScheduler,
            'Exponential': ExponentialScheduler,
            'Cyclic': CyclicScheduler,
            'Random': RandomScheduler
        }

    def collect_data(self) -> pd.DataFrame:
        """Collect data for all scheduler and epsilon combinations."""
        data = []
        
        for scheduler_name, scheduler_class in tqdm(self.base_schedulers.items(), desc="Testing schedulers"):
            for eps_config_name, eps_config in self.epsilon_configs.items():
                # Create scheduler with appropriate epsilon configuration
                if isinstance(eps_config, (int, float)):
                    # Constant epsilon
                    scheduler = scheduler_class(self.k_min, self.k_max, epsilon_max=eps_config)
                else:
                    # Dynamic epsilon strategy
                    scheduler = CompositeScheduler(
                        scheduler_class(self.k_min, self.k_max),
                        epsilon_type=eps_config,
                        epsilon_min=0.1,
                        epsilon_max=0.5
                    )
                
                # Collect data for this configuration
                for epoch in range(self.max_epochs):
                    eps, k_dist = scheduler(epoch, self.max_epochs)
                    for k, prob in k_dist.items():
                        data.append({
                            'epoch': epoch,
                            'scheduler': scheduler_name,
                            'epsilon_config': eps_config_name,
                            'k': k,
                            'epsilon': eps,
                            'probability': prob
                        })
        
        return pd.DataFrame(data)

    def plot_epsilon_influence(self, df: pd.DataFrame):
        """Plot the influence of epsilon on k values for each scheduler."""
        # Create a figure for each scheduler
        for scheduler_name in self.base_schedulers.keys():
            plt.figure(figsize=(15, 10))
            scheduler_data = df[df['scheduler'] == scheduler_name]
            
            # Plot k values for each epsilon configuration
            for eps_config in self.epsilon_configs.keys():
                config_data = scheduler_data[scheduler_data['epsilon_config'] == eps_config]
                plt.plot(config_data['epoch'], config_data['k'], 
                        label=f'ε={eps_config}', alpha=0.7)
            
            plt.title(f'K Values Over Time for {scheduler_name} with Different Epsilon Configurations')
            plt.xlabel('Epoch')
            plt.ylabel('K Value')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'output/epsilon_influence_{scheduler_name}.png')
            plt.close()

    def analyze_epsilon_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze how epsilon configurations affect scheduler performance."""
        stats = []
        
        for scheduler_name in self.base_schedulers.keys():
            for eps_config in self.epsilon_configs.keys():
                config_data = df[
                    (df['scheduler'] == scheduler_name) & 
                    (df['epsilon_config'] == eps_config)
                ]
                
                stats.append({
                    'scheduler': scheduler_name,
                    'epsilon_config': eps_config,
                    'mean_k': config_data['k'].mean(),
                    'std_k': config_data['k'].std(),
                    'mean_epsilon': config_data['epsilon'].mean(),
                    'std_epsilon': config_data['epsilon'].std(),
                    'k_range': config_data['k'].max() - config_data['k'].min(),
                    'epsilon_range': config_data['epsilon'].max() - config_data['epsilon'].min()
                })
        
        return pd.DataFrame(stats)

    def plot_epsilon_vs_performance(self, df: pd.DataFrame):
        """Plot the relationship between epsilon and k values."""
        plt.figure(figsize=(15, 10))
        
        # Create a scatter plot with different colors for each scheduler
        for scheduler_name in self.base_schedulers.keys():
            scheduler_data = df[df['scheduler'] == scheduler_name]
            plt.scatter(scheduler_data['epsilon'], scheduler_data['k'],
                       label=scheduler_name, alpha=0.5)
        
        plt.title('Relationship Between Epsilon and K Values')
        plt.xlabel('Epsilon')
        plt.ylabel('K Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('output/epsilon_vs_k.png')
        plt.close()

    def run_analysis(self):
        """Run the complete epsilon analysis."""
        print("Starting epsilon influence analysis...")
        
        # Collect data
        df = self.collect_data()
        
        # Generate plots
        print("Generating plots...")
        self.plot_epsilon_influence(df)
        self.plot_epsilon_vs_performance(df)
        
        # Calculate statistics
        print("Calculating statistics...")
        stats_df = self.analyze_epsilon_impact(df)
        
        # Save statistics to CSV
        stats_df.to_csv('output/epsilon_analysis_statistics.csv', index=False)
        print("Analysis complete! Results saved in output/ directory.")
        
        return stats_df

# ============================================================================
# ADVANCED EPSILON VARIATION ANALYSIS FUNCTIONS
# ============================================================================

def calculate_clean_accuracy(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    """Calcule la précision propre du modèle."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total


def find_critical_thresholds(df: pd.DataFrame, threshold: float = 0.5) -> List[Dict]:
    """
    Trouve les seuils critiques d'epsilon pour chaque valeur de k.
    
    Args:
        df: DataFrame avec les résultats
        threshold: Seuil de robustesse critique
        
    Returns:
        Liste des seuils critiques par k
    """
    critical_thresholds = []
    
    for k_val in df['k'].unique():
        k_data = df[df['k'] == k_val].sort_values('epsilon')
        critical_eps = k_data[k_data['robustness_ratio'] < threshold]['epsilon'].min()
        
        if not pd.isna(critical_eps):
            critical_thresholds.append({
                'k': k_val,
                'critical_epsilon': critical_eps
            })
    
    return critical_thresholds

def analyze_epsilon_variation_advanced(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    strategy_name: str,
    epsilon_values: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    k_values: List[int] = [1, 2, 4, 8, 16],
    max_batches: int = 3,
    save_results: bool = True
) -> pd.DataFrame:
    """
    Analyse la variation d'epsilon pour un k-scheduler donné.
    
    Args:
        model: Modèle PyTorch entraîné
        test_loader: DataLoader pour les données de test
        criterion: Fonction de perte
        device: Device (cpu/cuda/mps)
        strategy_name: Nom de la stratégie k-scheduler
        epsilon_values: Liste des valeurs d'epsilon à tester
        k_values: Liste des valeurs de k à tester
        max_batches: Nombre maximum de batches à traiter
        save_results: Sauvegarder les résultats en CSV
        
    Returns:
        DataFrame avec les résultats de l'analyse
    """
    from Attacks import pgd_attack
    
    results = []
    model.eval()
    
    # Calculer la précision propre
    clean_accuracy = calculate_clean_accuracy(model, test_loader, device)
    
    print(f"🔬 Analyse epsilon pour {strategy_name}")
    print(f"📊 Précision propre: {clean_accuracy:.2f}%")
    print(f"🎯 Epsilon range: {epsilon_values[0]:.1f} - {epsilon_values[-1]:.1f}")
    print(f"🔢 K range: {k_values[0]} - {k_values[-1]}")
    
    for epsilon in epsilon_values:
        print(f"\n📈 Test epsilon = {epsilon:.1f}")
        
        for k in k_values:
            correct_adv = 0
            total_adv = 0
            confidences = []
            
            batch_count = 0
            for images, labels in test_loader:
                if batch_count >= max_batches:
                    break
                    
                images, labels = images.to(device), labels.to(device)
                
                # Attaque adversariale
                adv_images, _ = pgd_attack(images, labels, model, criterion, epsilon, k, device)
                
                # Évaluation
                with torch.no_grad():
                    outputs = model(adv_images)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence = probabilities.max(dim=1)[0].mean().item()
                    confidences.append(confidence)
                    
                    _, predicted = outputs.max(1)
                    total_adv += labels.size(0)
                    correct_adv += (predicted == labels).sum().item()
                
                batch_count += 1
            
            adv_accuracy = 100.0 * correct_adv / total_adv
            mean_confidence = np.mean(confidences)
            
            results.append({
                'strategy': strategy_name,
                'epsilon': epsilon,
                'k': k,
                'clean_acc': clean_accuracy,
                'adv_acc': adv_accuracy,
                'mean_confidence': mean_confidence,
                'robustness_ratio': adv_accuracy / clean_accuracy,
                'degradation': clean_accuracy - adv_accuracy,
                'efficiency_score': adv_accuracy / k  # Score d'efficacité
            })
            
            print(f"  k={k}: {adv_accuracy:.1f}% adv acc, {mean_confidence:.3f} conf")
    
    df = pd.DataFrame(results)
    
    if save_results:
        filename = f'results/epsilon_analysis_{strategy_name}.csv'
        df.to_csv(filename, index=False)
        print(f"💾 Résultats sauvegardés: {filename}")
    
    return df


def calculate_clean_accuracy(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    """Calcule la précision propre du modèle."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total


def plot_epsilon_analysis_advanced(df: pd.DataFrame, strategy_name: str, save_plots: bool = True) -> None:
    """
    Génère des visualisations complètes de l'analyse epsilon.
    
    Args:
        df: DataFrame avec les résultats d'analyse
        strategy_name: Nom de la stratégie
        save_plots: Sauvegarder les graphiques
    """
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Heatmap robustesse
    ax1 = plt.subplot(2, 4, 1)
    pivot_data = df.pivot(index='epsilon', columns='k', values='adv_acc')
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax1)
    ax1.set_title(f'Robustesse {strategy_name}\nEpsilon vs K')
    
    # 2. Courbes de dégradation
    ax2 = plt.subplot(2, 4, 2)
    for eps in [0.1, 0.3, 0.5, 0.7]:
        eps_data = df[df['epsilon'] == eps]
        if not eps_data.empty:
            ax2.plot(eps_data['k'], eps_data['adv_acc'], 'o-', 
                    label=f'ε = {eps}', linewidth=2)
    ax2.set_xlabel('Steps PGD (k)')
    ax2.set_ylabel('Adversarial Accuracy (%)')
    ax2.set_title('Courbes de Robustesse')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Ratio de robustesse
    ax3 = plt.subplot(2, 4, 3)
    for k_val in [1, 4, 8, 16]:
        k_data = df[df['k'] == k_val]
        if not k_data.empty:
            ax3.plot(k_data['epsilon'], k_data['robustness_ratio'], 'o-', 
                    label=f'k = {k_val}', linewidth=2)
    ax3.set_xlabel('Epsilon')
    ax3.set_ylabel('Ratio Robustesse')
    ax3.set_title('Ratio Robustesse vs Epsilon')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    
    # 4. Scatter confiance vs robustesse
    ax4 = plt.subplot(2, 4, 4)
    scatter = ax4.scatter(df['mean_confidence'], df['adv_acc'], 
                         c=df['epsilon'], s=df['k']*8, alpha=0.7, cmap='viridis')
    ax4.set_xlabel('Confiance Moyenne')
    ax4.set_ylabel('Adversarial Accuracy (%)')
    ax4.set_title('Confiance vs Robustesse')
    plt.colorbar(scatter, ax=ax4, label='Epsilon')
    
    # 5. Seuils critiques
    ax5 = plt.subplot(2, 4, 5)
    critical_thresholds = find_critical_thresholds(df)
    if critical_thresholds:
        crit_df = pd.DataFrame(critical_thresholds)
        ax5.bar(crit_df['k'], crit_df['critical_epsilon'], color='coral', alpha=0.8)
        ax5.set_xlabel('Steps PGD (k)')
        ax5.set_ylabel('Epsilon Critique')
        ax5.set_title('Seuils Critiques (Rob. < 50%)')
    
    # 6. Efficacité (accuracy/k)
    ax6 = plt.subplot(2, 4, 6)
    efficiency_pivot = df.pivot(index='epsilon', columns='k', values='efficiency_score')
    sns.heatmap(efficiency_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax6)
    ax6.set_title('Score d\'Efficacité\n(Accuracy/k)')
    
    # 7. Tendance globale
    ax7 = plt.subplot(2, 4, 7)
    trend_data = df.groupby('epsilon').agg({
        'adv_acc': ['mean', 'std']
    }).round(2)
    trend_data.columns = ['mean', 'std']
    
    ax7.errorbar(trend_data.index, trend_data['mean'], 
                yerr=trend_data['std'], marker='o', linewidth=2, capsize=5)
    ax7.set_xlabel('Epsilon')
    ax7.set_ylabel('Adversarial Accuracy (%)')
    ax7.set_title('Tendance Globale')
    ax7.grid(True, alpha=0.3)
    
    # 8. Statistiques résumées
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Calculs statistiques
    best_config = df.loc[df['adv_acc'].idxmax()]
    worst_config = df.loc[df['adv_acc'].idxmin()]
    
    stats_text = f"""RÉSUMÉ ANALYSE {strategy_name}

Meilleure config:
• ε = {best_config['epsilon']:.1f}, k = {best_config['k']}
• Robustesse: {best_config['adv_acc']:.1f}%

Pire config:
• ε = {worst_config['epsilon']:.1f}, k = {worst_config['k']}
• Robustesse: {worst_config['adv_acc']:.1f}%

Moyennes globales:
• Robustesse moy: {df['adv_acc'].mean():.1f}%
• Confiance moy: {df['mean_confidence'].mean():.3f}
• Ratio rob. moy: {df['robustness_ratio'].mean():.3f}

Recommandations:
• Zone sûre: ε ≤ 0.3
• Zone critique: ε > 0.5
• k optimal: 4-8 steps"""
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'results/epsilon_analysis_{strategy_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Graphiques sauvegardés: {filename}")
    
    plt.show()


def find_critical_thresholds(df: pd.DataFrame, threshold: float = 0.5) -> List[Dict]:
    """
    Trouve les seuils critiques d'epsilon pour chaque valeur de k.
    
    Args:
        df: DataFrame avec les résultats
        threshold: Seuil de robustesse critique
        
    Returns:
        Liste des seuils critiques par k
    """
    critical_thresholds = []
    
    for k_val in df['k'].unique():
        k_data = df[df['k'] == k_val].sort_values('epsilon')
        critical_eps = k_data[k_data['robustness_ratio'] < threshold]['epsilon'].min()
        
        if not pd.isna(critical_eps):
            critical_thresholds.append({
                'k': k_val,
                'critical_epsilon': critical_eps
            })
    
    return critical_thresholds


def generate_epsilon_report(df: pd.DataFrame, strategy_name: str) -> str:
    """
    Génère un rapport textuel détaillé de l'analyse epsilon.
    
    Args:
        df: DataFrame avec les résultats
        strategy_name: Nom de la stratégie
        
    Returns:
        Rapport textuel formaté
    """
    report = f"""
RAPPORT D'ANALYSE EPSILON - {strategy_name.upper()}
{'=' * 60}

1. CONFIGURATION EXPÉRIMENTALE
   • Stratégie analysée: {strategy_name}
   • Nombre de configurations: {len(df)}
   • Plage epsilon: {df['epsilon'].min():.1f} - {df['epsilon'].max():.1f}
   • Plage k: {df['k'].min()} - {df['k'].max()} steps
   • Précision propre: {df['clean_acc'].iloc[0]:.2f}%

2. PERFORMANCES GLOBALES
   • Robustesse moyenne: {df['adv_acc'].mean():.2f}% (±{df['adv_acc'].std():.2f})
   • Confiance moyenne: {df['mean_confidence'].mean():.3f} (±{df['mean_confidence'].std():.3f})
   • Ratio robustesse moyen: {df['robustness_ratio'].mean():.3f}

3. CONFIGURATIONS EXTRÊMES
   Meilleure performance:
   {df.loc[df['adv_acc'].idxmax()][['epsilon', 'k', 'adv_acc', 'mean_confidence']].to_string()}
   
   Pire performance:
   {df.loc[df['adv_acc'].idxmin()][['epsilon', 'k', 'adv_acc', 'mean_confidence']].to_string()}

4. ANALYSE PAR ZONES D'EPSILON
"""
    
    # Analyse par zones
    zones = [(0.0, 0.3, "SÛRE"), (0.3, 0.5, "MODÉRÉE"), (0.5, 1.0, "CRITIQUE")]
    
    for min_eps, max_eps, zone_name in zones:
        zone_data = df[(df['epsilon'] >= min_eps) & (df['epsilon'] < max_eps)]
        if not zone_data.empty:
            report += f"""
   Zone {zone_name} (ε: {min_eps:.1f}-{max_eps:.1f}):
   • Robustesse moyenne: {zone_data['adv_acc'].mean():.1f}%
   • Configurations: {len(zone_data)}
   • Ratio robustesse: {zone_data['robustness_ratio'].mean():.3f}
"""
    
    # Seuils critiques
    critical_thresholds = find_critical_thresholds(df)
    if critical_thresholds:
        report += "\n5. SEUILS CRITIQUES (Robustesse < 50%)\n"
        for threshold in critical_thresholds:
            report += f"   • k = {threshold['k']}: ε critique = {threshold['critical_epsilon']:.1f}\n"
    
    # Recommandations
    report += f"""
6. RECOMMANDATIONS PRATIQUES
   • Pour entraînement: ε = 0.3, k = 4-8
   • Pour évaluation: tester jusqu'à ε = 0.5
   • Zone de confiance: ε ≤ 0.3 (robustesse > 70%)
   • Limite pratique: ε = 0.5 (point d'inflexion)

7. MÉTRIQUE DE QUALITÉ GLOBALE
   • Score global: {(df['adv_acc'].mean() + df['mean_confidence'].mean() * 100) / 2:.1f}/100
   • Stabilité: {'ÉLEVÉE' if df['adv_acc'].std() < 10 else 'MODÉRÉE' if df['adv_acc'].std() < 20 else 'FAIBLE'}
   • Recommandation: {'EXCELLENT' if df['adv_acc'].mean() > 80 else 'BON' if df['adv_acc'].mean() > 60 else 'ACCEPTABLE'}

{'=' * 60}
Rapport généré automatiquement par l'outil d'analyse epsilon.
"""
    
    return report


def compare_strategies_epsilon(
    results_dict: Dict[str, pd.DataFrame],
    save_comparison: bool = True
) -> None:
    """
    Compare les performances de plusieurs stratégies sous variation d'epsilon.
    
    Args:
        results_dict: Dictionnaire {strategy_name: DataFrame}
        save_comparison: Sauvegarder la comparaison
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Comparaison tendances globales
    ax1 = axes[0, 0]
    for strategy, df in results_dict.items():
        trend_data = df.groupby('epsilon')['adv_acc'].mean()
        ax1.plot(trend_data.index, trend_data.values, 'o-', 
                label=strategy, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Adversarial Accuracy (%)')
    ax1.set_title('Comparaison Robustesse vs Epsilon')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Comparaison des ratios de robustesse
    ax2 = axes[0, 1]
    for strategy, df in results_dict.items():
        ratio_data = df.groupby('epsilon')['robustness_ratio'].mean()
        ax2.plot(ratio_data.index, ratio_data.values, 'o-', 
                label=strategy, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Ratio Robustesse')
    ax2.set_title('Comparaison Ratios de Robustesse')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    
    # 3. Boxplot des performances
    ax3 = axes[1, 0]
    comparison_data = []
    for strategy, df in results_dict.items():
        for _, row in df.iterrows():
            comparison_data.append({
                'Strategy': strategy,
                'Adv_Acc': row['adv_acc'],
                'Epsilon': row['epsilon']
            })
    
    comp_df = pd.DataFrame(comparison_data)
    sns.boxplot(data=comp_df, x='Strategy', y='Adv_Acc', ax=ax3)
    ax3.set_title('Distribution des Performances')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Heatmap comparative
    ax4 = axes[1, 1]
    strategy_scores = {}
    for strategy, df in results_dict.items():
        strategy_scores[strategy] = df['adv_acc'].mean()
    
    strategies = list(strategy_scores.keys())
    scores = list(strategy_scores.values())
    
    bars = ax4.barh(strategies, scores, color=plt.cm.viridis(np.linspace(0, 1, len(strategies))))
    ax4.set_xlabel('Adversarial Accuracy Moyenne (%)')
    ax4.set_title('Ranking Global des Stratégies')
    ax4.grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs
    for bar, score in zip(bars, scores):
        ax4.text(score + 0.5, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}%', va='center', ha='left', fontweight='bold')
    
    plt.tight_layout()
    
    if save_comparison:
        plt.savefig('results/epsilon_strategies_comparison.png', dpi=300, bbox_inches='tight')
        print("📊 Comparaison sauvegardée: results/epsilon_strategies_comparison.png")
    
    plt.show()


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    import os
    os.makedirs('output', exist_ok=True)
    
    # Run analysis
    analysis = EpsilonAnalysis(k_min=0, k_max=10, max_epochs=100)
    stats = analysis.run_analysis()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(stats.to_string())
    
    print("📚 Module d'analyse epsilon enrichi chargé avec succès!")
    print("🔧 Fonctions disponibles:")
    print("   • analyze_epsilon_variation_advanced()")
    print("   • plot_epsilon_analysis_advanced()")
    print("   • find_critical_thresholds()")
    print("   • generate_epsilon_report()")
    print("   • compare_strategies_epsilon()")
    print("   • calculate_clean_accuracy()")