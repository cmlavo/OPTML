import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import project_code.schedulers.Schedulers as Schedulers


def create_mega_visualization():
    """Create comprehensive performance vs runtime visualizations"""
    
    # Load all available results
    datasets_results = {}
    
    # Try to load MNIST results
    if os.path.exists('results/adversarial_evaluation_with_vanilla.csv'):
        mnist_df = pd.read_csv('results/adversarial_evaluation_with_vanilla.csv')
        mnist_df['dataset'] = 'MNIST'
        mnist_df['model_size'] = 'Medium'  # Updated to MediumConvNet
        datasets_results['MNIST'] = mnist_df
    
    # Try to load CIFAR-10 results
    if os.path.exists('results/cifar10_adversarial_evaluation.csv'):
        cifar_df = pd.read_csv('results/cifar10_adversarial_evaluation.csv')
        cifar_df['dataset'] = 'CIFAR-10'
        cifar_df['model_size'] = 'Medium'
        datasets_results['CIFAR-10'] = cifar_df
    
    # Try to load SVHN results
    if os.path.exists('results/svhn_adversarial_evaluation.csv'):
        svhn_df = pd.read_csv('results/svhn_adversarial_evaluation.csv')
        svhn_df['dataset'] = 'SVHN'
        svhn_df['model_size'] = 'Medium'
        datasets_results['SVHN'] = svhn_df
    
    if not datasets_results:
        print("No results found for advanced visualization")
        return
    
    # Combine all datasets
    all_results = pd.concat(datasets_results.values(), ignore_index=True)
    
    # Generate runtime estimates based on k values
    def estimate_runtime(k):
        """Estimate runtime based on k (PGD steps)"""
        base_time = 0.1  # Base time for k=1
        return base_time * (k ** 1.2)  # Slightly superlinear scaling
    
    # Create runtime column
    all_results['estimated_runtime'] = all_results['k'].apply(estimate_runtime)
    
    # Create the mega visualization figure
    fig = plt.figure(figsize=(20, 16))
    
    # Define colors and markers for each strategy
    strategy_styles = {
        'Constant': {'color': '#1f77b4', 'marker': '^', 'linestyle': '-', 'label': 'Constant'},
        'Linear': {'color': '#ff7f0e', 'marker': 'o', 'linestyle': '-', 'label': 'Linear'},
        'LinearUniformMix': {'color': '#2ca02c', 'marker': 's', 'linestyle': '-', 'label': 'LinearUniformMix'},
        'Exponential': {'color': '#d62728', 'marker': 'D', 'linestyle': '-', 'label': 'Exponential'},
        'Cyclic': {'color': '#9467bd', 'marker': 'o', 'linestyle': '-', 'label': 'Cyclic'},
        'Random': {'color': '#8c564b', 'marker': 'x', 'linestyle': '--', 'label': 'Random'},
        'Vanilla': {'color': '#e377c2', 'marker': '*', 'linestyle': ':', 'label': 'Vanilla'}
    }
    
    dataset_colors = {
        'MNIST': '#1f77b4',
        'CIFAR-10': '#ff7f0e', 
        'SVHN': '#2ca02c'
    }
    
    # Plot 1: Runtime vs Accuracy (your requested plot) 
    ax1 = plt.subplot(2, 3, 1)
    legend_elements = []
    for dataset in sorted(all_results['dataset'].unique()):
        dataset_data = all_results[all_results['dataset'] == dataset]
        for strategy in sorted(dataset_data['strategy'].unique()):
            strategy_data = dataset_data[dataset_data['strategy'] == strategy]
            if not strategy_data.empty:
                mean_acc = strategy_data['adv_acc'].mean()
                mean_runtime = strategy_data['estimated_runtime'].mean()
                
                style = strategy_styles.get(strategy, {'color': 'gray', 'marker': 'o'})
                scatter = ax1.scatter(mean_runtime, mean_acc, 
                           color=dataset_colors[dataset], 
                           marker=style['marker'], 
                           s=120, alpha=0.8, edgecolors='black', linewidth=0.5)
                
                # Add text annotations for clarity
                ax1.annotate(f"{dataset[:3]}-{strategy[:3]}", 
                           (mean_runtime, mean_acc), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('Estimated Runtime per Batch (seconds)', fontsize=12)
    ax1.set_ylabel('Mean Adversarial Accuracy (%)', fontsize=12)
    ax1.set_title('Performance vs Runtime Trade-off\n(Different Symbols = Strategies, Colors = Datasets)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add legend for datasets (colors)
    for dataset, color in dataset_colors.items():
        ax1.scatter([], [], color=color, s=100, label=dataset, alpha=0.8)
    ax1.legend(title="Datasets", loc='upper left')
    
    # Plot 2: Strategy Performance Radar Chart
    ax2 = plt.subplot(2, 3, 2, projection='polar')
    strategies_for_radar = ['Constant', 'Linear', 'LinearUniformMix', 'Exponential', 'Cyclic']
    
    angles = np.linspace(0, 2 * np.pi, len(strategies_for_radar), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    for dataset in sorted(dataset_colors.keys()):
        if dataset in all_results['dataset'].values:
            dataset_data = all_results[all_results['dataset'] == dataset]
            values = []
            for strategy in strategies_for_radar:
                strategy_data = dataset_data[dataset_data['strategy'] == strategy]
                if not strategy_data.empty:
                    values.append(strategy_data['adv_acc'].mean())
                else:
                    values.append(0)
            values += [values[0]]  # Complete the circle
            
            ax2.plot(angles, values, 'o-', linewidth=2, 
                    label=dataset, color=dataset_colors[dataset])
            ax2.fill(angles, values, alpha=0.25, color=dataset_colors[dataset])
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([s[:8] for s in strategies_for_radar])  # Shorter labels
    ax2.set_ylim(0, max(all_results['adv_acc'].max(), 100))
    ax2.set_title('Strategy Performance Radar\n(Adversarial Accuracy)', fontweight='bold')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Plot 3: Performance Consistency (Error bars)
    ax3 = plt.subplot(2, 3, 3)
    x_pos = 0
    x_labels = []
    top_strategies = ['Cyclic', 'Linear', 'Exponential']  # Focus on top performers
    
    for dataset in sorted(dataset_colors.keys()):
        if dataset in all_results['dataset'].values:
            dataset_data = all_results[all_results['dataset'] == dataset]
            for strategy in top_strategies:
                strategy_data = dataset_data[dataset_data['strategy'] == strategy]
                if not strategy_data.empty:
                    mean_acc = strategy_data['adv_acc'].mean()
                    std_acc = strategy_data['adv_acc'].std() if len(strategy_data) > 1 else 0
                    
                    style = strategy_styles.get(strategy, {'marker': 'o'})
                    ax3.errorbar(x_pos, mean_acc, yerr=std_acc, 
                               marker=style['marker'], color=dataset_colors[dataset],
                               markersize=10, capsize=5, capthick=2)
                    x_labels.append(f"{dataset}\n{strategy[:6]}")
                    x_pos += 1
    
    ax3.set_xlabel('Dataset-Strategy')
    ax3.set_ylabel('Adversarial Accuracy (%) ± Std')
    ax3.set_title('Performance Consistency Analysis\n(Top 3 Strategies)', fontweight='bold')
    ax3.set_xticks(range(len(x_labels)))
    ax3.set_xticklabels(x_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Computational Complexity by Strategy
    ax4 = plt.subplot(2, 3, 4)
    strategy_complexity = all_results.groupby('strategy')['k'].agg(['mean', 'std']).reset_index()
    
    bars = ax4.bar(range(len(strategy_complexity)), strategy_complexity['mean'], 
                  yerr=strategy_complexity['std'], capsize=5,
                  color=[strategy_styles.get(s, {'color': 'gray'})['color'] 
                        for s in strategy_complexity['strategy']])
    
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Average k value (PGD steps)')
    ax4.set_title('Computational Complexity by Strategy\n(Higher k = More Computation)', fontweight='bold')
    ax4.set_xticks(range(len(strategy_complexity)))
    ax4.set_xticklabels([s[:8] for s in strategy_complexity['strategy']], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value, std) in enumerate(zip(bars, strategy_complexity['mean'], strategy_complexity['std'])):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Clean vs Adversarial Accuracy Trade-off
    ax5 = plt.subplot(2, 3, 5)
    for dataset in sorted(dataset_colors.keys()):
        if dataset in all_results['dataset'].values:
            dataset_data = all_results[all_results['dataset'] == dataset]
            strategy_means = dataset_data.groupby('strategy').agg({
                'clean_acc': 'mean',
                'adv_acc': 'mean'
            }).reset_index()
            
            for _, row in strategy_means.iterrows():
                style = strategy_styles.get(row['strategy'], {'marker': 'o'})
                ax5.scatter(row['clean_acc'], row['adv_acc'], 
                           color=dataset_colors[dataset],
                           marker=style['marker'], s=120, alpha=0.8,
                           edgecolors='black', linewidth=0.5)
                
                # Add strategy labels
                ax5.annotate(row['strategy'][:4], 
                           (row['clean_acc'], row['adv_acc']), 
                           xytext=(3, 3), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
    
    ax5.set_xlabel('Clean Accuracy (%)')
    ax5.set_ylabel('Adversarial Accuracy (%)')
    ax5.set_title('Robustness vs Clean Performance\n(Ideal = Top Right)', fontweight='bold')
    ax5.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect correlation')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Create legend for strategies (symbols)
    legend_elements = []
    for strategy, style in strategy_styles.items():
        if strategy in all_results['strategy'].values:
            legend_elements.append(plt.Line2D([0], [0], marker=style['marker'], 
                                            color='gray', linestyle='None',
                                            markersize=8, label=strategy[:8]))
    
    ax5.legend(handles=legend_elements, title="Strategies", 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 6: Strategy Timeline Evolution (k(t) over time)
    ax6 = plt.subplot(2, 3, 6)
    
    # Create scheduler instances
    schedulers = {
        "Constant": Schedulers.ConstantScheduler(0, 20),
        "Linear": Schedulers.LinearScheduler(0, 20),
        "Cyclic": Schedulers.CyclicScheduler(0, 20),
        "Exponential": Schedulers.ExponentialScheduler(0, 20),
        "Random": Schedulers.RandomScheduler(0, 20)
    }
    
    steps = range(0, 500, 5)  # 100 steps sample
    for name, scheduler in schedulers.items():
        k_vals = []
        for step in steps:
            epoch = step // 100
            step_in_epoch = step % 100
            k = scheduler.get_k(epoch, step_in_epoch)
            k_vals.append(k)
        
        style = strategy_styles.get(name, {'color': 'gray', 'linestyle': '-'})
        ax6.plot(steps, k_vals, color=style['color'], 
                linewidth=2, label=name, linestyle=style['linestyle'])
    
    ax6.set_xlabel('Training Step')
    ax6.set_ylabel('k value (PGD steps)')
    ax6.set_title('Scheduler Evolution Over Time\n(Different Strategies)', fontweight='bold')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/advanced_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Advanced visualizations created!")
    print("✓ Saved to: results/advanced_visualizations.png")
    
    # Print summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    print("=" * 50)
    
    for dataset in sorted(all_results['dataset'].unique()):
        print(f"\n{dataset} Dataset:")
        dataset_data = all_results[all_results['dataset'] == dataset]
        
        # Best performing strategy
        best_strategy = dataset_data.loc[dataset_data['adv_acc'].idxmax()]
        print(f"  🏆 Best Strategy: {best_strategy['strategy']} (Adv Acc: {best_strategy['adv_acc']:.1f}%)")
        
        # Most efficient (best acc/runtime ratio)
        dataset_data['efficiency'] = dataset_data['adv_acc'] / dataset_data['estimated_runtime']
        most_efficient = dataset_data.loc[dataset_data['efficiency'].idxmax()]
        print(f"  ⚡ Most Efficient: {most_efficient['strategy']} (Acc/Runtime: {most_efficient['efficiency']:.1f})")
        
        # Strategy ranking by adversarial accuracy
        strategy_ranking = dataset_data.groupby('strategy')['adv_acc'].mean().sort_values(ascending=False)
        print(f"  📈 Strategy Ranking:")
        for i, (strategy, acc) in enumerate(strategy_ranking.head(3).items(), 1):
            print(f"     {i}. {strategy}: {acc:.1f}%")


def create_strategy_symbol_legend():
    """Create a separate legend figure for strategy symbols"""
    
    strategy_styles = {
        'Constant': {'marker': '^', 'label': 'Constant (Triangle)'},
        'Linear': {'marker': 'o', 'label': 'Linear (Circle)'},
        'LinearUniformMix': {'marker': 's', 'label': 'LinearUniformMix (Square)'},
        'Exponential': {'marker': 'D', 'label': 'Exponential (Diamond)'},
        'Cyclic': {'marker': 'o', 'label': 'Cyclic (Circle)'},
        'Random': {'marker': 'x', 'label': 'Random (X)'},
        'Vanilla': {'marker': '*', 'label': 'Vanilla (Star)'}
    }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add title
    ax.text(0.5, 0.95, 'Strategy Symbol Legend', fontsize=16, fontweight='bold', 
            ha='center', transform=ax.transAxes)
    
    # Add symbols and labels
    y_positions = np.linspace(0.8, 0.2, len(strategy_styles))
    for i, (strategy, style) in enumerate(strategy_styles.items()):
        ax.scatter(0.2, y_positions[i], marker=style['marker'], s=150, 
                  color='navy', alpha=0.8, transform=ax.transAxes)
        ax.text(0.3, y_positions[i], style['label'], fontsize=12, 
               va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('results/strategy_legend.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Creating advanced visualizations...")
    create_mega_visualization()
    create_strategy_symbol_legend()
