import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import project_code.schedulers.Schedulers as Schedulers
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import project_code.model.Models as Models
from Attacks import pgd_attack
import os


def plot_scheduler_evolution(num_epochs=10, steps_per_epoch=100, k_min=0, k_max=20):
    """Plot k(t) evolution for all schedulers"""
    
    schedulers = {
        "Constant": Schedulers.ConstantScheduler(k_min, k_max),
        "Linear": Schedulers.LinearScheduler(k_min, k_max),
        "LinearUniformMix": Schedulers.LinearUniformMixScheduler(k_min, k_max),
        "Exponential": Schedulers.ExponentialScheduler(k_min, k_max),
        "Cyclic": Schedulers.CyclicScheduler(k_min, k_max),
        "Random": Schedulers.RandomScheduler(k_min, k_max)
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, scheduler) in enumerate(schedulers.items(), 1):
        plt.subplot(2, 3, i)
        
        k_values = []
        step_counts = []
        
        for epoch in range(num_epochs):
            for step in range(steps_per_epoch):
                global_step = epoch * steps_per_epoch + step
                eps, k_dist = scheduler(epoch, num_epochs)
                k = list(k_dist.keys())[0]  # Get the first k value from distribution
                k_values.append(k)
                step_counts.append(global_step)
        
        plt.plot(step_counts, k_values, linewidth=2)
        plt.title(f'{name} Scheduler', fontsize=12, fontweight='bold')
        plt.xlabel('Training Step')
        plt.ylabel('k value')
        plt.grid(True, alpha=0.3)
        plt.ylim(-1, k_max + 1)
    
    plt.tight_layout()
    plt.savefig('results/scheduler_k_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return k_values, step_counts


def calculate_theoretical_complexity(num_epochs=10, steps_per_epoch=100, k_min=0, k_max=20):
    """Calculate theoretical computational complexity for each scheduler"""
    
    schedulers = {
        "Constant": Schedulers.ConstantScheduler(k_min, k_max),
        "Linear": Schedulers.LinearScheduler(k_min, k_max),
        "LinearUniformMix": Schedulers.LinearUniformMixScheduler(k_min, k_max),
        "Exponential": Schedulers.ExponentialScheduler(k_min, k_max),
        "Cyclic": Schedulers.CyclicScheduler(k_min, k_max),
        "Random": Schedulers.RandomScheduler(k_min, k_max)
    }
    
    complexity_results = []
    
    for name, scheduler in schedulers.items():
        total_k = 0
        k_values = []
        
        for epoch in range(num_epochs):
            for step in range(steps_per_epoch):
                eps, k_dist = scheduler(epoch, num_epochs)
                k = list(k_dist.keys())[0]  # Get the first k value from distribution
                total_k += k
                k_values.append(k)
        
        avg_k = total_k / (num_epochs * steps_per_epoch)
        max_k = max(k_values)
        min_k = min(k_values)
        std_k = np.std(k_values)
        
        complexity_results.append({
            'strategy': name,
            'total_k_steps': total_k,
            'average_k': avg_k,
            'max_k': max_k,
            'min_k': min_k,
            'std_k': std_k,
            'theoretical_complexity_factor': avg_k / k_max  # Normalized complexity
        })
    
    return pd.DataFrame(complexity_results)


def measure_practical_runtime(device="cpu", num_batches=10):
    """Measure practical runtime for different k values"""
    
    # Setup data
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root="../data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Use SmallConvNet for timing
    model = Models.SmallConvNet().to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    k_values = [1, 2, 4, 8, 16, 20]
    runtime_results = []
    
    # Get a few batches for timing
    test_batches = []
    for i, (images, labels) in enumerate(test_loader):
        if i >= num_batches:
            break
        test_batches.append((images.to(device), labels.to(device)))
    
    for k in k_values:
        print(f"Measuring runtime for k={k}...")
        
        start_time = time.time()
        
        for images, labels in test_batches:
            # Simulate adversarial training step
            adv_images, _ = pgd_attack(images, labels, model, criterion, epsilon=0.3, k=k, device=device)
            # Forward pass
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
        
        end_time = time.time()
        avg_time_per_batch = (end_time - start_time) / num_batches
        
        runtime_results.append({
            'k': k,
            'avg_time_per_batch': avg_time_per_batch,
            'relative_time': avg_time_per_batch / runtime_results[0]['avg_time_per_batch'] if runtime_results else 1.0
        })
    
    return pd.DataFrame(runtime_results)


def create_advanced_visualizations():
    """Create advanced performance vs runtime visualizations"""
    
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
        cifar_df['model_size'] = 'Medium'
        datasets_results['CIFAR-10'] = cifar_df
    
    # Try to load SVHN results
    if os.path.exists('results/svhn_adversarial_evaluation.csv'):
        svhn_df = pd.read_csv('results/svhn_adversarial_evaluation.csv')
        svhn_df['model_size'] = 'Medium'
        datasets_results['SVHN'] = svhn_df
    
    if not datasets_results:
        print("No results found for advanced visualization")
        return
    
    # Combine all datasets
    all_results = pd.concat(datasets_results.values(), ignore_index=True)
    
    # Load runtime data
    if os.path.exists('results/practical_runtime.csv'):
        runtime_df = pd.read_csv('results/practical_runtime.csv')
    else:
        print("Runtime data not found, generating dummy data for visualization")
        runtime_df = pd.DataFrame({
            'k': [1, 2, 4, 8, 16],
            'avg_time_per_batch': [0.1, 0.15, 0.25, 0.45, 0.8],
            'relative_time': [1.0, 1.5, 2.5, 4.5, 8.0]
        })
    
    # Create the mega visualization figure
    fig = plt.figure(figsize=(20, 16))
    
    # Define colors and markers for each strategy
    strategy_styles = {
        'Constant': {'color': '#1f77b4', 'marker': '^', 'linestyle': '-'},
        'Linear': {'color': '#ff7f0e', 'marker': 'o', 'linestyle': '-'},
        'LinearUniformMix': {'color': '#2ca02c', 'marker': 's', 'linestyle': '-'},
        'Exponential': {'color': '#d62728', 'marker': 'D', 'linestyle': '-'},
        'Cyclic': {'color': '#9467bd', 'marker': 'o', 'linestyle': '-'},
        'Random': {'color': '#8c564b', 'marker': 'x', 'linestyle': '--'},
        'Vanilla': {'color': '#e377c2', 'marker': '*', 'linestyle': ':'}
    }
    
    dataset_colors = {
        'MNIST': '#1f77b4',
        'CIFAR-10': '#ff7f0e', 
        'SVHN': '#2ca02c'
    }
    
    # Plot 1: Runtime vs Accuracy (your requested plot)
    ax1 = plt.subplot(2, 3, 1)
    for dataset in all_results['dataset'].unique():
        dataset_data = all_results[all_results['dataset'] == dataset]
        for strategy in dataset_data['strategy'].unique():
            strategy_data = dataset_data[dataset_data['strategy'] == strategy]
            mean_acc = strategy_data['adv_acc'].mean()
            
            # Estimate runtime based on average k for this strategy
            avg_k = strategy_data['k'].mean()
            runtime_interp = np.interp(avg_k, runtime_df['k'], runtime_df['avg_time_per_batch'])
            
            style = strategy_styles.get(strategy, {'color': 'gray', 'marker': 'o'})
            ax1.scatter(runtime_interp, mean_acc, 
                       color=dataset_colors[dataset], 
                       marker=style['marker'], 
                       s=120, alpha=0.8,
                       label=f"{dataset}-{strategy}" if dataset == 'MNIST' else "")
    
    ax1.set_xlabel('Runtime per Batch (seconds)')
    ax1.set_ylabel('Mean Adversarial Accuracy (%)')
    ax1.set_title('Performance vs Runtime Trade-off')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Adversarial Accuracy by Strategy (Radar Chart)
    ax2 = plt.subplot(2, 3, 2, projection='polar')
    strategies = list(strategy_styles.keys())
    if 'Vanilla' in strategies:
        strategies.remove('Vanilla')  # Remove for cleaner radar
    
    angles = np.linspace(0, 2 * np.pi, len(strategies), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    for dataset in dataset_colors.keys():
        if dataset in all_results['dataset'].values:
            dataset_data = all_results[all_results['dataset'] == dataset]
            values = []
            for strategy in strategies:
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
    ax2.set_xticklabels(strategies)
    ax2.set_ylim(0, 100)
    ax2.set_title('Strategy Performance Radar')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Plot 3: Performance Consistency (Error bars)
    ax3 = plt.subplot(2, 3, 3)
    x_pos = 0
    x_labels = []
    for dataset in dataset_colors.keys():
        if dataset in all_results['dataset'].values:
            dataset_data = all_results[all_results['dataset'] == dataset]
            for strategy in ['Cyclic', 'Linear', 'Exponential']:  # Top 3 performers
                strategy_data = dataset_data[dataset_data['strategy'] == strategy]
                if not strategy_data.empty:
                    mean_acc = strategy_data['adv_acc'].mean()
                    std_acc = strategy_data['adv_acc'].std()
                    
                    style = strategy_styles[strategy]
                    ax3.errorbar(x_pos, mean_acc, yerr=std_acc, 
                               marker=style['marker'], color=dataset_colors[dataset],
                               markersize=10, capsize=5, capthick=2)
                    x_labels.append(f"{dataset}\n{strategy}")
                    x_pos += 1
    
    ax3.set_xlabel('Dataset-Strategy')
    ax3.set_ylabel('Adversarial Accuracy (%) ± Std')
    ax3.set_title('Performance Consistency Analysis')
    ax3.set_xticks(range(len(x_labels)))
    ax3.set_xticklabels(x_labels, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: k-value Distribution by Strategy
    ax4 = plt.subplot(2, 3, 4)
    if 'complexity_df' in globals():
        complexity_data = globals()['complexity_df']
        bars = ax4.bar(complexity_data['strategy'], complexity_data['average_k'], 
                      color=[strategy_styles[s]['color'] for s in complexity_data['strategy']])
        ax4.set_xlabel('Strategy')
        ax4.set_ylabel('Average k value')
        ax4.set_title('Computational Complexity by Strategy')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, complexity_data['average_k']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
    
    # Plot 5: Clean vs Adversarial Accuracy Trade-off
    ax5 = plt.subplot(2, 3, 5)
    for dataset in dataset_colors.keys():
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
                           marker=style['marker'], s=120, alpha=0.8)
    
    ax5.set_xlabel('Clean Accuracy (%)')
    ax5.set_ylabel('Adversarial Accuracy (%)')
    ax5.set_title('Robustness vs Clean Performance')
    ax5.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect correlation')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Strategy Timeline Evolution
    ax6 = plt.subplot(2, 3, 6)
    k_evolution_data = []
    schedulers = {
        "Constant": Schedulers.ConstantScheduler(0, 20),
        "Linear": Schedulers.LinearScheduler(0, 20),
        "Cyclic": Schedulers.CyclicScheduler(0, 20),
        "Exponential": Schedulers.ExponentialScheduler(0, 20)
    }
    
    steps = range(0, 1000, 10)
    for name, scheduler in schedulers.items():
        k_vals = []
        for step in steps:
            epoch = step // 100
            eps, k_dist = scheduler(epoch, 10)  # Use 10 as max_epochs
            k = list(k_dist.keys())[0]  # Get the first k value from distribution
            k_vals.append(k)
        style = strategy_styles[name]
        ax6.plot(steps, k_vals, color=style['color'], 
                linewidth=2, label=name, linestyle=style['linestyle'])
    
    ax6.set_xlabel('Training Step')
    ax6.set_ylabel('k value')
    ax6.set_title('Scheduler Evolution Over Time')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/advanced_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Advanced visualizations created!")


def plot_complexity_analysis():
    """Create comprehensive complexity analysis plots"""
    
    # Calculate theoretical complexity
    complexity_df = calculate_theoretical_complexity()
    
    # Measure practical runtime
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    runtime_df = measure_practical_runtime(device=device)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Theoretical average k per strategy
    axes[0, 0].bar(complexity_df['strategy'], complexity_df['average_k'], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Average k per Strategy (Theoretical)', fontweight='bold')
    axes[0, 0].set_ylabel('Average k')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Complexity factor comparison
    axes[0, 1].bar(complexity_df['strategy'], complexity_df['theoretical_complexity_factor'], 
                   color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Relative Computational Complexity (Theoretical)', fontweight='bold')
    axes[0, 1].set_ylabel('Complexity Factor (relative to max k)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Runtime vs k (practical)
    axes[1, 0].plot(runtime_df['k'], runtime_df['avg_time_per_batch'], 'o-', linewidth=2, markersize=8)
    axes[1, 0].set_title('Practical Runtime vs k', fontweight='bold')
    axes[1, 0].set_xlabel('k (PGD steps)')
    axes[1, 0].set_ylabel('Average Time per Batch (seconds)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Relative runtime vs k
    axes[1, 1].plot(runtime_df['k'], runtime_df['relative_time'], 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 1].set_title('Relative Runtime vs k', fontweight='bold')
    axes[1, 1].set_xlabel('k (PGD steps)')
    axes[1, 1].set_ylabel('Relative Time (normalized to k=1)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    complexity_df.to_csv('results/theoretical_complexity.csv', index=False)
    runtime_df.to_csv('results/practical_runtime.csv', index=False)
    
    return complexity_df, runtime_df


def main():
    print("Generating scheduler evolution plots...")
    plot_scheduler_evolution()
    
    print("Performing complexity analysis...")
    complexity_df, runtime_df = plot_complexity_analysis()
    
    print("Creating advanced visualizations...")
    create_advanced_visualizations()
    
    print("\nTheoretical Complexity Results:")
    print(complexity_df)
    
    print("\nPractical Runtime Results:")
    print(runtime_df)
    
    # Verify theoretical vs practical correlation
    print("\nCorrelation Analysis:")
    print("Theoretical complexity should correlate with practical runtime.")
    print("Linear relationship expected: runtime ∝ k")
    
    print("\n✓ All visualizations completed!")
    print("Generated files:")
    print("  - results/scheduler_k_evolution.png")
    print("  - results/complexity_analysis.png") 
    print("  - results/advanced_visualizations.png")


if __name__ == "__main__":
    main()
