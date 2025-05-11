import numpy as np
import matplotlib.pyplot as plt
from Schedulers import (
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

class SchedulerEpochAnalysis:
    def __init__(self, k_min: int = 0, k_max: int = 10, max_epochs: int = 100):
        self.k_min = k_min
        self.k_max = k_max
        self.max_epochs = max_epochs
        
        # Initialize base schedulers with default epsilon
        self.base_schedulers = {
            'Constant': ConstantScheduler(k_min, k_max),
            'Linear': LinearScheduler(k_min, k_max),
            'LinearUniformMix': LinearUniformMixScheduler(k_min, k_max),
            'Exponential': ExponentialScheduler(k_min, k_max),
            'Cyclic': CyclicScheduler(k_min, k_max),
            'Random': RandomScheduler(k_min, k_max)
        }
        
        # Different epsilon configurations to test
        self.epsilon_configs = {
            'constant_small': 0.1,
            'constant_medium': 0.3,
            'constant_large': 0.5,
            'linear': 'linear',
            'cyclic': 'cyclic',
            'random': 'random'
        }

    def analyze_base_schedulers(self) -> pd.DataFrame:
        """Analyze base schedulers without epsilon variations."""
        data = []
        
        for scheduler_name, scheduler in self.base_schedulers.items():
            for epoch in range(self.max_epochs):
                eps, k_dist = scheduler(epoch, self.max_epochs)
                for k, prob in k_dist.items():
                    data.append({
                        'epoch': epoch,
                        'scheduler': scheduler_name,
                        'k': k,
                        'probability': prob,
                        'epsilon': eps
                    })
        
        return pd.DataFrame(data)

    def analyze_epsilon_influence(self, best_scheduler: str) -> pd.DataFrame:
        """Analyze how epsilon affects the best scheduler."""
        data = []
        
        for eps_config_name, eps_config in self.epsilon_configs.items():
            if isinstance(eps_config, (int, float)):
                scheduler = self.base_schedulers[best_scheduler].__class__(
                    self.k_min, self.k_max, epsilon_max=eps_config
                )
            else:
                scheduler = CompositeScheduler(
                    self.base_schedulers[best_scheduler].__class__(self.k_min, self.k_max),
                    epsilon_type=eps_config,
                    epsilon_min=0.1,
                    epsilon_max=0.5
                )
            
            for epoch in range(self.max_epochs):
                eps, k_dist = scheduler(epoch, self.max_epochs)
                for k, prob in k_dist.items():
                    data.append({
                        'epoch': epoch,
                        'epsilon_config': eps_config_name,
                        'k': k,
                        'epsilon': eps,
                        'probability': prob
                    })
        
        return pd.DataFrame(data)

    def plot_scheduler_comparison(self, df: pd.DataFrame):
        """Plot comparison of different schedulers."""
        plt.figure(figsize=(15, 10))
        
        for scheduler_name in self.base_schedulers.keys():
            scheduler_data = df[df['scheduler'] == scheduler_name]
            plt.plot(scheduler_data['epoch'], scheduler_data['k'], 
                    label=scheduler_name, alpha=0.7)
        
        plt.title('Comparison of Different Scheduler Strategies')
        plt.xlabel('Epoch')
        plt.ylabel('K Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('output/scheduler_comparison.png')
        plt.close()

    def plot_epsilon_influence_on_best(self, df: pd.DataFrame, best_scheduler: str):
        """Plot how epsilon affects the best scheduler."""
        plt.figure(figsize=(15, 10))
        
        for eps_config in self.epsilon_configs.keys():
            config_data = df[df['epsilon_config'] == eps_config]
            plt.plot(config_data['epoch'], config_data['k'], 
                    label=f'ε={eps_config}', alpha=0.7)
        
        plt.title(f'Influence of Epsilon on {best_scheduler} Scheduler')
        plt.xlabel('Epoch')
        plt.ylabel('K Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'output/epsilon_influence_on_best.png')
        plt.close()

    def calculate_scheduler_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics for each scheduler."""
        metrics = []
        
        for scheduler_name in self.base_schedulers.keys():
            scheduler_data = df[df['scheduler'] == scheduler_name]
            
            # Calculate various metrics
            metrics.append({
                'scheduler': scheduler_name,
                'mean_k': scheduler_data['k'].mean(),
                'std_k': scheduler_data['k'].std(),
                'k_range': scheduler_data['k'].max() - scheduler_data['k'].min(),
                'k_progression': scheduler_data['k'].iloc[-1] - scheduler_data['k'].iloc[0],
                'stability': 1 / (1 + scheduler_data['k'].std()),  # Higher is more stable
                'coverage': len(scheduler_data['k'].unique()) / (self.k_max - self.k_min + 1)
            })
        
        return pd.DataFrame(metrics)

    def run_analysis(self):
        """Run the complete analysis."""
        print("Starting scheduler analysis...")
        
        # Step 1: Analyze base schedulers
        print("Analyzing base schedulers...")
        base_df = self.analyze_base_schedulers()
        self.plot_scheduler_comparison(base_df)
        
        # Calculate metrics and find best scheduler
        metrics_df = self.calculate_scheduler_metrics(base_df)
        metrics_df.to_csv('output/scheduler_metrics.csv', index=False)
        
        # Find best scheduler based on a combination of metrics
        metrics_df['score'] = (
            metrics_df['stability'] * 0.3 +
            metrics_df['coverage'] * 0.3 +
            metrics_df['k_progression'] * 0.4
        )
        best_scheduler = metrics_df.loc[metrics_df['score'].idxmax(), 'scheduler']
        
        print(f"\nBest scheduler found: {best_scheduler}")
        print("\nScheduler metrics:")
        print(metrics_df.to_string())
        
        # Step 2: Analyze epsilon influence on best scheduler
        print(f"\nAnalyzing epsilon influence on {best_scheduler}...")
        epsilon_df = self.analyze_epsilon_influence(best_scheduler)
        self.plot_epsilon_influence_on_best(epsilon_df, best_scheduler)
        
        # Calculate epsilon impact metrics
        epsilon_metrics = []
        for eps_config in self.epsilon_configs.keys():
            config_data = epsilon_df[epsilon_df['epsilon_config'] == eps_config]
            epsilon_metrics.append({
                'epsilon_config': eps_config,
                'mean_k': config_data['k'].mean(),
                'std_k': config_data['k'].std(),
                'k_range': config_data['k'].max() - config_data['k'].min(),
                'mean_epsilon': config_data['epsilon'].mean()
            })
        
        epsilon_metrics_df = pd.DataFrame(epsilon_metrics)
        epsilon_metrics_df.to_csv('output/epsilon_metrics.csv', index=False)
        
        print("\nEpsilon influence metrics:")
        print(epsilon_metrics_df.to_string())
        
        return metrics_df, epsilon_metrics_df

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    import os
    os.makedirs('output', exist_ok=True)
    
    # Run analysis
    analysis = SchedulerEpochAnalysis(k_min=0, k_max=10, max_epochs=100)
    metrics_df, epsilon_metrics_df = analysis.run_analysis() 