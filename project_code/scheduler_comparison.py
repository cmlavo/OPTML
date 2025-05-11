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
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

class SchedulerComparison:
    def __init__(self, k_min: int = 0, k_max: int = 10, max_epochs: int = 100):
        self.k_min = k_min
        self.k_max = k_max
        self.max_epochs = max_epochs
        
        # Initialize all schedulers
        self.schedulers = {
            'Constant': ConstantScheduler(k_min),
            'Linear': LinearScheduler(k_min, k_max),
            'LinearUniformMix': LinearUniformMixScheduler(k_min, k_max),
            'Exponential': ExponentialScheduler(k_min, k_max),
            'Cyclic': CyclicScheduler(k_min, k_max),
            'Random': RandomScheduler(k_min, k_max)
        }
        
        # Initialize composite schedulers with different epsilon strategies
        self.epsilon_strategies = ['constant', 'linear', 'cyclic', 'random']
        self.composite_schedulers = {}
        
        for k_name, k_scheduler in self.schedulers.items():
            for eps_strategy in self.epsilon_strategies:
                name = f"{k_name}_{eps_strategy}"
                self.composite_schedulers[name] = CompositeScheduler(
                    k_scheduler,
                    epsilon_type=eps_strategy,
                    epsilon_min=0.0,
                    epsilon_max=0.3
                )

    def collect_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect data for all schedulers across epochs."""
        k_data = []
        epsilon_data = []
        
        for epoch in tqdm(range(self.max_epochs), desc="Collecting scheduler data"):
            # Basic schedulers
            for name, scheduler in self.schedulers.items():
                eps, k_dist = scheduler(epoch, self.max_epochs)
                for k, prob in k_dist.items():
                    k_data.append({
                        'epoch': epoch,
                        'scheduler': name,
                        'k': k,
                        'probability': prob,
                        'epsilon': eps
                    })
            
            # Composite schedulers
            for name, scheduler in self.composite_schedulers.items():
                eps, k_dist = scheduler(epoch, self.max_epochs)
                for k, prob in k_dist.items():
                    k_data.append({
                        'epoch': epoch,
                        'scheduler': name,
                        'k': k,
                        'probability': prob,
                        'epsilon': eps
                    })
                    epsilon_data.append({
                        'epoch': epoch,
                        'scheduler': name,
                        'epsilon': eps
                    })
        
        return pd.DataFrame(k_data), pd.DataFrame(epsilon_data)

    def plot_k_distributions(self, df: pd.DataFrame):
        """Plot k distributions over time for each scheduler."""
        plt.figure(figsize=(15, 10))
        
        # Plot basic schedulers
        basic_schedulers = [s for s in self.schedulers.keys()]
        for scheduler in basic_schedulers:
            scheduler_data = df[df['scheduler'] == scheduler]
            plt.plot(scheduler_data['epoch'], scheduler_data['k'], 
                    label=scheduler, alpha=0.7)
        
        plt.title('K Values Over Time for Basic Schedulers')
        plt.xlabel('Epoch')
        plt.ylabel('K Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('output/k_distributions_basic.png')
        plt.close()

        # Plot composite schedulers
        plt.figure(figsize=(15, 10))
        for scheduler in self.composite_schedulers.keys():
            scheduler_data = df[df['scheduler'] == scheduler]
            plt.plot(scheduler_data['epoch'], scheduler_data['k'], 
                    label=scheduler, alpha=0.7)
        
        plt.title('K Values Over Time for Composite Schedulers')
        plt.xlabel('Epoch')
        plt.ylabel('K Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output/k_distributions_composite.png')
        plt.close()

    def plot_epsilon_distributions(self, df: pd.DataFrame):
        """Plot epsilon distributions over time for composite schedulers."""
        plt.figure(figsize=(15, 10))
        
        for scheduler in self.composite_schedulers.keys():
            scheduler_data = df[df['scheduler'] == scheduler]
            plt.plot(scheduler_data['epoch'], scheduler_data['epsilon'], 
                    label=scheduler, alpha=0.7)
        
        plt.title('Epsilon Values Over Time for Composite Schedulers')
        plt.xlabel('Epoch')
        plt.ylabel('Epsilon Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output/epsilon_distributions.png')
        plt.close()

    def analyze_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistics for each scheduler."""
        stats = []
        
        for scheduler in df['scheduler'].unique():
            scheduler_data = df[df['scheduler'] == scheduler]
            
            stats.append({
                'scheduler': scheduler,
                'mean_k': scheduler_data['k'].mean(),
                'std_k': scheduler_data['k'].std(),
                'min_k': scheduler_data['k'].min(),
                'max_k': scheduler_data['k'].max(),
                'mean_epsilon': scheduler_data['epsilon'].mean(),
                'std_epsilon': scheduler_data['epsilon'].std(),
                'min_epsilon': scheduler_data['epsilon'].min(),
                'max_epsilon': scheduler_data['epsilon'].max()
            })
        
        return pd.DataFrame(stats)

    def run_comparison(self):
        """Run the complete comparison analysis."""
        print("Starting scheduler comparison analysis...")
        
        # Collect data
        k_df, epsilon_df = self.collect_data()
        
        # Generate plots
        print("Generating plots...")
        self.plot_k_distributions(k_df)
        self.plot_epsilon_distributions(epsilon_df)
        
        # Calculate statistics
        print("Calculating statistics...")
        stats_df = self.analyze_statistics(k_df)
        
        # Save statistics to CSV
        stats_df.to_csv('output/scheduler_statistics.csv', index=False)
        print("Analysis complete! Results saved in output/ directory.")
        
        return stats_df

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    import os
    os.makedirs('output', exist_ok=True)
    
    # Run comparison
    comparison = SchedulerComparison(k_min=0, k_max=10, max_epochs=100)
    stats = comparison.run_comparison()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(stats.to_string()) 