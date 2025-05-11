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