# Scheduling Adversarial Examples for Effective and Robust Learning

This repository contains the implementation and experimental results for the CS-439 Optimization for Machine Learning Mini-Project investigating how different adversarial training schedules affect neural network performance on benchmark classification tasks (MNIST and CIFAR-10).

**Authors:** Julien Stalhandske, Christophe Michaud-Lavoie, Hugues Louis Christophe  
**Institution:** EPFL - École Polytechnique Fédérale de Lausanne

## Abstract

This project investigates how different adversarial training schedules affect neural network performance on benchmark classification tasks (MNIST and CIFAR-10). Each schedule varies k, the number of gradient ascent steps used to generate adversarial examples during training. We evaluated six schedulers based on their impact on accuracy, runtime, and robustness to attacks, followed by a qualitative analysis of decision boundaries. Results highlight the importance of curriculum: exposing models to high k-diversity later in training is especially beneficial for smaller models prone to forgetting.

## Setup and Prerequisites

### Environment Setup

This repository includes an `environment.yml` file containing all necessary Python Conda environment packages. Create the conda environment by running:

```bash
conda env create -f environment.yml
conda activate OPTML_env
```

Alternatively, you can use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### HuggingFace Access

You will need a HuggingFace access token with read permissions for the following repositories:

- [MNIST-SmallConvs-AdversarialSchedulers](https://huggingface.co/JulienStal/MNIST-SmallConvs-AdversarialSchedulers)
- [MNIST-MediumConvs-AdversarialSchedulers](https://huggingface.co/JulienStal/MNIST-MediumConvs-AdversarialSchedulers)
- [SSNP Model](https://huggingface.co/cmlavo/SSNP)

## Repository Structure

```
OPTML/
├── environment.yml              # Conda environment specification
├── requirements.txt             # Python package requirements
├── README.md                   # This file
├── LICENSE                     # Project license
├── data/                       # Dataset storage
│   ├── MNIST/                  # MNIST dataset
│   └── cifar-10-batches-py/    # CIFAR-10 dataset
├── models/                     # Trained model storage
│   └── ssnp/                   # Self-Supervised Network Projection models
├── output/                     # Generated outputs
│   └── images/                 # Decision boundary maps and visualizations
└── project_code/               # Main implementation
    ├── run_scheduler_experiments.ipynb  # Main training notebook
    ├── Visualization.ipynb             # Decision boundary visualization
    ├── Models.py                       # Neural network architectures
    ├── Attacks.py                      # Adversarial attack implementations
    ├── Defences.py                     # Defense mechanisms
    ├── Tools.py                        # Utility functions
    ├── ssnp.py                         # SSNP implementation
    └── schedulers/                     # Scheduler implementations
        └── Schedulers.py               # All scheduler variants
```

## Methodology

### Datasets

We conduct experiments on two benchmark datasets:

- **MNIST**: Lower dimensionality dataset that allows for efficient experimentation and effective dimensionality reduction for decision boundary visualization
- **CIFAR-10**: Higher visual complexity dataset with adversarial examples requiring smaller, less perceptible ε values

### Neural Network Architectures

#### MNIST Models

**SmallConvNet**: Lightweight baseline with two convolutional layers (8 and 16 channels), each followed by 2×2 max-pooling, and two fully connected layers.

**MediumConvNet**: Enhanced architecture with three convolutional layers (32, 64, 128 channels), a fully connected layer with 128 units, and dropout for better capacity.

#### CIFAR-10 Models

- Modified MediumConvNet adapted for 3-channel input
- ResNet-18 and ResNet-40 architectures for scalability testing

### Adversarial Training Schedulers

We define a scheduler as a mapping `s : [0, 1] → Δ(K)` where:
- Input `t ∈ [0, 1]` represents training progress (current epoch / total epochs)
- `Δ(K)` denotes probability distributions over discrete k-values for PGD adversarial example generation
- `K = [0, 1, ..., k_max]` represents the range of gradient ascent steps

#### Implemented Schedulers

1. **Linear Uniform Mix**: `s(t) = U{0, 1, ..., ⌊t · k_max⌋}` (based on Curriculum Adversarial Training)
2. **Linear**: Progressive increase without uniform sampling
3. **Exponential**: Exponential growth in adversarial strength
4. **Cyclic**: Periodic variation in k values
5. **Random**: Random sampling from full k range
6. **Constant**: Always uses k_max
7. **Vanilla**: No adversarial training (k=0)

### Decision Boundary Visualization

We adapt the Self-Supervised Network Projection (SSNP) method to visualize complex multi-class decision boundaries:

1. **Dimensionality Reduction**: Train a regression neural network to project images into 2D coordinates while maximizing category cluster separation
2. **Inverse Projection**: Convert 2D points back to synthetic full-sized images
3. **Decision Boundary Mapping**: Generate 300×300 grid where each pixel is back-projected and classified to create Decision Boundary Maps (DBMs)

## Running the Code

### 1. Training Performance Experiments

The main training script is located in `project_code/run_scheduler_experiments.ipynb`. This Jupyter Notebook:

- Trains selected models using all implemented schedulers
- Outputs performance comparisons across schedulers
- Currently configured for MNIST training with SmallConv model
- Settings and model selection can be modified in the "Hyper-parameters" cell

**Usage:**
```bash
jupyter notebook project_code/run_scheduler_experiments.ipynb
```

### 2. Decision Boundary Visualization

The visualization script `Visualization.ipynb` provides:

- DBM generation for MNIST-trained classification models from HuggingFace
- Support for both imported and newly trained SSNP models
- Configurable parameters in cell 2
- Currently configured for MediumConvNet models from HuggingFace

**Key Features:**
- Import classification models from HuggingFace repositories
- Generate DBMs with/without MNIST training points highlighted
- Inverse projection visualization for specific pixels
- Customizable model selection and visualization parameters

**Usage:**
```bash
jupyter notebook Visualization.ipynb
```

### 3. Scheduler Analysis

Generate scheduler profile plots used in the research by running:

```bash
cd project_code/schedulers
python Schedulers.py
```

This script produces the k(t) profile plots for all implemented schedulers.

## Key Findings

### MNIST Results

#### SmallConvNet Performance
- **Linear Uniform Mix** and **Cyclic** schedulers achieved best overall performance
- Linear Uniform Mix: 97.2% clean accuracy, 87.4% adversarial accuracy (k=16)
- Only 1.3% behind Vanilla training in clean accuracy while maintaining strong robustness
- **Constant** and **Random** schedulers showed poor performance, often collapsing to single-class predictions

#### MediumConvNet Performance  
- **Cyclic** scheduler slightly outperformed Linear Uniform Mix
- Higher model capacity helped mitigate catastrophic forgetting effects
- Better overall robustness compared to SmallConvNet across all schedulers

### Runtime-Accuracy Trade-offs

Our complexity analysis revealed that training runtime scales directly with expected k-value `E[k]`:

| Scheduler | Complexity | Runtime Ratio | Key Advantage |
|-----------|------------|---------------|---------------|
| Constant | O(k_max) | 1.00 | Baseline comparison |
| Linear | O(k_max/2) | 0.44 | Simple curriculum |
| Linear Uniform | O(k_max/4) | 0.23 | **Best efficiency** |
| Exponential | O(k_max/ln(k_max+1)-1) | 0.21 | **Favorable for high k_max** |
| Cyclic | O(k_max/2) | 0.44 | Prevents forgetting |
| Random | O(k_max/2) | 0.52 | Poor curriculum |

**Key Insight**: Exponential and Linear Uniform schedulers offer the most favorable accuracy-cost trade-off, achieving comparable adversarial accuracy with roughly half the computational cost.

### Decision Boundary Analysis

Decision Boundary Maps revealed important insights:

1. **Vanilla vs. Adversarial Training**: Vanilla models contain exploitable zones with class exclaves, while adversarially trained models show more robust, continuous boundaries

2. **Scheduler Comparison**: Despite performance differences, DBMs between top-performing schedulers (Linear Uniform Mix vs. Exponential) were remarkably similar, with differences mainly in low-confidence regions

3. **Model Size Impact**: Larger models (MediumConvNet) showed less variation between schedulers in DBM analysis, suggesting that computational efficiency becomes the primary differentiator at scale

### CIFAR-10 Challenges

Higher-dimensional CIFAR-10 experiments revealed:
- Previously successful schedulers (Linear Uniform Mix, Cyclic) failed to work reliably out-of-the-box
- Maintaining balance between clean and adversarial accuracy becomes increasingly difficult
- Requires more careful hyperparameter tuning compared to MNIST

## Conclusions

1. **Curriculum Importance**: Effective schedulers must start with easy examples before introducing difficult ones. Immediate exposure to hard examples (Constant, Random) often leads to training collapse.

2. **Model Scale Effects**: Larger models are less sensitive to specific scheduler choices, making computational efficiency the primary consideration.

3. **Practical Recommendations**:
   - For smaller models: Use **Cyclic** or **Linear Uniform Mix** schedulers
   - For computational efficiency: Use **Exponential** or **Linear Uniform** schedulers  
   - Avoid **Constant** and **Random** schedulers in most scenarios

4. **Scalability**: Findings from MNIST provide valuable insights, but higher-dimensional datasets require additional consideration and hyperparameter tuning.

## File Descriptions

### Core Implementation Files

- **`project_code/run_scheduler_experiments.ipynb`**: Main training notebook for comparing all schedulers
- **`Visualization.ipynb`**: Decision boundary map generation and analysis
- **`project_code/Attacks.py`**: Implementation of adversarial attacks (PGD, FGSM)
- **`project_code/Defences.py`**: Adversarial defense mechanisms and training procedures
- **`project_code/model/Models.py`**: Neural network architectures (SmallConvNet, MediumConvNet, ResNets)
- **`project_code/schedulers/Schedulers.py`**: All scheduler implementations and plotting utilities
- **`project_code/ssnp.py`**: Self-Supervised Network Projection for dimensionality reduction
- **`project_code/Tools.py`**: Utility functions for data loading, preprocessing, and evaluation

### Experimental Scripts

- **`project_code/experiment_real.py`**: Real-world experiment execution
- **`project_code/run_cifar10_experiment.py`**: CIFAR-10 specific experiments
- **`project_code/run_k_strategy_experiment.py`**: k-strategy comparison experiments

### Data and Models

- **`data/MNIST/`**: MNIST dataset storage
- **`data/cifar-10-batches-py/`**: CIFAR-10 dataset storage  
- **`models/ssnp/`**: Trained SSNP models for visualization
- **`output/images/`**: Generated decision boundary maps and scheduler plots

## Dependencies

### Core Requirements
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
jupyter>=1.3.0
```

### Additional Libraries
```
huggingface-hub>=0.8.0
scikit-learn>=1.0.0
scipy>=1.7.0
pandas>=1.3.0
seaborn>=0.11.0
```

See `environment.yml` or `requirements.txt` for complete dependency specifications.

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for faster training
- **RAM**: Minimum 8GB, 16GB+ recommended for larger models
- **Storage**: ~2GB for datasets and model checkpoints

Experiments in the paper were conducted on EPFL Gnoto server using one GPU.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{stalhandske2024scheduling,
  title={Scheduling Adversarial Examples for Effective and Robust Learning},
  author={Stalhandske, Julien and Michaud-Lavoie, Christophe and Christophe, Hugues Louis},
  journal={CS-439 Optimization for Machine Learning Mini-Project},
  institution={EPFL},
  year={2024}
}
```

## License

This project is licensed under the terms specified in the `LICENSE` file.

## Contact

For questions or issues, please open an issue on this repository or contact the authors through EPFL channels.

## Acknowledgments

- EPFL CS-439 Optimization for Machine Learning course staff
- HuggingFace for model hosting and access
- EPFL Gnoto server for computational resources
