import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import project_code.model.Models as Models
import project_code.Defences as Defences
import project_code.schedulers.Schedulers as Schedulers
import pandas as pd
import numpy as np
from project_code.Attacks import pgd_attack
import torch.nn.functional as F


def get_svhn_data_loaders(batch_size=16):
    print("Loading SVHN data...")
    # SVHN specific transforms
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.SVHN(root="../../project_code/data", split='train', download=True, transform=transform_train)
    test_dataset = datasets.SVHN(root="../../project_code/data", split='test', download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
    print("SVHN data loaded.")
    return train_loader, test_loader


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_loss /= test_total
    test_acc = 100.0 * test_correct / test_total
    return test_loss, test_acc


def run_all_k_strategies_svhn(k_min=0, k_max=20, epsilon=8/255, num_epochs=2, device=None):
    """Run K strategy experiments on SVHN with MediumConvNet"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader, test_loader = get_svhn_data_loaders()
    results = []
    
    schedulers = {
        "Vanilla": Schedulers.VanillaScheduler(),
        "Constant": Schedulers.ConstantScheduler(k_min, k_max),
        "Linear": Schedulers.LinearScheduler(k_min, k_max),
        "LinearUniformMix": Schedulers.LinearUniformMixScheduler(k_min, k_max),
        "Exponential": Schedulers.ExponentialScheduler(k_min, k_max),
        "Cyclic": Schedulers.CyclicScheduler(k_min, k_max),
        "Random": Schedulers.RandomScheduler(k_min, k_max)
    }
    
    os.makedirs("results", exist_ok=True)
    
    for name, scheduler in schedulers.items():
        print(f"\nTraining SVHN with {name} scheduler...")
        
        # Use MediumConvNet for SVHN (3 input channels)
        model = Models.MediumConvNetCIFAR().to(device)  # Same architecture as CIFAR
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        Defences.train_with_adversarial_scheduler(
            model, train_loader, test_loader, optimizer, criterion,
            epsilon, scheduler, device, num_epochs=num_epochs, test_eval_rate=3
        )
        
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        results.append({
            "strategy": name,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "dataset": "SVHN"
        })
        
        # Save model after training
        torch.save(model.state_dict(), f"results/svhn_model_{name}.pth")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("results/svhn_k_strategy_results.csv", index=False)
    print("\nSVHN Summary of results:")
    print(df)
    return df


def load_svhn_models(strategies, device):
    """Load trained SVHN models for evaluation"""
    model_dict = {}
    for name in strategies:
        model = Models.MediumConvNetCIFAR().to(device)
        model.load_state_dict(torch.load(f"results/svhn_model_{name}.pth", map_location=device))
        model.eval()
        model_dict[name] = model
    return model_dict


def evaluate_strategies_on_attacks_svhn(model_dict, test_loader, device, epsilon=8/255, k_list=[1,2,4,8,16]):
    """Evaluate SVHN models against adversarial attacks"""
    results = []
    criterion = nn.CrossEntropyLoss()
    
    for strategy, model in model_dict.items():
        print(f"Evaluating {strategy} on SVHN...")
        
        # Clean accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        clean_acc = 100.0 * correct / total

        # Adversarial accuracy for each k
        for k in k_list:
            correct_adv = 0
            total_adv = 0
            confidences = []
            
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                adv_images, _ = pgd_attack(images, labels, model, criterion, epsilon, k, device)
                outputs = model(adv_images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                total_adv += labels.size(0)
                correct_adv += (predicted == labels).sum().item()
                
                for i in range(labels.size(0)):
                    confidences.append(probs[i, predicted[i]].item())
            
            adv_acc = 100.0 * correct_adv / total_adv
            mean_conf = float(np.mean(confidences)) if confidences else 0.0
            
            results.append({
                "strategy": strategy,
                "k": k,
                "clean_acc": clean_acc,
                "adv_acc": adv_acc,
                "mean_confidence": mean_conf,
                "dataset": "SVHN"
            })
    
    df = pd.DataFrame(results)
    df.to_csv("results/svhn_adversarial_evaluation.csv", index=False)
    print("\nSVHN Adversarial evaluation results:")
    print(df.head(10))
    return df


def main():
    print("Starting SVHN K-strategy experiments...")
    
    # Training phase
    train_results = run_all_k_strategies_svhn(device="cpu")
    
    # Evaluation phase
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader = get_svhn_data_loaders()
    
    strategies = ["Vanilla", "Constant", "Linear", "LinearUniformMix", "Exponential", "Cyclic", "Random"]
    model_dict = load_svhn_models(strategies, device)
    
    eval_results = evaluate_strategies_on_attacks_svhn(model_dict, test_loader, device)
    
    print("SVHN experiments completed!")


if __name__ == "__main__":
    main()
