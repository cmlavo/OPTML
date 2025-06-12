import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import model.Models as Models, Defences, schedulers.Schedulers as Schedulers
import pandas as pd
import os
import numpy as np
from Attacks import pgd_attack
import torch.nn.functional as F


def get_cifar10_data_loaders(batch_size=200):
    """Get CIFAR-10 data loaders with appropriate transforms"""
    # CIFAR-10 specific transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root="../data", train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
    
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


def run_all_k_strategies_cifar10(k_min=0, k_max=7, epsilon=1/255, num_epochs=15, device=None):
    """Run K strategy experiments on CIFAR-10 with MediumConvNet"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    train_loader, test_loader = get_cifar10_data_loaders()
    results = []
    
    schedulers = {
        "CyclicUniformMix": Schedulers.CyclicUniformMixScheduler(k_min = k_min, k_max = k_max, epsilon_max = epsilon),
        "LinearUniformMix": Schedulers.LinearUniformMixScheduler(k_min, k_max, epsilon),
        #"Cyclic": Schedulers.CyclicScheduler(k_min, k_max),
        #"Random": Schedulers.RandomScheduler(k_min, k_max),
        #"Vanilla": Schedulers.ConstantScheduler(k_min, k_min),
        #"Constant": Schedulers.ConstantScheduler(k_max, k_max),
        #"Linear": Schedulers.LinearScheduler(k_min, k_max),
        #"Exponential": Schedulers.ExponentialScheduler(k_min, k_max),
    }
    
    os.makedirs("results", exist_ok=True)
    
    for name, scheduler in schedulers.items():
        print(f"\nTraining CIFAR-10 with {name} scheduler...")
        
        # Use MediumConvNet for CIFAR-10 (3 input channels)
        #model = Models.MediumConvNetCIFAR().to(device)
        model = Models.resnet18_cifar10().to(device)
        #model = Models.resnet40_cifar10().to(device)
        #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Linear LR scheduler: decays LR linearly to 0 over training
        Defences.train_with_adversarial_scheduler(
            model, train_loader, test_loader, optimizer, criterion,
            epsilon, scheduler, device, num_epochs=num_epochs, test_eval_rate=3, 
            sched_lr = True 
        )
        
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        results.append({
            "strategy": name,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "dataset": "CIFAR-10"
        })
        
        # Save model after training
        torch.save(model.state_dict(), f"results/cifar10_model_{name}.pth")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("results/cifar10_k_strategy_results.csv", index=False)
    print("\nCIFAR-10 Summary of results:")
    print(df)
    return df


def load_cifar10_models(strategies, device):
    """Load trained CIFAR-10 models for evaluation"""
    model_dict = {}
    for name in strategies:
        #model = Models.MediumConvNetCIFAR().to(device)
        model = Models.resnet18_cifar10().to(device)
        #model = Models.resnet40_cifar10().to(device)
        model.load_state_dict(torch.load(f"results/cifar10_model_{name}.pth", map_location=device))
        model.eval()
        model_dict[name] = model
    return model_dict


def evaluate_strategies_on_attacks_cifar10(model_dict, test_loader, device, epsilon=8/255, k_list=[1,2,4,8,16]):
    """Evaluate CIFAR-10 models against adversarial attacks"""
    results = []
    criterion = nn.CrossEntropyLoss()
    
    for strategy, model in model_dict.items():
        print(f"Evaluating {strategy} on CIFAR-10...")
        
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
                "dataset": "CIFAR-10"
            })
    
    df = pd.DataFrame(results)
    df.to_csv("results/cifar10_adversarial_evaluation.csv", index=False)
    print("\nCIFAR-10 Adversarial evaluation results:")
    print(df.head(10))
    return df


def main():

    print("Starting CIFAR-10 K-strategy experiments...")
    # Training phase
    train_results = run_all_k_strategies_cifar10()

    # Evaluation phase
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Training on ", device)
    _, test_loader = get_cifar10_data_loaders()

    #strategies = ["Constant", "Linear", "LinearUniformMix", "Exponential", "Cyclic", "Random"]
    #strategies = ["Vanilla", "Constant", "Linear", "LinearUniformMix", "Exponential", "Cyclic", "Random"]
    strategies = ["LinearUniformMix", "CyclicUniformMix"]
    model_dict = load_cifar10_models(strategies, device)
    
    eval_results = evaluate_strategies_on_attacks_cifar10(model_dict, test_loader, device)
    
    print("CIFAR-10 experiments completed!")


if __name__ == "__main__":
    main()