import torch
import random
from time import time
try:
    import Attacks
except: 
    from project_code import attacks

"""
This script implements functions to train our models under adversarial attacks.
"""

def train_vanilla(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=6, test_eval_rate = 1):
    """
    Vanilla training script with no defense.
    Since computing the loss on the test set is costly and not part of training, we do it every "test_eval_rate" iterations
    to control overfitting.
    """

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backpropagate gradients
            loss.backward()
            optimizer.step()

            # Compute accuracy
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        model.eval()

        # --- Evaluation on test set ---
        if (epoch + 1) % test_eval_rate == 0: 
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

            print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | " f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        


def train_with_adversarial_scheduler(model, train_loader, test_loader, optimizer, criterion, epsilon, adversarial_scheduler, device, num_epochs=6, test_eval_rate=1, sched_lr = False, clip_norm = None):
    """
    Train a model using adversarial examples generated according to a dynamic scheduler.
    """

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0, total_iters=num_epochs)    

    total_time = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Call the scheduler to get the distribution of k values for the current epoch.
        _, k_distribution = adversarial_scheduler(epoch, num_epochs)

        start_time = time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Sample a k value according to the distribution. Note: it would probably be better to generate k in a deterministic way (easier to analyse).
            k_values = list(k_distribution.keys())
            probs = list(k_distribution.values())
            sampled_k = random.choices(k_values, weights=probs, k=1)[0]
            
            # Generate adversarial image
            images, perturbations = Attacks.pgd_attack(images, labels, model, criterion, epsilon, sampled_k, device)

            # Normal training from now on
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)  # Clip gradients
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        end_time = time()

        total_time += end_time - start_time
        
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        if sched_lr: lr_scheduler.step()

        model.eval()

        if (epoch + 1) % test_eval_rate == 0:
            test_loss, test_adv_loss = 0.0, 0.0
            test_correct, test_adv_correct = 0, 0
            test_total, test_adv_total = 0, 0

            #with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                adv_images, _ = Attacks.pgd_attack(images, labels, model, criterion, epsilon, 7, device)
                with torch.no_grad():
                    outputs = model(images)
                    adv_outputs = model(adv_images)
                    loss = criterion(outputs, labels)
                    adv_loss = criterion(adv_outputs, labels)
    
                    test_loss += loss.item() * images.size(0)
                    test_adv_loss += adv_loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    _, predicted_adv = adv_outputs.max(1)
                    test_total += labels.size(0)
                    test_adv_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    test_adv_correct += (predicted_adv == labels).sum().item()

            test_loss /= test_total
            test_adv_loss /= test_adv_total
            test_acc = 100.0 * test_correct / test_total
            #if test_acc < 20: assert False # REMOVE at the end
            test_adv_acc = 100.0 * test_adv_correct / test_adv_total

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"Test Adv Loss (k=7): {test_adv_loss:.4f}, Test Adv Acc: {test_adv_acc:.2f}%")
            if sched_lr: print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f"Training time (without eval): {minutes} min {seconds:.2f} sec")
