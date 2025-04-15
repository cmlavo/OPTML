import torch

"""
This script implements adversarial attacks
"""

def fgsm_attack(image, label, model, loss_fun, epsilon, device):
    """
    FGSM attack (untargeted, single-iteration)

    inputs:
        image: input image as a mini-batch of size [B,C,W,H] for example [Batch, 1, 28, 28] with MNIST 
        label: ground truth label id
        model: the model to attack
        loss_function: the network's loss function
        epsilon: magnitude of the perturbation
    
    returns:
        adversarial_image: perturbed image
        perturbation: the perturbation
    """

    # Freeze the model weights so that they are not part of the backprop of gradients
    model.eval()

    image = image.to(device)
    label = label.to(device)

    # We need gradients with respect to the images
    image.requires_grad = True

    outputs = model(image)
    loss = loss_fun(outputs, label)

    # Zero all existing gradients in the model
    model.zero_grad()

    # Get gradients
    loss.backward()

    # Create perturbation
    image_grad = image.grad
    sign_img_grad = torch.sign(image_grad)
    perturbation = epsilon*sign_img_grad

    # No need to track gradients with respect to the image(s) anymore
    image.requires_grad = False

    # Add perturbation to the image
    adversarial_image = image + perturbation

    # If some pixels went above or below 1 or -1, clamp to keep image valid
    adversarial_image = torch.clamp(adversarial_image, 0, 1)

    return adversarial_image, perturbation


def pgd_attack(images, labels, model, loss_fun, epsilon, k, device):
    """
    Somewhat naive/intuitive implementation of the Projected Gradient Descent attack.
    Would be good to check other codebases to make sure this is how it's done.
    k is the number of gradient steps.
    """
    total_perturbation = torch.zeros_like(images).to(device)

    # If k == 0, return clean images.
    if k == 0:
        return images, torch.zeros_like(images)
    
    """
    Although the attack is called PROJECTED Gradient Descent attack, we have no explicit projections here for 2 reasons:
    - We take k steps of size epsilon/k. By triangular inequality the combined step cannot excede epsilon:
      |step1 + step2 + ... + stepk| <= |step1| + ... + |stepk| = k*epsilon/k = epsilon
      Therefore we don't need to project the perturbation back to the epsilon-sized ball !
    
    - The other projection (clamp image to [0, 1] to keep valid pixel values) is already done inside the sub-function fgsm_attack
    """
    mini_epsilon = epsilon/k
    for iteration in range(k):
        images, current_perturbation = fgsm_attack(images, labels, model, loss_fun, mini_epsilon, device)
        total_perturbation += current_perturbation
    return images, total_perturbation