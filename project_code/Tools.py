from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F


"""
TODO: add a pipeline to evaluate models: clean accuracy, k-adversarial accuracy
"""

"""
Useful function to visualize model output
"""
def plot_predictions(model, images, gt_labels, device, n_max):
    model.eval()
    n_images = images.shape[0]
    assert n_images == gt_labels.shape[0]
    images, gt_labels = images.to(device), gt_labels.to(device)

    # Run model
    with torch.no_grad():
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

    # Plot the first n examples
    n = min(n_images, n_max)
    for i in range(n):
        plt.figure(figsize=(10, 3))

        # Plot the image
        plt.subplot(1, 2, 1)
        plt.imshow(images[i].squeeze().cpu(), cmap='gray')
        plt.title(f"True: {gt_labels[i].item()}")
        plt.axis('off')

        probs_list = list(probs[i])
        probs_list = [round(p.item(), 10) for p in probs[i]]

        """
        # Plot the probabilities
        plt.subplot(1, 2, 2)
        plt.bar(range(10), probs[i].cpu().numpy())
        plt.xticks(range(10))
        plt.title(f"Predicted: {probs[i].argmax().item()}")
        plt.xlabel("Digit")
        plt.ylabel("Probability")"
        """

        plt.tight_layout()
        plt.show()
        print("Output probabilities:", " | ".join(f"{i}: {p:.2f}" for i, p in enumerate(probs_list)))