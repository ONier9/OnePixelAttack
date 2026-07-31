import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pygad
from training.data_loader import get_data_loaders, get_classes
from training.model import Net
import os
import config

CURRENT_ARCH = os.getenv("CURRENT_ARCH", getattr(config, "CURRENT_ARCH", "resnet50"))
CURRENT_N_PIXELS = int(os.getenv("CURRENT_N_PIXELS", getattr(config, "CURRENT_N_PIXELS", 10)))
CURRENT_IMAGES = int(os.getenv("CURRENT_IMAGES", getattr(config, "CURRENT_IMAGES", 50)))
CURRENT_TARGET_CLASS = int(os.getenv("CURRENT_TARGET_CLASS", getattr(config, "CURRENT_TARGET_CLASS", 1)))

import time

device = None
model = None
IMAGENET_MEAN = None
IMAGENET_STD = None

def normalize(x):
    return (x - IMAGENET_MEAN) / IMAGENET_STD

def predict(image_tensor):
    """image_tensor is expected in [0, 1] range; normalization happens here."""
    with torch.no_grad():
        logits = model(normalize(image_tensor))
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()[0]

def apply_n_pixel_attack(image, params, n_pixels):
    perturbed = image.clone()
    for i in range(n_pixels):
        base = i * 5
        x = int(params[base + 0])
        y = int(params[base + 1])
        r = params[base + 2]
        g = params[base + 3]
        b = params[base + 4]
        perturbed[0, 0, y, x] = r
        perturbed[0, 1, y, x] = g
        perturbed[0, 2, y, x] = b
    return torch.clamp(perturbed, 0, 1)

def targeted_objective(params, image, target_class, n_pixels):
    perturbed = apply_n_pixel_attack(image, params, n_pixels)
    probs = predict(perturbed)
    return -probs[target_class]

def show_image(ax, tensor, title):
    im = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    ax.imshow(im)
    ax.set_title(title, fontsize=8)
    ax.axis("off")

def run_attack_on_image(img, target_class, n_pixels):
    H, W = img.shape[-2], img.shape[-1]
    bounds = []
    for _ in range(n_pixels):
        bounds.extend([
            (0, W - 1),  # x
            (0, H - 1),  # y
            (0, 1),      # R
            (0, 1),      # G
            (0, 1)       # B
        ])
    def batch_fitness_func(ga_instance, solutions, solutions_indices):
        solutions = np.asarray(solutions, dtype=np.float64)
        N = solutions.shape[0]
        perturbed = img.repeat(N, 1, 1, 1)
    
        params_t = torch.tensor(solutions, device=device, dtype=torch.float32)
        xs = torch.round(params_t[:, 0::5]).long().clamp(0, W - 1)
        ys = torch.round(params_t[:, 1::5]).long().clamp(0, H - 1)
        rs = params_t[:, 2::5].clamp(0, 1)
        gs = params_t[:, 3::5].clamp(0, 1)
        bs = params_t[:, 4::5].clamp(0, 1)

        batch_indices = torch.arange(N, device=device).repeat_interleave(n_pixels)
        y_flat = ys.flatten()
        x_flat = xs.flatten()

        perturbed[batch_indices, 0, y_flat, x_flat] = rs.flatten()
        perturbed[batch_indices, 1, y_flat, x_flat] = gs.flatten()
        perturbed[batch_indices, 2, y_flat, x_flat] = bs.flatten()

        with torch.no_grad():
            logits = model(normalize(perturbed))
            probs = F.softmax(logits, dim=1)
            target_probs = probs[:, target_class].cpu().numpy()

        return target_probs  
        
    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=10, 
        fitness_func=batch_fitness_func,
        fitness_batch_size=15,
        sol_per_pop=15,
        num_genes=len(bounds),
        gene_space=[{'low': low, 'high': high, 'step': None if low != high else 0} 
                    for low, high in bounds],
        mutation_by_replacement=True,
        random_mutation_min_val=0,
        random_mutation_max_val=1,
        gene_type=[int if i % 5 < 2 else float for i in range(len(bounds))],
        suppress_warnings=True
    )
    ga_instance.run()
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    adv_img = apply_n_pixel_attack(img, best_solution, n_pixels)
    return adv_img

def main():
    global device, model, IMAGENET_MEAN, IMAGENET_STD
    os.makedirs("GE", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    NUM_CLASSES = 100
    ARCH = CURRENT_ARCH
    CHECKPOINT_PATH = f"/kaggle/input/datasets/oliwianieradzik/models/{ARCH}.pt"

    model = Net(num_classes=NUM_CLASSES, arch=ARCH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    N_IMAGES = CURRENT_IMAGES
    N_PIXELS = CURRENT_N_PIXELS
    TARGET_CLASS = CURRENT_TARGET_CLASS
    BATCH_SIZE = N_IMAGES       

    _, testloader = get_data_loaders(batch_size=BATCH_SIZE, data_dir='./data', only_test=True)
    classes = get_classes()

    images, labels = next(iter(testloader))
    images = images.to(device)
    images = images * IMAGENET_STD + IMAGENET_MEAN
    images = torch.clamp(images, 0, 1)
    labels = labels.tolist()

    with torch.no_grad():
        logits = model(normalize(images))
        _, orig_preds = torch.max(logits, 1)
    orig_correct = (orig_preds.cpu().numpy() == np.array(labels)).sum()
    orig_acc = orig_correct / N_IMAGES * 100
    print(f"Original accuracy on {N_IMAGES} images: {orig_acc:.2f}%")

    print(f"Running attack (target class = {classes[TARGET_CLASS]}) on {N_IMAGES} images...")
    adv_images = []
    success_count = 0
    target_confidences = []
    attack_times = []

    for i in range(N_IMAGES):
        img = images[i:i+1]   
        t0 = time.time()
        adv_img = run_attack_on_image(img, TARGET_CLASS, N_PIXELS)
        t1 = time.time()
        attack_times.append(t1 - t0)
        adv_images.append(adv_img)

        adv_probs = predict(adv_img)
        adv_pred = np.argmax(adv_probs)
        if adv_pred == TARGET_CLASS:
            success_count += 1
        target_confidences.append(adv_probs[TARGET_CLASS])

        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{N_IMAGES} images")

    adv_tensor = torch.cat(adv_images, dim=0)   
    with torch.no_grad():
        logits = model(normalize(adv_tensor))
        _, adv_preds = torch.max(logits, 1)
    adv_correct = (adv_preds.cpu().numpy() == np.array(labels)).sum()
    adv_acc = adv_correct / N_IMAGES * 100

    success_rate = success_count / N_IMAGES * 100
    avg_target_conf = np.mean(target_confidences)
    avg_time = np.mean(attack_times)

    print("\n--- Results ---")
    print(f"Original accuracy: {orig_acc:.2f}%")
    print(f"Adversarial accuracy: {adv_acc:.2f}%")
    print(f"Targeted success rate (to class {classes[TARGET_CLASS]}): {success_rate:.2f}%")
    print(f"Average target confidence: {avg_target_conf:.4f}")
    print(f"Average attack time per image: {avg_time:.2f}s")

    model_name = ARCH
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"GE/attack_results_{model_name}_{N_PIXELS}_GA.txt"

    with open(filename, 'w') as f:
        f.write("=== One-Pixel Attack Efficiency Analysis ===\n")
        f.write(f"Model: {ARCH}\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Number of images: {N_IMAGES}\n")
        f.write(f"Number of pixels modified: {N_PIXELS}\n")
        f.write(f"Target class: {classes[TARGET_CLASS]} (index {TARGET_CLASS})\n\n")
        f.write(f"Original accuracy: {orig_acc:.2f}%\n")
        f.write(f"Adversarial accuracy: {adv_acc:.2f}%\n")
        f.write(f"Targeted success rate: {success_rate:.2f}%\n")
        f.write(f"Average target confidence: {avg_target_conf:.4f}\n")
        f.write(f"Average attack time per image: {avg_time:.2f} s\n")
        f.write(f"Total attack time: {sum(attack_times):.2f} s\n")

    print(f"\nResults saved to {filename}")

    N_PLOT = min(10, N_IMAGES)
    fig, axes = plt.subplots(2, N_PLOT, figsize=(2*N_PLOT, 5))
    for i in range(N_PLOT):
        show_image(axes[0, i], images[i:i+1], f"orig: {classes[orig_preds[i].item()]}")
        show_image(axes[1, i], adv_images[i], f"adv: {classes[adv_preds[i].item()]}")
    plt.tight_layout()
    plt.savefig(f"GE/attack_samples_{model_name}_{N_PIXELS}_GA.png")
    #plt.show()

if __name__ == "__main__":
    main()
