import os
import subprocess
import time
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from training.data_loader import get_data_loaders
GENERATION_ARCH = "resnet50" 
EVALUATION_ARCHITECTURES = ["efficientnet_b0", "mobilenet_v3_small", "densenet121", "resnet50", "resnet18"]
PIXELS = [1, 10]
IMAGES = 200
TARGET_CLASS = 1
scripts = ["cmaes_test.py", "genetic_test.py", "differential_evolution_test.py"]
images = 200
target_class = 1

SCRIPT_INFO = {
    "differential_evolution_test.py": ("DE", "DE"),
    "genetic_test.py": ("GE", "GA"),
    "cmaes_test.py": ("CMAES", "CMAES")
}

def run_generation(script, pixel_count):
    folder, suffix = SCRIPT_INFO[script]
    generation_file = f"{folder}/adversarial_images_{GENERATION_ARCH}_{pixel_count}_pixels.pt"

    if os.path.exists(generation_file):
        print(f"SKIPPING generation: {suffix} | {pixel_count} pixels (already exists)")
        return

    print(f"\nGenerating {suffix} attacks with {pixel_count} pixels using {GENERATION_ARCH}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["CURRENT_ARCH"] = GENERATION_ARCH
    env["GENERATION_ARCH"] = GENERATION_ARCH
    env["CURRENT_N_PIXELS"] = str(pixel_count)
    env["CURRENT_IMAGES"] = str(IMAGES)
    env["CURRENT_TARGET_CLASS"] = str(TARGET_CLASS)
    env["MODE"] = "generate"

    subprocess.run(["python", script], check=True, env=env)
    print(f"Generation completed: {suffix} | {pixel_count} pixels")


def run_evaluation(task_and_gpu):
    task, gpu_id = task_and_gpu
    arch = task["arch"]
    pix = task["pix"]
    script = task["script"]
    folder, suffix = SCRIPT_INFO[script]

    expected_file = f"{folder}/attack_results_{arch}_{pix}_{suffix}.txt"
    if os.path.exists(expected_file):
        print(f"[GPU {gpu_id}] SKIPPING: {script} | ARCH={arch} | PIX={pix} (already exists)")
        return

    print(f"\n[GPU {gpu_id}] Evaluating {suffix} on {arch} with {pix} pixels (attack generated on {GENERATION_ARCH})")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["CURRENT_ARCH"] = arch
    env["GENERATION_ARCH"] = GENERATION_ARCH
    env["CURRENT_N_PIXELS"] = str(pix)
    env["CURRENT_IMAGES"] = str(IMAGES)
    env["CURRENT_TARGET_CLASS"] = str(TARGET_CLASS)
    env["MODE"] = "evaluate"

    t0 = time.time()
    subprocess.run(["python", script], check=True, env=env)

if __name__ == "__main__":
    os.makedirs("DE", exist_ok=True)
    os.makedirs("GE", exist_ok=True)
    os.makedirs("CMAES", exist_ok=True)
    
    for script, pix in product(scripts, PIXELS):
            run_generation(script, pix)

    all_tasks = []
    for arch, pix, script in product(EVALUATION_ARCHITECTURES, PIXELS, scripts):
        all_tasks.append({"arch": arch, "pix": pix, "script": script})

    task_pairs = [(task, idx % 2) for idx, task in enumerate(all_tasks)]

    with ProcessPoolExecutor(max_workers=2) as executor:
        list(executor.map(run_evaluation, task_pairs))
