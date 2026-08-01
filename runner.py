import os
import subprocess
import time
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from training.data_loader import get_data_loaders
#"resnet50", "resnet18","efficientnet_b0", "mobilenet_v3_small",
#,
architectures = ["densenet121"]
pixels = [1,10]
#, "genetic_test.py", "cmaes_test.py"
scripts = ["differential_evolution_test.py", "genetic_test.py", "cmaes_test.py"]
images = 200
target_class = 1

SCRIPT_INFO = {
    "differential_evolution_test.py": ("DE", "DE"),
    "genetic_test.py": ("GE", "GA"),
    "cmaes_test.py": ("CMAES", "CMAES")
}

def worker(task_and_gpu):
    task, gpu_id = task_and_gpu
    arch = task["arch"]
    pix = task["pix"]
    script = task["script"]
    
    folder, suffix = SCRIPT_INFO[script]
    expected_file = f"{folder}/attack_results_{arch}_{pix}_{suffix}.txt"

    if os.path.exists(expected_file):
        print(f"[GPU {gpu_id}] SKIPPING: {script} | ARCH={arch} | PIX={pix} (Already exists)")
        return

    print(f"\n[GPU {gpu_id}] STARTING: {script} | ARCH={arch} | PIX={pix}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["CURRENT_ARCH"] = arch
    env["CURRENT_N_PIXELS"] = str(pix)
    env["CURRENT_IMAGES"] = str(task["images"])
    env["CURRENT_TARGET_CLASS"] = str(task["target_class"])

    t0 = time.time()
    try:
        subprocess.run(["python", script], check=True, env=env)
        elapsed = time.time() - t0
        print(f"[GPU {gpu_id}] FINISHED: {script} | ARCH={arch} | PIX={pix} in {elapsed:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"[GPU {gpu_id}] ERROR: {script} | ARCH={arch} | PIX={pix} - {e}")

if __name__ == "__main__":
    os.makedirs("DE", exist_ok=True)
    os.makedirs("GE", exist_ok=True)
    os.makedirs("CMAES", exist_ok=True)
    
    print("Pre-downloading CIFAR-100 dataset to avoid process race conditions...")
    get_data_loaders(batch_size=1, data_dir='./data', only_test=True)
    
    all_tasks = []
    for arch, pix, script in product(architectures, pixels, scripts):
        all_tasks.append({
            "arch": arch,
            "pix": pix,
            "script": script,
            "images": images,
            "target_class": target_class
        })

    task_pairs = [(task, idx % 2) for idx, task in enumerate(all_tasks)]

    print(f"Total jobs to run: {len(task_pairs)} across 2 GPUs in parallel.\n")

    with ProcessPoolExecutor(max_workers=2) as executor:
        list(executor.map(worker, task_pairs))
