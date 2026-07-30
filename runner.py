import subprocess
import shutil
from itertools import product


#
architectures = ["resnet50","resnet18", "efficientnet_b0", "mobilenet_v3_small", "densenet121"]
pixels = [1,10]
images = 50
target_class = 1

original_config = "config.py"
backup_config = "config.py.bak"

shutil.copyfile(original_config, backup_config)

try:
    for arch, pix in product(architectures, pixels):
        print(f"\n=== Running with ARCH={arch}, PIXELS={pix} ===")
        
        with open(original_config, "w") as f:
            f.write(f"CURRENT_ARCH = '{arch}'\n")
            f.write(f"CURRENT_N_PIXELS = {pix}\n")
            f.write(f"CURRENT_IMAGES = {images}\n")
            f.write(f"CURRENT_TARGET_CLASS = {target_class}\n")
        #"genetic_test.py","differential_evolution_test.py" , "cmaes_test.py"
        for script in ["differential_evolution_test.py"]:
            print(f"--- {script} ---")
            subprocess.run(["python", script], check=True)
finally:
    shutil.move(backup_config, original_config)