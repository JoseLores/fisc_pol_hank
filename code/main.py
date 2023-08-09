import subprocess
import time
import os

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the current working directory to the directory of the script
os.chdir(current_dir)

scripts = ["tax_progressivity_figures.py", "table_3.py", "transfers_irfs.py",
           "heatmap_policy.py", "params_robustness.py", "deficit_shocks.py", "heatmap_sticky.py"]

# table and irf do not generate output??

for i, script in enumerate(scripts, start=1):
    start = time.time()

    # run the script and wait until it finishes
    subprocess.run(['python', script])

    elapsed = time.time() - start
    print(
        f"Script {i} of {len(scripts)} ({script}) executed in {elapsed:.2f} seconds.")
