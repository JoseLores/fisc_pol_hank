import subprocess
import time

scripts = ["tax_progressivity_figures.py", "table_3.py", "transfers_irfs.py",
           "heatmap_policy.py", "params_robustness.py", "deficit_shocks.py", "heatmap_sticky.py"]

for i, script in enumerate(scripts, start=1):
    start = time.time()

    # run the script and wait until it finishes
    subprocess.run(['python', script])

    elapsed = time.time() - start
    print(
        f"Script {i} of {len(scripts)} ({script}) executed in {elapsed:.2f} seconds.")
