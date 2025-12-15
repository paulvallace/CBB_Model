import subprocess
import sys

scripts = [
    "/Users/PaulVallace/Desktop/College Basketball/Model/gameday.py",
    "/Users/PaulVallace/Desktop/College Basketball/Model/kenpom_automate.py",
    "/Users/PaulVallace/Desktop/College Basketball/Model/past_games_automate.py",  
    "/Users/PaulVallace/Desktop/College Basketball/Model/CBB_Model.py"
    "/Users/PaulVallace/Desktop/College Basketball/Model/train_model.py",
]

for script in scripts:
    print(f"\n================ Running {script} ================\n")
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"❌ {script} failed. Stopping pipeline.")
        break

print("\n✅ Pipeline finished")
