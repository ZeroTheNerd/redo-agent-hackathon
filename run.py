import subprocess
print("\n Running MAP step...")
subprocess.run(["python", "map_agents.py"])
print("\n Running REDUCE step...")
subprocess.run(["python", "reduce.py"])
