# run.py
import subprocess, sys, os

print("\n🧱 Building agents (skips if unchanged)…")
subprocess.run([sys.executable, "build_agents.py"], check=False)

print("\n🚀 Evaluating product with cached agents…")
subprocess.run([sys.executable, "evaluate_product_async.py"], check=False)

print("\n📊 Running reducer…")
subprocess.run([sys.executable, "reduce.py"], check=False)
