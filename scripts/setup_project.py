import os
import subprocess
import sys

def run_command(command):
    try:
        # Use sys.executable to ensure we use the same python that's running this script
        subprocess.check_call(f'"{sys.executable}" -m {command}', shell=True)
    except subprocess.CalledProcessError:
        print(f"❌ Error while running: {command}")
        return False
    return True

def main():
    print("🚀 --- Expert AI Project Setup ---")
    
    # DETERMINE PROJECT ROOT
    # This script is located in <ROOT>/scripts/setup_project.py
    script_path = os.path.abspath(__file__)     # .../scripts/setup_project.py
    script_dir = os.path.dirname(script_path)   # .../scripts
    project_root = os.path.dirname(script_dir)  # .../
    
    # Change working directory to Project Root so all paths work
    os.chdir(project_root)
    print(f"📂 Working in project root: {project_root}")

    # 1. Install Jupytext and other dependencies
    print("\n📦 Installing/Updating dependencies...")
    if not run_command("pip install -r requirements.txt"):
        print("⚠️ Pip install failed. If you use conda, make sure your environment is active.")
    
    # Files (relative to root)
    py_file = os.path.join("scripts", "notebook.py")
    ipynb_file = "notebook.ipynb"

    # 2. Check if notebook.py exists (the source of truth)
    if not os.path.exists(py_file):
        print(f"❌ Error: {py_file} not found! Something is wrong with the repo structure.")
        return

    # 3. Create the notebook file if it is missing (since it's gitignored)
    if not os.path.exists(ipynb_file):
        print(f"📓 Creating {ipynb_file} from {py_file}...")
        # explicitly output to the root directory
        run_command(f"jupytext --to ipynb --output {ipynb_file} {py_file}")

    # 4. Activate the pairing
    print("🔗 Linking the Notebook and the Python script...")
    # This format tells Jupytext to link the root ipynb with the py in scripts/
    run_command(f"jupytext --set-formats ipynb,scripts//py:percent {ipynb_file}")

    print("\n✅ ALL SET!")
    print(f"1. You can now open '{ipynb_file}' in Jupyter.")
    print(f"2. Every time you SAVE the notebook, the '{py_file}' file will update automatically.")
    print("3. ONLY commit and push the 'scripts/' folder to Git.")
    print("\n💡 Tip for VS Code users: Install the 'Jupytext Sync' extension.")

if __name__ == "__main__":
    main()
