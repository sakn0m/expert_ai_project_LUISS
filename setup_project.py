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
    
    # 1. Install Jupytext and other dependencies
    print("\n📦 Installing/Updating dependencies...")
    if not run_command("pip install -r requirements.txt"):
        print("⚠️ Pip install failed. If you use conda, make sure your environment is active.")
        # We continue anyway to try and pair the notebook if jupytext is already there
    
    # Files
    py_file = "notebook.py"
    ipynb_file = "notebook.ipynb"

    # 2. Check if notebook.py exists (the source of truth)
    if not os.path.exists(py_file):
        print(f"❌ Error: {py_file} not found! Are you in the right folder?")
        return

    # 3. Create the notebook file if it is missing (since it's gitignored)
    if not os.path.exists(ipynb_file):
        print(f"📓 Creating {ipynb_file} from {py_file}...")
        run_command(f"jupytext --to ipynb {py_file}")

    # 4. Activate the pairing
    print("🔗 Linking the Notebook and the Python script...")
    run_command(f"jupytext --set-formats ipynb,py:percent {ipynb_file}")

    print("\n✅ ALL SET!")
    print(f"1. You can now open '{ipynb_file}' in Jupyter.")
    print("2. Every time you SAVE the notebook, the '.py' file will update automatically.")
    print("3. ONLY commit and push the '.py' file to Git.")
    print("\n💡 Tip for VS Code users: Install the 'Jupytext Sync' extension.")

if __name__ == "__main__":
    main()
