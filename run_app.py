import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def launch_app():
    """Launch the Streamlit application"""
    print("Launching Burn Analysis Application...")
    os.system("streamlit run burn_analysis_app.py")

if __name__ == "__main__":
    try:
        install_requirements()
        launch_app()
    except Exception as e:
        print(f"Error: {e}")
        print("Please install requirements manually: pip install -r requirements.txt")
