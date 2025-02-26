import os
import subprocess

def init_dvc():
    # Initialize DVC
    subprocess.run(["dvc", "init"]) # Will setup the necessary DVC files and directories

    # Create .dvc/config
    os.makedirs(".dvc", exist_ok=True)
    # Make sure that the dvc directory exists and create a config file
    with open(".dvc/config", "w") as f:
        f.write("[core]\n    remote = myremote\n[remote \"myremote\"]\n    url = ./dvc-storage")

        """The f.write() function writes the specified configuration settings to the .dvc/config file. 
           This configures DVC to use a local directory (./dvc-storage) as the remote storage location named myremote.
        """
    # Create remote storage
    os.makedirs("dvc-storage", exist_ok=True)

    # Add data directory to DVC
    subprocess.run(["dvc", "add", "data/raw"])


if __name__ == "__main__":
    init_dvc()
# Will ensure that the init_dvc() func will be executed only if run direclty, and not when
# imported as a module