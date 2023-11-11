import os
import subprocess
import webbrowser

def run_tensorboard():
    # Path to the training logs directory
    log_dir = os.path.join(os.getcwd(), "training_logs")

    # Check if the training logs directory exists
    if not os.path.exists(log_dir):
        print("Error: Training logs directory does not exist.")
    else:
        # Path to the TensorBoard executable
        tensorboard_executable = "tensorboard"

        # Command to run TensorBoard with the specified log directory
        command = [tensorboard_executable, "--logdir=" + log_dir]

        # Launch TensorBoard using subprocess
        try:
            process = subprocess.Popen(command, shell=True)
            print("TensorBoard launched successfully.")

            # Open TensorBoard in the default web browser
            webbrowser.open("http://localhost:6006/")

            process.wait()

        except Exception as e:
            print("Error launching TensorBoard:", str(e))


if __name__ == "__main__":
    run_tensorboard()