import os
import shutil

class CleanSweep:
    def __init__(self):
        self.clean_results_folder()

    def clean_results_folder(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "../Results")
        if os.path.exists(results_dir):
            for item in os.listdir(results_dir):
                item_path = os.path.join(results_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            
            print(f"Cleaned the Results folder: {results_dir}\n\n")
        else:
            print(f"Results directory {results_dir} does not exist.\n\n")

if __name__ == "__main__":
    CleanSweep()
