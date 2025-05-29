import os
import shutil

class Util:
    @staticmethod
    def remove_folder_content(results_dir):
        """Remove all files in the specified directory.

        Args:
            results_dir (str): The directory from which to remove all files.
        """
        # only remove files if the folder contains files, not if the folder exists but is empty
        if os.path.exists(results_dir) and os.listdir(results_dir):
            for filename in os.listdir(results_dir):
                file_path = os.path.join(results_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
