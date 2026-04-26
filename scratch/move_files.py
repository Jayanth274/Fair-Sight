import shutil
import os

files_to_move = [
    "GEMINI.md", "test_fairsight.py", "test_scanner.py", "download_datasets.py", 
    "adult.csv", "german_credit.csv", "bank_marketing.csv", "iris.csv"
]
source_dir = r"c:\Users\SriSaiJayanth\Desktop\Fair Sight"
dest_dir = r"C:\Users\SriSaiJayanth\Desktop"

for file in files_to_move:
    src = os.path.join(source_dir, file)
    dst = os.path.join(dest_dir, file)
    if os.path.exists(src):
        try:
            shutil.move(src, dst)
            print(f"Successfully moved: {file}")
        except Exception as e:
            print(f"Failed to move {file}: {str(e)}")
    else:
        print(f"File not found: {file}")
