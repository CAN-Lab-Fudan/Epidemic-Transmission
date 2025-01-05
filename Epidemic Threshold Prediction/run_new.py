import os
from tqdm import tqdm

if __name__ == "__main__":
    all_experiments = [
        "experiments3_1.py",
        "experiments3_2.py",
        # "experiments2_3.py",
        # "experiments2_4.py",
        # "experiments1_1.py",
        # "experiments1_2.py",
        # "experiments1_3.py",
        # "experiments4_3.py",
    ]

    for python_file in tqdm(all_experiments):
        os.system(command="python " + python_file)