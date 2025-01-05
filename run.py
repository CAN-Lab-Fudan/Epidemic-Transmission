import os
from tqdm import tqdm


if __name__ == '__main__':
    prefix = "experiments4_"
    subfix = '.py'

    python_files = []

    for i in range(1, 6):
        curstr = prefix+str(i)+subfix
        python_files.append(curstr)

    for python_file in tqdm(python_files):
        os.system(command='python ' + python_file)
