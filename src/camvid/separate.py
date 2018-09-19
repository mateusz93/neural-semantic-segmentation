import os
import shutil


mapping = {
    'splits/train.txt': ('X/train/data', 'y/train/data'),
    'splits/val.txt': ('X/val/data', 'y/val/data'),
    'splits/test.txt': ('X/test/data', 'y/test/data'),
}


def make_dir_overwrite(_dir: str):
    """Make a directory and overwrite if it already exists."""
    shutil.rmtree(_dir, ignore_errors=True)
    os.makedirs(_dir)


for files, (X, y) in mapping.items():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    X = os.path.join(this_dir, X)
    y = os.path.join(this_dir, y)
    make_dir_overwrite(X)
    make_dir_overwrite(y)
    with open(files, 'r') as files:
        files = [line.replace('\n', '') for line in files]
    for file in files:
        file_X_from = os.path.join(this_dir, 'X/data', file)
        file_X_to = os.path.join(X, file)
        shutil.copy(file_X_from, file_X_to)
        file = file.replace('.png', '_L.png')
        file_y_from = os.path.join(this_dir, 'y/data', file)
        file_y_to = os.path.join(y, file)
        shutil.copy(file_y_from, file_y_to)

