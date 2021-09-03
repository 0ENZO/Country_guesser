import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path_to_dll = os.path.join(ROOT_DIR, "mlp_library/cmake-build-debug/mlp_library.dll").replace(os.sep, '/')
SAVE_FOLDER = os.path.join(ROOT_DIR, "models/").replace(os.sep, '/')
DATASET_FOLDER = os.path.join(ROOT_DIR, "dataset").replace(os.sep, '/')

classes = ["jordanie", "palestine", "soudan"]
IMAGE_SIZE = 32

