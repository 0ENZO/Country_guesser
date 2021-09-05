import os
import warnings
warnings.filterwarnings("ignore")

CLASSES = ["jordanie", "palestine", "soudan"]
IMAGE_SIZE = 32

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_TO_DLL = os.path.join(ROOT_DIR, "mlp_library/cmake-build-debug/mlp_library.dll").replace(os.sep, '/')

DATASET_FOLDER = os.path.join(ROOT_DIR, "dataset").replace(os.sep, '/')
DATASET_FOLDER2 = os.path.join(ROOT_DIR, "resized_dataset").replace(os.sep, '/')
DATASET_FOLDER3 = os.path.join(ROOT_DIR, "second_dataset").replace(os.sep, '/')

SAVE_FOLDER = os.path.join(ROOT_DIR, "models/").replace(os.sep, '/')
SAVE_FOLDER2 = os.path.join(ROOT_DIR, "models/grid_search_v2/").replace(os.sep, '/')
SAVE_FOLDER3 = os.path.join(ROOT_DIR, "models/grid_search_v3/").replace(os.sep, '/')

TRAIN_FOLDER = os.path.join(DATASET_FOLDER2, "train")
TEST_FOLDER = os.path.join(DATASET_FOLDER2, "test")

TRAIN_FOLDER2 = os.path.join(DATASET_FOLDER2, "train")
TEST_FOLDER2 = os.path.join(DATASET_FOLDER2, "test")

TRAIN_FOLDER3 = os.path.join(DATASET_FOLDER3, "train")
TEST_FOLDER3 = os.path.join(DATASET_FOLDER3, "test")

TRAIN_FIRST_FOLDER = os.path.join(TRAIN_FOLDER, CLASSES[0])
TRAIN_SECOND_FOLDER = os.path.join(TRAIN_FOLDER, CLASSES[1])
TRAIN_THIRD_FOLDER = os.path.join(TRAIN_FOLDER, CLASSES[1])

TEST_FIRST_FOLDER = os.path.join(TEST_FOLDER, CLASSES[0])
TEST_SECOND_FOLDER = os.path.join(TEST_FOLDER, CLASSES[1])
TEST_THIRD_FOLDER = os.path.join(TEST_FOLDER, CLASSES[2])

MLP_0HNL = "models_by_hands/2M_80e4_32px_2hl_3n_63p_03_09_23_05"
MLP_1HNL_8N = "models_by_hands/2M_80e4_32px_2hl_3n_63p_03_09_23_05"
MLP_1HNL_32 = "models_by_hands/2M_80e4_32px_2hl_3n_63p_03_09_23_05"
MLP_2HNL_32 = "models_by_hands/2M_80e4_32px_2hl_3n_63p_03_09_23_05"
