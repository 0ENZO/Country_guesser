import os
import warnings
warnings.filterwarnings("ignore")

CLASSES = ["jordanie", "palestine", "soudan"]
IMAGE_SIZE = 24

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_TO_DLL = os.path.join(ROOT_DIR, "mlp_library/cmake-build-debug/mlp_library.dll").replace(os.sep, '/')

DATASET_FOLDER2 = os.path.join(ROOT_DIR, "resized_dataset").replace(os.sep, '/')
SAVE_FOLDER = os.path.join(ROOT_DIR, "models/").replace(os.sep, '/')

TRAIN_FOLDER2 = os.path.join(DATASET_FOLDER2, "train")
TEST_FOLDER2 = os.path.join(DATASET_FOLDER2, "test")

TRAIN_FOLDER = os.path.join(DATASET_FOLDER2, "train")
TEST_FOLDER = os.path.join(DATASET_FOLDER2, "test")

TRAIN_FIRST_FOLDER = os.path.join(TRAIN_FOLDER, CLASSES[0])
TEST_FIRST_FOLDER = os.path.join(TEST_FOLDER, CLASSES[0])

TRAIN_SECOND_FOLDER = os.path.join(TRAIN_FOLDER, CLASSES[1])
TEST_SECOND_FOLDER = os.path.join(TEST_FOLDER, CLASSES[1])

TEST_THIRD_FOLDER = os.path.join(TEST_FOLDER, CLASSES[2])
TRAIN_THIRD_FOLDER = os.path.join(TRAIN_FOLDER, CLASSES[1])

MLP_0HNL = "MLP_0hl_24px_0.01a_200e_500it_90.2acc_81.2test_acc_06_09_15_42"
MLP_1HNL_8N = "MLP_1hl_8n_24px_0.01a_400e_300it_89.6acc_81.2test_acc_06_09_12_37"
MLP_1HNL_32 = "MLP_1hl_32n_24px_0.01a_225e_400it_89.6acc_82.4test_acc_06_09_14_54"
MLP_2HNL_32 = "MLP_2hl_32n_24px_0.01a_225e_400it_89.2acc_83test_acc_06_09_14_30"

# OLD
CLASSES_AFI = ["allemagne", "france", "italie"]

DATASET_FOLDER = os.path.join(ROOT_DIR, "dataset").replace(os.sep, '/')
DATASET_FOLDER3 = os.path.join(ROOT_DIR, "second_dataset").replace(os.sep, '/')

SAVE_FOLDER2 = os.path.join(ROOT_DIR, "models/grid_search_v2/").replace(os.sep, '/')
SAVE_FOLDER3 = os.path.join(ROOT_DIR, "models/grid_search_v3/").replace(os.sep, '/')

TRAIN_FOLDER3 = os.path.join(DATASET_FOLDER3, "train")
TEST_FOLDER3 = os.path.join(DATASET_FOLDER3, "test")





