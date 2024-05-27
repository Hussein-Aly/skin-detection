import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'sfa')

NUM_TRAIN_SKIN = 51000  # 46541
NUM_TRAIN_NOT_SKIN = 80000  # 79200
NUM_TEST_SKIN = 0  # 11880
NUM_TEST_NOT_SKIN = 0  # 19800
NUM_VAL_SKIN = 1951
NUM_VAL_NOT_SKIN = 1620

BATCH_SIZE = 32
NUM_EPOCHS = 1