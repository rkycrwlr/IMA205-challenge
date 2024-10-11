import torch

INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256
BATCH_SIZE_SEG = 64
NUM_EPOCHS_SEG = 200
NUM_EPOCHS_CLS = 100
BATCH_SIZE_CLS = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Your working dir here, you should have the Train and Test datasets, the two metadata csv and the SampleSubmission csv in this folder
WORKING_DIR = "/home/ids/epaquin-22/challenge" 