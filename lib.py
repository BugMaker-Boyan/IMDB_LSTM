import pickle
import torch

ws = pickle.load(open('./model/ws.pkl', 'rb'))

max_len = 200

BATCH_SIZE = 512
TEST_BATCH_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYER = 2
BIDIRECTIONAL = True
DROPOUT = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
