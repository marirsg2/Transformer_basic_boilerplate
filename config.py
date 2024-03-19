import torch
import math 
batch_size = 20
eval_batch_size = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bptt = 35 # back prop through time, here it means it considers 35 time steps or seq len
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability