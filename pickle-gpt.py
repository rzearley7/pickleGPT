##########################################
"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        #print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------


# get default config and overrides from the command line, if any
config = get_config()
config.merge_from_args(sys.argv[1:])
#print(config)
setup_logging(config)
set_seed(config.system.seed)

# construct the training dataset
text = open('./pickle.txt', 'r').read() # don't worry we won't run out of file handles
train_dataset = CharDataset(config.data, text)

# construct the model
config.model.vocab_size = train_dataset.get_vocab_size()
config.model.block_size = train_dataset.get_block_size()
model = GPT(config.model)

model.load_state_dict(torch.load("./pickleGPT/out/chargpt/model.pt"))

# construct the trainer object
trainer = Trainer(config.trainer, model, train_dataset)


model.eval()

context = input("Enter your prompt (\"quit\" to quit): ")
while context != "quit":
  x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
  y = model.generate(x, 1000, temperature=1.0, do_sample=True, top_k=10)[0]
  # changed 500 to 20   ^
  completion = ''.join([train_dataset.itos[int(i)] for i in y])
  print(completion)
  context = input("Enter your prompt (\"quit\" to quit): ")


#################################