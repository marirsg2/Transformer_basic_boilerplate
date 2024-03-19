import os
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformer_model import *
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
#-------------------------
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
#---------------------
from config import *
from datasets import load_dataset
from support_functions import *

class WrapperTransformerModel(pl.LightningModule):
    def __init__(self, ntokens, d_model, nhead, d_hid, nlayers, dropout):
        super(WrapperTransformerModel, self).__init__()
        self.ntokens = ntokens
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntokens, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntokens)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(src.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

    def training_step(self, batch, batch_idx):
        src, tgt = batch #these are now given in a seq_len (train param) x Batch size
        src,tgt = src.permute(1,0),tgt.permute(1,0)
        output = self(src)
        output_flat = output.view(-1, self.ntokens) # this and tgt.view(-1) helps cross entropy compare the token distrib (After softmax) to the target token
        loss = F.cross_entropy(output_flat, tgt.reshape(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        src,tgt = src.permute(1,0),tgt.permute(1,0)
        # src.reshape((-1,eval_batch_size))
        output = self(src)
        output_flat = output.view(-1, self.ntokens)
        loss = F.cross_entropy(output_flat, tgt.reshape(-1))
        # loss = F.cross_entropy(output.view(-1, output.size(-1)), tgt.view(-1))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        src, tgt = batch
        src,tgt = src.permute(1,0),tgt.permute(1,0)
        output = self(src)
        output_flat = output.view(-1, self.ntokens)
        loss = F.cross_entropy(output_flat, tgt.reshape(-1))
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)
        

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
    #     scheduler = StepLR(optimizer, 1.0, gamma=0.95)
    #     return [optimizer], [scheduler]


class WikiTextDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, eval_batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = get_tokenizer('basic_english')
        self.prepare_data()

    def prepare_data(self):
        self.train_data = load_dataset("wikitext", name="wikitext-2-v1")
        self.train_iter = (sub_dict['text'] for sub_dict in self.train_data["train"].__iter__() if len(sub_dict['text']) > 0)
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, self.train_iter), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_iter = (sub_dict['text'] for sub_dict in self.train_data["train"].__iter__() if len(sub_dict['text']) > 0)
            val_iter = (sub_dict['text'] for sub_dict in self.train_data["validation"].__iter__() if len(sub_dict['text']) > 0)
            self.train_data = data_process(self.train_iter, self.vocab, self.tokenizer)
            self.val_data = data_process(val_iter, self.vocab, self.tokenizer)

    def train_dataloader(self):
        data_raw_batches = batchify(self.train_data,self.batch_size,device)
        data_src_target_list = []
        for batch, i in enumerate(range(0, data_raw_batches.size(0) - 1, bptt)): 
            #the batch above is just the batch_index, it is really bad naming. i is really the text_data_index
            #each unit of train_Data is a long sequence length) X batch size; note NOT CONTEXT LENGTH, just length to divide data into batch_size units
            # i is iterating over the context length.
            # each get_batch returns the src and target as offset by 1 over the long length, BUT limited to seq_length of 35 (Training param)
            # since the train_data is in length X batch_size, each offset will also be in batch size. The length will be limited to max 35
            data_batch, targets_batch = get_batch(data_raw_batches, i) #the code from pytorch puts it in batches, rather than let dataloader do it
            targets_batch = targets_batch.reshape((-1,self.batch_size))
             #rather than let dataloader do it. note the targets are single values. For each sequence, one target token
            flattened_list_pairs = [(data_batch[:,x],targets_batch[:,x]) for x in range(data_batch.shape[1]) if data_batch[:,x].shape[0] == bptt]
            data_src_target_list += flattened_list_pairs
        return DataLoader(data_src_target_list, batch_size=self.batch_size, shuffle=True) #num_workers=3

    def val_dataloader(self):
        data_raw_batches = batchify(self.val_data,self.eval_batch_size,device)
        data_src_target_list = []
        for batch, i in enumerate(range(0, data_raw_batches.size(0) - 1, bptt)): 
            #the batch above is just the batch_index, it is really bad naming. i is really the text_data_index
            #each unit of train_Data is a long sequence length) X batch size; note NOT CONTEXT LENGTH, just length to divide data into batch_size units
            # i is iterating over the context length.
            # each get_batch returns the src and target as offset by 1 over the long length, BUT limited to seq_length of 35 (Training param)
            # since the train_data is in length X batch_size, each offset will also be in batch size. The length will be limited to max 35
            data_batch, targets_batch = get_batch(data_raw_batches, i) #the code from pytorch puts it in batches,
            targets_batch = targets_batch.reshape((-1,self.eval_batch_size))
             #rather than let dataloader do it. note the targets are single values. For each sequence, one target token
            flattened_list_pairs = [(data_batch[:,x],targets_batch[:,x]) for x in range(data_batch.shape[1]) if data_batch[:,x].shape[0] == bptt]
            data_src_target_list += flattened_list_pairs
        return DataLoader(data_src_target_list, batch_size=self.eval_batch_size, shuffle=True) #num_workers=3

    def test_dataloader(self):
        test_iter = (sub_dict['text'] for sub_dict in self.train_data["test"].__iter__() if len(sub_dict['text']) > 0)
        test_data = data_process(test_iter, self.vocab, self.tokenizer)
        data_raw_batches = batchify(test_data,self.eval_batch_size,device)
        data_src_target_list = []
        for batch, i in enumerate(range(0, data_raw_batches.size(0) - 1, bptt)): 
            #the batch above is just the batch_index, it is really bad naming. i is really the text_data_index
            #each unit of train_Data is a long sequence length) X batch size; note NOT CONTEXT LENGTH, just length to divide data into batch_size units
            # i is iterating over the context length.
            # each get_batch returns the src and target as offset by 1 over the long length, BUT limited to seq_length of 35 (Training param)
            # since the train_data is in length X batch_size, each offset will also be in batch size. The length will be limited to max 35
            data_batch, targets_batch = get_batch(data_raw_batches, i) #the code from pytorch puts it in batches, rather than let dataloader do it
            targets_batch = targets_batch.reshape((-1,self.eval_batch_size))
             #rather than let dataloader do it. note the targets are single values. For each sequence, one target token
            flattened_list_pairs = [(data_batch[:,x],targets_batch[:,x]) for x in range(data_batch.shape[1]) if data_batch[:,x].shape[0] == bptt]
            data_src_target_list += flattened_list_pairs
        return DataLoader(data_src_target_list, batch_size=self.eval_batch_size, shuffle=True) #num_workers=3

batch_size = 20
eval_batch_size = 10
bptt = 35
lr = 5.0


data_module = WikiTextDataModule(batch_size, eval_batch_size)
model = WrapperTransformerModel(len(data_module.vocab), emsize, nhead, d_hid, nlayers, dropout)
trainer = pl.Trainer(max_epochs=3)
trainer.fit(model, data_module)
