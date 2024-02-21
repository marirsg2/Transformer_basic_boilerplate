from config import *
from datasets import load_dataset
from positional_encoding import *
from transformer_model import *
from support_functions import *
#-------------------------
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
#---------------------

train_data = load_dataset("wikitext", name="wikitext-2-v1")
# train_iter = (item for sub_dict in train_data["train"].__iter__() for item in sub_dict['text'])
train_iter = (sub_dict['text'] for sub_dict in train_data["train"].__iter__() if len(sub_dict['text']) > 0)
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# ``train_iter`` was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter = (sub_dict['text'] for sub_dict in train_data["train"].__iter__() if len(sub_dict['text']) > 0)
val_iter = (sub_dict['text'] for sub_dict in train_data["validation"].__iter__() if len(sub_dict['text']) > 0)
test_iter = (sub_dict['text'] for sub_dict in train_data["test"].__iter__() if len(sub_dict['text']) > 0)
train_data = data_process(train_iter,vocab,tokenizer) # the train data is a list of strings. Each string is a doc/para. see function for details
val_data = data_process(val_iter,vocab,tokenizer)
test_data = data_process(test_iter,vocab,tokenizer)


train_data = batchify(train_data, batch_size,device)  # shape ``[seq_len, batch_size]``
val_data = batchify(val_data, eval_batch_size,device)
test_data = batchify(test_data, eval_batch_size,device)
ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)


#====================================================================================
import time

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        output = model(data)
        output_flat = output.view(-1, ntokens)
        loss = criterion(output_flat, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

#====================================================================

best_val_loss = float('inf')
epochs = 3

with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model)
        val_loss = evaluate(model, val_data)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        scheduler.step()
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states

#===================================================================

test_loss = evaluate(model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)

