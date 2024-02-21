from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Assume we have some training data
train_data = ["This is the first sentence.", "This is another sentence.", "Yet another sentence."]

# Define the tokenizer
tokenizer = get_tokenizer('basic_english')

# Tokenize the training data
train_iter = [tokenizer(item) for item in train_data]
flat_iter = (item for sublist in train_iter for item in sublist)

# Build the vocabulary
vocab = build_vocab_from_iterator(map(tokenizer, flat_iter), specials=['<unk>'])

# Now you can use the vocab to convert tokens to indices
# For example, to convert the first sentence to indices:
indices = [vocab[token] for token in train_iter[0]]
print(indices)