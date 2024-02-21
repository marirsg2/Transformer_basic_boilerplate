
# ? IS it truly an encoder-only? doesn't seem so

Look at the mask, is it a causal attention mask, that is usually the give away

# Building the vocab
The function
    build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
Takes a FLATTENED list. The map function applies tokenizer to each element in train iter, which itself is an iterator
over a flattened list.