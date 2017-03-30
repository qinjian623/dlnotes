import lstm
import mxnet.bucket_io


def read_content(path):
    with open(path) as ins:
        return ins.read()


# Return a dict which maps each char into an unique int id
def build_vocab(path):
    content = list(read_content(path))
    idx = 1  # 0 is left for zero-padding
    the_vocab = {}
    for word in content:
        if len(word) == 0:
            continue
        if word not in the_vocab:
            the_vocab[word] = idx
            idx += 1
    return the_vocab


# Encode a sentence with int ids
def text2id(sentence, the_vocab):
    words = list(sentence)
    return [the_vocab[w] for w in words if len(w) > 0]


# build char vocabluary from input
vocab = build_vocab("./obama.txt")
print('vocab size = %d' % (len(vocab)))


# Each line contains at most 129 chars.
seq_len = 129
# embedding dimension, which maps a character to a 256-dimension vector
num_embed = 256
# number of lstm layers
num_lstm_layer = 3
# hidden unit in LSTM cell
num_hidden = 512

symbol = lstm.lstm_unroll(
    num_lstm_layer,
    seq_len,
    len(vocab) + 1,
    num_hidden=num_hidden,
    num_embed=num_embed,
    num_label=len(vocab) + 1,
    dropout=0.2)

"""test_seq_len"""
data_file = open("./obama.txt")
for line in data_file:
    assert len(line) <= seq_len + 1, "seq_len is smaller than maximum line length. \
    Current line length is %d. Line content is: %s" % (len(line), line)

data_file.close()

# The batch size for training
batch_size = 32

# initalize states for LSTM
init_c = [('l%d_init_c' % l, (batch_size, num_hidden))
          for l in range(num_lstm_layer)]
init_h = [('l%d_init_h' % l, (batch_size, num_hidden))
          for l in range(num_lstm_layer)]
init_states = init_c + init_h

# Even though BucketSentenceIter supports various length examples,
# we simply use the fixed length version here
data_train = bucket_io.BucketSentenceIter(
    "./obama.txt",
    vocab,
    [seq_len],
    batch_size,
    init_states,
    seperate_char='\n',
    text2id=text2id,
    read_content=read_content)
