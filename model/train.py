import itertools
import numpy
import os
import os.path
import pdb
import pickle
import random
import re
import sys
import torch
from torch import nn, optim
import tqdm
import transformers
import uuid

import encoder
import decoder

#pdb.signal.signal = lambda *_: None

TRAINING_SET = 0.9
MINIBATCH_SIZE = 12
LEARNING_RATE = 1e-3
CHECKPOINT_AT = 1
DEVICE = "cpu"
MFCC_DIR = "4096"

device = torch.device(DEVICE)
cpu = torch.device("cpu")

tokenizer = transformers.GPT2Tokenizer.from_pretrained(decoder.MODEL_SIZE)
repeated_whitespace = re.compile("(\s)\s+")
unicode_subs = str.maketrans({
    '«': '"',
    '\xad': '-',
    '´': "'",
    '»': '"',
    '÷': '/',
    'ǀ': '|',
    'ǃ': '!',
    'ʹ': "'",
    'ʺ': '"',
    'ʼ': "'",
    '˄': '^',
    'ˆ': '^',
    'ˈ': "'",
    'ˋ': '`',
    'ˍ': '_',
    '˜': '~',
    '̀': '`',
    '́': "'",
    '̂': '^',
    '̃': '~',
    '̋': '"',
    '̎': '"',
    '̱': '_',
    '̲': '_',
    '̸': '/',
    '։': '":',
    '׀': '|',
    '׃': '":',
    '٪': '%',
    '٭': '*',
    '\u200b': ' ',
    '‐': '-',
    '‑': '-',
    '‒': '-',
    '–': '-',
    '—': '-',
    '―': '-',
    '‖': '|',
    '‗': '_',
    '‘': "'",
    '’': "'",
    '‚': ',',
    '‛': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‟': '"',
    '′': "'",
    '″': '"',
    '‴': "'",
    '‵': '`',
    '‶': '"',
    '‷': "'",
    '‸': '^',
    '‹': '<',
    '›': '>',
    '‽': '?',
    '⁄': '/',
    '⁎': '*',
    '⁒': '%',
    '⁓': '~',
    '\u2060': ' ',
    '⃥': '\\',
    '−': '-',
    '∕': '/',
    '∖': '\\',
    '∗': '*',
    '∣': '|',
    '∶': ':',
    '∼': '~',
    '≤': '<',
    '≥': '>',
    '≦': '<',
    '≧': '>',
    '⌃': '^',
    '〈': '<',
    '〉': '>',
    '♯': '#',
    '✱': '*',
    '❘': '|',
    '❢': '!',
    '⟦': '[',
    '⟨': '<',
    '⟩': '>',
    '⦃': '{',
    '⦄': '}',
    '〃': '"',
    '〈': '<',
    '〉': '>',
    '〛': ']',
    '〜': '~',
    '〝': '"',
    '〞': '"',
    '\ufeff': ' '
})


def clean(string):
    cleaned = repeated_whitespace.sub("\\1", string.translate(unicode_subs))
    return torch.tensor(tokenizer.encode(cleaned), device=cpu, dtype=torch.long)


def debug_hook(graph_var, i):
    if hasattr(graph_var, "next_functions"):
        graph_var = graph_var.next_functions[i]

    graph_var = graph_var[0]

    def f(*inp, var=graph_var):
        pdb.set_trace()
        return

    graph_var.register_hook(f)


def debug_walk_graph(root):
    agenda = [root]
    leaves = []

    while len(agenda) > 0:
        parent = agenda.pop()

        for child, _ in parent.next_functions:
            if child:
                try:
                    leaf = child.variable
                    leaves.append((parent, child, leaf))
                except AttributeError:
                    agenda.append(child)

    return leaves


if __name__ == "__main__":
    model_id = str(uuid.uuid1())

    try:
        encode, key_transform, value_transform = torch.load(sys.argv[2])
        
        print("Loading from checkpoint...")
        encode = encode.to(device)
        decode = decoder.Decoder(encoder.MFCC_SIZE << 1, key=key_transform, value=value_transform).to(device)
    except (FileNotFoundError, IndexError):
        encode = encoder.Encoder().to(device)
        decode = decoder.Decoder(encoder.MFCC_SIZE << 1).to(device)

    max_caption_length = 0
    max_mfcc_length = 0

    captions = []
    masks = []

    mfccs = []
    mfcc_lengths = []

    print("\nLoading dataset...")

    for folder, _, files in os.walk(sys.argv[1]):
        for f in tqdm.tqdm(files):
            if f.endswith(".txt"):
                with open(folder + "/" + f, "r") as captionfile:
                    caption = clean(captionfile.read())
                    max_caption_length = max(max_caption_length, len(caption))

                    captions.append(caption)

                with open(MFCC_DIR + "/" + os.path.splitext(f)[0] + ".mfcc", "rb") as mfccfile:
                    mfcc = torch.tensor(pickle.load(mfccfile), device=cpu)
                    max_mfcc_length = max(max_mfcc_length, mfcc.shape[0])

                    mfccs.append(mfcc)

    try:
        seed = int(sys.argv[3])
    except (ValueError, IndexError):
        seed = random.getrandbits(32)

    print("Shuffling seed: " + str(seed))

    random.seed(seed)
    random.shuffle(captions)

    random.seed(seed)
    random.shuffle(mfccs)

    split = int(len(captions) * TRAINING_SET)
    print("Training begin: " + str(split) + " train/" + str(len(captions) - split) + " test\n")

    train = optim.Adam(itertools.chain(encode.parameters(), decode.parameters()), lr=LEARNING_RATE)

    for i in range(len(captions)):
        caption_length = len(captions[i])
        mfcc_length = len(mfccs[i])

        captions[i] = torch.cat((captions[i], -torch.ones(max_caption_length - caption_length, device=cpu, dtype=torch.long)))
        masks.append(torch.cat((torch.ones(caption_length, device=cpu), torch.zeros(max_caption_length - caption_length, device=cpu))))

        mfccs[i] = torch.cat((mfccs[i], torch.zeros(max_mfcc_length - mfcc_length, encoder.MFCC_SIZE, device=cpu)))
        mfcc_lengths.append(mfcc_length)

    captions = torch.stack(captions)
    masks = torch.stack(masks)

    mfccs = torch.stack(mfccs)
    mfcc_lengths = torch.tensor(mfcc_lengths, device=device)

    try:
        checkpoints_hit = 0

        for epoch in itertools.cycle(range(CHECKPOINT_AT)):
            loss_diagnostic = 0

            for i in tqdm.trange(0, split, MINIBATCH_SIZE):
                batch_size = min(split - i, MINIBATCH_SIZE)

                caption = captions.narrow(0, i, batch_size).to(device)
                mask = masks.narrow(0, i, batch_size).to(device)

                padded_mfcc = mfccs.narrow(0, i, batch_size).to(device)
                lengths = mfcc_lengths.narrow(0, i, batch_size)
                mfcc = nn.utils.rnn.pack_padded_sequence(padded_mfcc, lengths=lengths, batch_first=True, enforce_sorted=False)

                train.zero_grad()

                context = encode.forward(mfcc)
                loss = decode.forward(context, labels=(caption, mask))[0]

                loss_diagnostic += float(loss)
                loss.backward()
                train.step()

                del loss
                del context
                del mfcc
                del lengths
                del padded_mfcc
                del mask
                del caption

            print("Epoch: " + str(loss_diagnostic))

            if epoch == CHECKPOINT_AT - 1:
                checkpoints_hit += 1
                print("\nCheckpoint " + str(checkpoints_hit) + "\n")
                torch.save((encode, decode.psuedo_self.key_transform, decode.psuedo_self.value_transform), model_id + "_" + str(seed) + "_" + str(checkpoints_hit) + ".pt")
    except KeyboardInterrupt:
        pdb.set_trace()
