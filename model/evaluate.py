import aubio
import nltk
import numpy
import pickle
import random
import torch
import tqdm
import transformers
import sys

import encoder
import decoder
import train

WINDOW_SIZE = 4096
MFCC_SIZE = 64
HOP_SIZE = WINDOW_SIZE >> 1


def process(frame, pvoc, mfcc):
    frequencies = pvoc(numpy.pad(frame, (0, HOP_SIZE - len(frame)), mode="constant"))
    return numpy.nan_to_num(mfcc(frequencies))


if __name__ == "__main__":
    encode, key_transform, value_transform = torch.load(sys.argv[1])

    encode = encode.to(train.device)
    decode = decoder.Decoder(MFCC_SIZE << 1, key=key_transform, value=value_transform).to(train.device)

    if sys.argv[2] == "length":
        with aubio.source(sys.argv[4], hop_size=HOP_SIZE) as snd:
            pvoc = aubio.pvoc(WINDOW_SIZE, HOP_SIZE)
            mfcc = aubio.mfcc(buf_size=WINDOW_SIZE, n_coeffs=MFCC_SIZE, samplerate=snd.samplerate)
            mfccs = torch.tensor(numpy.vstack([process(frame, pvoc, mfcc) for frame in snd]), device=train.device).unsqueeze(1)

            context = encode.forward(mfccs)
            caption = decode.forward(context, length=int(sys.argv[3]))[0]

            print(train.tokenizer.decode(caption.tolist()))
    elif sys.argv[2] == "test":
        pass
