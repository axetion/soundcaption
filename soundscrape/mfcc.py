import aubio
import numpy
import os
import os.path
import pickle
import subprocess
import sys

SAMPLE_RATE = "44100"
TMP = "/tmp/mfcc.wav"
WINDOW_SIZE = 4096
MFCC_SIZE = 64
HOP_SIZE = WINDOW_SIZE >> 1


def resample_file(f):
    try:
        os.remove(TMP)
    except OSError:
        pass

    try:
        result = subprocess.check_output(["ffmpeg", "-v", "error", "-i", f, "-ac", "1", "-ar", SAMPLE_RATE, TMP], stderr=subprocess.STDOUT)
        return b"Error" not in result, result
    except subprocess.CalledProcessError:
        return False, b""


def process(frame, pvoc, mfcc):
    frequencies = pvoc(numpy.pad(frame, (0, HOP_SIZE - len(frame)), mode="constant"))
    return numpy.nan_to_num(mfcc(frequencies))


if __name__ == "__main__":
    success = 0
    fail = 0

    for infile in sys.argv[2:]:
        outfilename = sys.argv[1] + "/" + os.path.splitext(os.path.basename(infile))[0] + ".mfcc"

        if os.path.exists(outfilename):
            continue

        print(infile)

        ok, msg = resample_file(infile)
        if not ok:
            sys.stdout.buffer.write(msg)
            print("ffmpeg didn't like that...")
            fail += 1
            continue

        with open(outfilename, "wb") as outfile:
            with aubio.source(TMP, hop_size=HOP_SIZE) as snd:
                pvoc = aubio.pvoc(WINDOW_SIZE, HOP_SIZE)
                mfcc = aubio.mfcc(buf_size=WINDOW_SIZE, n_coeffs=MFCC_SIZE, samplerate=snd.samplerate)

                pickle.dump(numpy.vstack([process(frame, pvoc, mfcc) for frame in snd]), outfile)
                success += 1

    print(str(success) + " files processed, " + str(fail) + " files rejected")
