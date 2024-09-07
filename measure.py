from gguf import GGUFReader, ReaderTensor
import math, argparse

def measure(file):
    r = GGUFReader(file)
    t:ReaderTensor
    bytes = 0
    parameters = 0
    for t in r.tensors:
        bytes += t.n_bytes
        parameters += math.prod(t.shape)
    print(f"{file} has {8*bytes/parameters:>3.1f} bits per parameter")

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--file', required=True)
    args = a.parse_args()
    measure(args.file)

if __name__=='__main__': main()