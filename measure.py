from gguf import GGUFReader
import math, argparse, torch
from safetensors.torch import load_file

def measure(tensor_iter, count_function, byte_function) -> tuple[int,int]:
    bytes = 0
    parameters = 0
    for t in tensor_iter:
        parameters += count_function(t)
        bytes      += byte_function(t)
    return parameters, bytes

def measure_gguf(file) -> tuple[int,int]:
    r = GGUFReader(file)
    return measure(r.tensors, lambda t: math.prod(t.shape), lambda t:t.n_bytes)

TS = { torch.float8_e4m3fn : 1, torch.bfloat16 : 2, torch.float16 :2, torch.uint8 :1, torch.float32 : 4 }
def measure_safetensors(file) -> tuple[int,int]:
    sd = load_file(file)
    return measure( (sd[k] for k in sd), lambda t: math.prod(t.shape), lambda t: math.prod(t.shape) * TS[t.dtype] )

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--file', required=True)
    args = a.parse_args()
    if args.file.endswith("gguf"):
        parameters, bytes = measure_gguf(args.file)
    else:
        parameters, bytes = measure_safetensors(args.file)
    print(f"{args.file} has {8*bytes/parameters:>3.1f} bits per parameter (parameters: {int(parameters)})")
    


if __name__=='__main__': main()