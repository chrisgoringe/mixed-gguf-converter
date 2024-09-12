from gguf import GGUFReader
import math, argparse, torch, os
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

def measure_file(file):
    if file.endswith("gguf"):
        parameters, bytes = measure_gguf(file)
    else:
        parameters, bytes = measure_safetensors(file)
    return parameters, bytes

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--load', required=True)
    a.add_argument('--model_dir', help="base directory for all models")
    args = a.parse_args()
    file_load = os.path.join(args.model_dir, args.load) if args.model_dir else args.load

    parameters, bytes = measure_file(file_load)

    print(f"{args.load} has {8*bytes/parameters:>3.1f} bits per parameter (parameters: {int(parameters)})")
    


if __name__=='__main__': main()