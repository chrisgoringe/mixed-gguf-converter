
from modules.qtensor import QuantizedTensor
from modules.utils import shared, layer_iteratable_from_string, log
from modules.loader import load_layer_stack
from gguf import GGMLQuantizationType, GGUFWriter, GGUFReader, ReaderTensor
import torch
from argparse import ArgumentParser
import os, time
from tqdm import tqdm
from .measure import measure_file

CONFIGURATIONS = {
    "9_6" : { 
        'casts': [
            {'layers': '0-10',             'castto': 'BF16'},
            {'layers': '11-14, 54',        'castto': 'Q8_0'},
            {'layers': '15-36, 39-53, 55', 'castto': 'Q5_1'},
            {'layers': '37-38, 56',        'castto': 'Q4_1'},
        ],
        'notes': 'might just fit on a 16GB card'
    },
    "8_4" : { 
        'casts': [
            {'layers': '0-4, 10',      'castto': 'BF16'},
            {'layers': '5-9, 11-14',   'castto': 'Q8_0'},
            {'layers': '15-35, 41-55', 'castto': 'Q5_1'},
            {'layers': '36-40, 56',    'castto': 'Q4_1'},
        ],
        'notes': 'good balance for 16GB card'
    },
    "7_4" : {
        'casts': [
            {'layers': '0-2',                  'castto': 'BF16'},
            {'layers': '5, 7-12',              'castto': 'Q8_0'},
            {'layers': '3-4, 6, 13-33, 42-55', 'castto': 'Q5_1'},
            {'layers': '34-41, 56',            'castto': 'Q4_1'},
        ],
        'notes': 'roughly same size as 8bit model'
    },
    "5_9" : {
        'casts': [
            {'layers': '0-25, 27-28, 44-54', 'castto': 'Q5_1'},
            {'layers': '26, 29-43, 55-56',   'castto': 'Q4_1'},
        ],
        'notes': 'should work on 12GB card'
    },
    "5_1" : {
        'casts': [
            {'layers': '0-56', 'castto': 'Q4_1'},
        ],
        'notes': 'full Q4_1 quantization - smallest currently available'
    },
    "5_3" : {
        'casts': [
            {'layers': '0-2, 4, 6, 8-9, 11-13, 49-54', 'castto': 'Q5_1'},
            {'layers': '3, 5, 7, 10, 14, 16-29, 32-33, 42, 44-48, 55-56', 'castto': 'Q4_1'},
            {'layers': '15, 30-31, 34-41, 43', 'castto': 'patch:flux1-dev-Q3_K_S.gguf'},
        ],
        'notes': 'Requires a Q3_K_S quantized version to patch from '
    },
    # Insert configs created by optimization.py in here. Note that in python the indentation matters, so copy and paste with care
    "9_2" : {
        'casts': [
            {'layers': '0-8, 10, 12', 'castto': 'BF16'},
            {'layers': '9, 11, 13-21, 49-54', 'castto': 'patch:flux1-dev-Q6_K.gguf'},
            {'layers': '22-34, 41-48, 55', 'castto': 'patch:flux1-dev-Q5_K_S.gguf'},
            {'layers': '35-40', 'castto': 'patch:flux1-dev-Q4_K_S.gguf'},
            {'layers': '56', 'castto': 'Q4_1'},
        ],
        'notes': 'Perfect for 16GB cards'
    },
    # ------
}

HELP_TEXT = '''Produce a mixed gguf model from a flux safetensors. 
Usage:
python convert.py [--verbose=n] [--load [flux_model].safetensors] [--config x_x] [--config x_x] [--config x_x]

Default for load is "./flux1-dev.safetensors". Ideally this is the full 16bit version.

Default for config is to apply all available configurations.
Default for verbose is 1. Values 0-2.

Files will be saved in the same location as the loaded file, as '[flux_model]_mx[config].gguf'.
They can be loaded in Comfy using the nodes at https://github.com/city96/ComfyUI-GGUF by
putting them in your models/unet directory.

The config number represents the average bits per parameter across the full model.  

Configurations current available are:
''' + "\n".join(f"{x:>4} {CONFIGURATIONS[x]['notes']}" for x in CONFIGURATIONS)

VERBOSITY = 0

MODEL_BASE_DIR = None
def modelpath(model): return os.path.join(MODEL_BASE_DIR, model) if MODEL_BASE_DIR else model

def convert(infile, outfile, config):
    default_cast = 'F32'
    shared.set_model(modelpath(infile), dump_existing=True)
        
    log ("(a) Getting layer casts")
    layer_casts = [default_cast]*57
    for nod in config['casts']:
        cast = nod['castto']
        for layer_index in layer_iteratable_from_string(nod['layers']):
            layer_casts[layer_index] = cast

    log ("(b) Loading layer stack")
    layers = load_layer_stack()

    writer = GGUFWriter(modelpath(outfile), "flux", use_temp_file=True)
    patchfiles:dict[str, list[str]] = {}
    def write(key, tensor:torch.Tensor, cast_chooser:callable):
        cast:str = cast_chooser(key, tensor)
        if cast.startswith('patch:'):
            if (patchfile := cast[6:]) not in patchfiles: patchfiles[patchfile] = []
            patchfiles[patchfile].append(key)
        else:
            qtype = getattr(GGMLQuantizationType, cast)
            qt = QuantizedTensor.from_unquantized_tensor(tensor, qtype)
            writer.add_tensor(key, qt._tensor.numpy(), raw_dtype=qtype)
            writer.add_array(f"comfy.gguf.orig_shape.{key}", tensor.shape)
            log(f"{key:>50} {cast:<20}", log.DETAILS)
    
    log("(c) Casting leftovers")
    for key in shared.sd: 
        write(key, shared.sd[key], lambda a,b: default_cast )

    log("(d) Casting layers")
    for i, layer in enumerate(tqdm(layers)):
        cast = layer_casts[i]
        prefix = f"double_blocks.{i}." if i<19 else f"single_blocks.{i-19}."
        sd = layer.state_dict()
        def get_cast(key:str, tensor:torch.Tensor):
            if len(tensor.shape)==1 and tensor.shape[0]<2000:
                return default_cast
            else:
                return cast
        for key in sd: write( prefix+key, sd[key], get_cast)

    log(f"(e) Applying patches {[p in patchfiles]}")
    for patchfile, keys in patchfiles.items():
        reader = GGUFReader(modelpath(patchfile))
        tensor:ReaderTensor
        for tensor in tqdm(reader.tensors, desc=patchfile):
            if tensor.name in keys:
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
                log(f"{tensor.name:>50} {tensor.tensor_type.name:<20}", log.DETAILS)

    log(f"(f) Writing to {outfile}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    log(f"(g) Measuring {outfile}...", end="")
    p, b = measure_file(outfile)
    log(f" {8*b/p:>4.1f} bits per parameter")

def main():
    a = ArgumentParser(add_help=False)
    a.add_argument('-h', '--help', action="store_true")
    a.add_argument('--load', default="flux1-dev.safetensors")
    a.add_argument('--config', action="append", required=False, choices=[k for k in CONFIGURATIONS])
    a.add_argument('--verbose', type=int, default=1)
    a.add_argument('--model_dir', help="base directory for all models")

    args = a.parse_args()
    if args.help:
        print(HELP_TEXT)
        return
    
    global MODEL_BASE_DIR
    MODEL_BASE_DIR = args.model_dir
    
    log.set_log_level(args.verbose)
    
    configs = args.config or CONFIGURATIONS
    for i, config in enumerate(configs):
        outfile = (os.path.splitext(args.load)[0] + f"_mx{config}.gguf")
        log("----------------------------------------------------------------------------", log.ALWAYS)
        log(f"Conversion {i+1}/{len(configs)} - Converting {args.load} to {outfile}", log.ALWAYS)
        log("----------------------------------------------------------------------------", log.ALWAYS)
        convert(infile  = args.load, outfile = outfile, config  = CONFIGURATIONS[config])

if __name__=='__main__': main()
    
