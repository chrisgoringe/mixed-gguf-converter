
from modules.qtensor import QuantizedTensor
from modules.utils import shared, layer_iteratable_from_string, log
from modules.loader import load_layer_stack
from gguf import GGMLQuantizationType, GGUFWriter, GGUFReader, ReaderTensor
import torch
from argparse import ArgumentParser
import os, time
from tqdm import tqdm
from measure import measure_file
from configurations import configurations

HELP_TEXT = '''Produce a mixed gguf model from a flux safetensors. 

Files will be saved in the same location as the loaded file, as '[flux_model]_mx[config].gguf'.
They can be loaded in Comfy using the nodes at https://github.com/city96/ComfyUI-GGUF by
putting them in your models/unet directory.

The config number represents the average bits per parameter across the full model.  

Configurations current available can be found with convert.py --list
''' 

VERBOSITY = 0



def convert(infile, outfile, config):
    default_cast = 'F32'
    shared.set_model(configurations.modelpath(infile), dump_existing=True)
        
    log ("(a) Getting layer casts")
    layer_casts = [default_cast]*57
    for nod in config['casts']:
        cast = nod['castto']
        for layer_index in layer_iteratable_from_string(nod['layers']):
            layer_casts[layer_index] = cast

    log ("(b) Loading layer stack")
    layers = load_layer_stack()

    writer = GGUFWriter(configurations.modelpath(outfile), "flux", use_temp_file=True)
    patchfiles:dict[str, list[str]] = {}
    def write(key, tensor:torch.Tensor, cast_chooser:callable):
        cast:str = cast_chooser(key, tensor)
        if not configurations.available_natively(cast):
            if cast not in patchfiles: patchfiles[cast] = []
            patchfiles[cast].append(key)
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
    for cast, keys in patchfiles.items():
        reader = GGUFReader(configurations.patchpath(cast))
        tensor:ReaderTensor
        for tensor in tqdm(reader.tensors, desc=cast):
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
    a = ArgumentParser(description=HELP_TEXT)
    a.add_argument('--load', default="flux1-dev.safetensors", help="Base model to convert")
    a.add_argument('--patch', help="A model from which patches can be extracted (only need to specify one)")

    c = a.add_mutually_exclusive_group()
    c.add_argument('--config', action="append", choices=configurations.as_list, help="Which configuration(s) to run (can be used multiple times)")
    c.add_argument('--all', action="store_true", help="Use all configurations")

    a.add_argument('--verbose', type=int, default=1, help="Verbosity 0, 1 or 2")
    a.add_argument('--model_dir', help="base directory for all models")
    a.add_argument('--list', action="store_true", help="List configuration options available")

    args = a.parse_args()

    if args.list:
        print(configurations.all_as_string_with_notes)
        return
    else:
        if not (args.all or args.config):
            print("One of --h, --list, --all or --config is required")
            return
    
    configurations.base_dir     = args.model_dir
    configurations.base_patcher = args.patch
    
    log.set_log_level(args.verbose)
    
    configs = args.config if not args.all else configurations.as_list
    for i, config in enumerate(configs):
        outfile = (os.path.splitext(args.load)[0] + f"_mx{config}.gguf")
        log("----------------------------------------------------------------------------", log.ALWAYS)
        log(f"Conversion {i+1}/{len(configs)} - Converting {args.load} to {outfile}", log.ALWAYS)
        log("----------------------------------------------------------------------------", log.ALWAYS)
        convert(infile  = args.load, outfile = outfile, config = configurations.configuration(config))

if __name__=='__main__': 
    main()
    
