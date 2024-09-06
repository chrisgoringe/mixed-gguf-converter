from modules.comfy_flux import DoubleStreamBlock, SingleStreamBlock
import torch
from tqdm import trange

from typing import Union
from modules.utils import is_double, shared

def _new_layer(n) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    if is_double(n):
        return DoubleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn, qkv_bias=True)
    else:
        return SingleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn)

def _load_single_layer(layer_number:int, remove_from_sd=True) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    layer_sd = shared.layer_sd(layer_number, and_drop=remove_from_sd)
    layer:torch.nn.Module = _new_layer(layer_number)
    layer.load_state_dict(layer_sd)
    return layer

def load_layer_stack(dry_run=False):
    layer_stack = torch.nn.Sequential( *[_load_single_layer(layer_number=x) for x in trange(shared.last_layer+1)] )
    return layer_stack