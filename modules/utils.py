import os, time
from safetensors.torch import load_file
from functools import partial
import torch
from typing import Iterable

filepath = partial(os.path.join,os.path.split(__file__)[0],"..")

class FluxFacts:
    last_layer = 56
    first_double_layer = 0
    last_double_layer  = 18
    first_single_layer = 19
    last_single_layer  = 56    
    bits_at_bf16       = 190422221824

class SingletonAddin:
    _instance = None
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def layer_list_from_string(s:str) -> list[int]:
    return list(layer_iteratable_from_string(s))

def layer_iteratable_from_string(s:str) -> Iterable[int]:
    if isinstance(s, int): return [s,]
    if s.lower()=='all':    return range(FluxFacts.last_layer+1)
    if s.lower()=='double': return range(FluxFacts.first_double_layer, FluxFacts.last_double_layer+1)
    if s.lower()=='single': return range(FluxFacts.first_single_layer, FluxFacts.last_single_layer+1)

    def parse():
        for section in (x.strip() for x in str(s or "").split(',')):
            if section:
                a,b = (int(x.strip()) for x in section.split('-')) if '-' in section else (int(section), int(section))
                for i in range(a,b+1): yield i
    return parse()

class SharedSD(SingletonAddin, FluxFacts):
    def __init__(self):
        self._model_path                                 = None
        self._sd:dict[str,torch.Tensor]                  = None
        self._layerssd:dict[str,dict[str,torch.Tensor]]  = None

    def set_model(self, model_path, dump_existing=True):
        self._model_path = model_path
        if dump_existing:
            self._sd       = None
            self._layerssd = None

    def _block_prefix(self, layer_index):
        if is_double(layer_index):
            return f"double_blocks.{layer_index}."
        else:
            return f"single_blocks.{layer_index-FluxFacts.first_single_layer}."
        
    def _remove_prefixes(self):
        prefixes = ['model.diffusion_model.',]
        def clean(s:str): 
            for prefix in prefixes:
                if s.startswith(prefix):
                    log(f"Removing prefix {prefix}", log.DETAILS, once_only=True)
                    s = s[len(prefix):] 
            return s
        self._sd = { clean(k):self._sd[k] for k in self._sd  }       

    @property
    def sd(self):
        if self._sd is None: 
            assert self._model_path, "No model path set"
            self._sd = load_file(self._model_path)
            self._remove_prefixes()
        return self._sd

    def layer_sd(self, layer_index, and_drop=False):
        if self._layerssd is None: self._split_sd()
        r = self._layerssd[self._block_prefix(layer_index)]
        if and_drop:
            for k in self._layerssd.pop(self._block_prefix(layer_index)): self._sd.pop(self._block_prefix(layer_index)+k)
        return r
    
    def _split_sd(self):
        self._layerssd = { self._block_prefix(x):{} for x in range(self.last_layer+1) }
        for k in self.sd:
            for pf in self._layerssd:
                if k.startswith(pf): 
                    self._layerssd[pf][k[len(pf):]] = self.sd[k]
                    break

class Log(SingletonAddin):
    ALWAYS    = 0
    MOSTLY    = 1
    DETAILS = 2
    RARELY    = 3
    def __init__(self):
        self.start_time = time.monotonic()
        self.level = self.MOSTLY
        self.onces = []

    def set_log_level(self, level:int):
        self.level = level

    def __call__(self, message, level=1, once_only=False):
        if self.level>=level: 
            if once_only:
                if message in self.onces: return
                self.onces.append(message)
            print(f"{time.monotonic()-self.start_time:>8.1f}s : {message}")
   
shared = SharedSD.instance()
log = Log.instance()

def is_double(layer_number): return (layer_number <= FluxFacts.last_double_layer)

