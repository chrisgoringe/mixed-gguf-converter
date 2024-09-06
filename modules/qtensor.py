import torch
import numpy as np
from gguf import GGMLQuantizationType, quants
from gguf.gguf_reader import ReaderTensor
from gguf.quants import quantize, dequantize
from functools import partial

class QuantizedTensor():
    def __init__(self, data=None, tensor_type=None, tensor_shape=None, patches=[], data_is_unquantized_tensor=False, **kwargs):
        self.tensor_type:GGMLQuantizationType = tensor_type
        self.tensor_shape:torch.Size          = tensor_shape
        self.patches:list                     = patches.copy()
        self._tensor:torch.Tensor             = None
        self._set_data(data, data_is_unquantized_tensor)
        self._cached = None

    def dequantized(self, dtype, device=None) -> torch.Tensor:
        if self._cached is None: self._cached = dequantize_tensor(self, dtype, device)
        return self._cached
    
    def purge(self): self._cached = None

    @property
    def tensor_description(self):
        try:
            return torch.tensor([int(self.tensor_type),] + [int(x) for x in self.tensor_shape], device="cpu")
        except:
            raise Exception()
    
    @classmethod
    def load_from_description(cls, description, tnsr):
        return QuantizedTensor( data=tnsr, tensor_type=int(description[0]), tensor_shape=torch.Size(description[1:]))
    
    @classmethod
    def load_from_reader_tensor(cls, reader_tensor:ReaderTensor):
        return QuantizedTensor( data=reader_tensor.data, tensor_type=reader_tensor.tensor_type, tensor_shape=torch.Size(np.flip(list(reader_tensor.shape))))

    @classmethod
    def from_unquantized_tensor(cls, tnsr:torch.Tensor, tensor_type:GGMLQuantizationType):
        return QuantizedTensor( tnsr, tensor_shape=tnsr.shape, tensor_type=tensor_type, data_is_unquantized_tensor=True )
    
    def _set_data(self, data, data_is_unquantized_tensor=False):
        if data_is_unquantized_tensor:
            assert isinstance(data, torch.Tensor)
            try:              data = quantize(data.numpy(), qtype=self.tensor_type)
            except TypeError: data = quantize(data.to(torch.float).numpy(), qtype=self.tensor_type)
        self._tensor = data if isinstance(data, torch.Tensor) or data is None else torch.as_tensor(data)  

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for a in args:
            if isinstance(a, QuantizedTensor):
                return_qt = QuantizedTensor(None, a.tensor_type, a.tensor_shape, a.patches)
                break

        args = [getattr(a, '_tensor', a) for a in args]
        return_qt._set_data( func(*args, **kwargs) )
        return return_qt
    
    def wrap(self, fn, *args, **kwargs):
        x = fn(*args, **kwargs)
        return QuantizedTensor(x, self.tensor_type, self.tensor_shape, self.patches) if isinstance(x, torch.Tensor) else x
    
    def __getattr__(self, __name: str):
        a = getattr(self._tensor, __name)
        return partial(self.wrap, a) if hasattr(a,'__call__') else a


def quantise_tensor(t:torch.Tensor, gtype:GGMLQuantizationType) -> QuantizedTensor:
    if t is None: return None
    t = t.to(torch.float32)
    return QuantizedTensor(quants.quantize(t.squeeze().numpy(), gtype), tensor_type=gtype, tensor_shape=t.shape)

def dequantize_tensor(tensor:QuantizedTensor, dtype, device):
    if tensor is None: return None
    if tensor.tensor_type in [GGMLQuantizationType.F32, GGMLQuantizationType.F16]:
        out = tensor._tensor
    else:
        out = dequantize(tensor._tensor, tensor.tensor_type, tensor.tensor_shape, dtype=dtype)
    return out.to(dtype).to(device)
