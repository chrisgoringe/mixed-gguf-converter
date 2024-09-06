# Mixed GGUF converter

This repository must be installed next to an install of ComfyUI to work. 
In otherwords, relative to the directory where this file is, `../ComfyUI` or `../ComfyUI_windows_portable` need to exist.

It should be run in the ComfyUI virtual environment.

A collection of already converted files can be found in my [hugging face](https://huggingface.co/ChrisGoringe/MixedQuantFlux) repository.

## Usage

```
python convert.py [-h] [--verbose=n] [--load [flux_model].safetensors] [--config xx_x] [--config xx_x] [--config xx_x]
```

- `--verbose` for n = 0,1,2 controls the output verbosity
    - Default is 1
- `--load` the model to be converted. 
    - Should be a local file `[flux_model].safetensors`, and a full 16 bit verion (ideally). 
    - Default is `./flux1-dev.safetensors`
- `--config` used zero or more times to specify the configurations to be used. 
    - Each will be saved in a file `[flux_model]_mx[config].gguf`
    - Default is to apply all available configurations in turn.
- `-h` can be used to list the help, including the config options available (and no conversion is done)

Configs are of the form `XY_Z` or `Y_Z` (X, Y and Z being digits), and represent the approximate reduction in GB of the 22GB model. So 14_1
produces a model just under 8GB in size.

The output files should be places in `models/unet` and can be loaded with [GGUF Loader Node](https://github.com/city96/ComfyUI-GGUF)

## Optimiser

A script to produce configs. WIP.

## Future considerations

- measurement of inference speed for different quants
- possible inclusion of `torch.float8_e3m4fn` (significantly less accurate that GGUF, but also faster)
- work out how to include other (more recent, better) quants like `Q5_K_S`, `Q5_K_S`, `Q5_K_S`
- document the optimiser script and integrate it so it produces configs