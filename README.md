# Mixed GGUF converter

This repository must be installed next to an install of ComfyUI to work. 
In otherwords, relative to the directory where this file is, `../ComfyUI` or `../ComfyUI_windows_portable` need to exist.

It should be run in the ComfyUI virtual environment.

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