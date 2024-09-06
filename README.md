# Mixed GGUF converter

This repository must be installed next to an install of ComfyUI to work. 
In otherwords, relative to the directory where this file is, `../ComfyUI` or `../ComfyUI_windows_portable` need to exist.

It should be run in the ComfyUI virtual environment. You may need to install or update gguf:

```
pip install gguf --update
```

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

```
Configurations current available are:
 9_0 might just fit on a 16GB card
10_6 good balance for 16GB card
12_0 roughly same size as 8bit model
14_1 should work on 12GB card
15_2 full Q4_1 quantization - smallest currently available
```

The output files should be placed in `models/unet` and can be loaded with [GGUF Loader Node](https://github.com/city96/ComfyUI-GGUF).

Be aware this can take a long time to run, but it gives you progress reports...

## Creating new configurations

If you want to try a different model reduction, you can quickly create it with `optimization.py`

To make an optimization that removes `xx.yy` GB from the model:

```
python optimization.py --gb xx.yy
```

The output will look like this:

```
Full model is 22.043 GB
 4.161 GB saved at cost of 38.600

    "4_2" : {
        'casts': [
            {'layers': '0-23', 'castto': 'BF16'},
            {'layers': '24-38, 40-55', 'castto': 'Q8_0'},
            {'layers': '39, 56', 'castto': 'Q5_1'},
        ],
        'notes': 'replace this with a comment!'
    },
```
Ignoring the first two lines (and the blank line), you have a configuration option that can be added to `convert.py` as instructed in the code (around line 50).

### Note

This example used `--gb 4.1` - but it actually removed 4.16, so is named `4_2`. The script works by calculating a sequence of possible quantizations, sorting them from the lowest to highest values of `error_induced / bits_saved`, and then applies them in order until the desired number of GB have been removed.

### GB

I use `1 GB = 2^30 bytes`, not `1GB = 10^3 bytes`. There's about a 7% difference - that's why I report the model as 22.043GB, not 24.

## Future considerations

- measurement of inference speed for different quants
- possible inclusion of `torch.float8_e3m4fn` (significantly less accurate that GGUF, but also faster)
- work out how to include other (more recent, better) quants like `Q5_K_S`, `Q5_K_S`, `Q5_K_S`