# Mixed GGUF converter

This repository must be installed next to an install of ComfyUI to work. 
In otherwords, relative to the directory where this file is, `../ComfyUI` or `../ComfyUI_windows_portable` need to exist.

It should be run in the ComfyUI virtual environment. You may need to install or update gguf:

```
pip install gguf --upgrade
```

A collection of already converted files can be found in my [hugging face](https://huggingface.co/ChrisGoringe/MixedQuantFlux) repository.

## Usage

```
python convert.py [-h] [--load LOAD] [--patch PATCH] [--config X_Y] [--verbose VERBOSE] [--model_dir MODEL_DIR] [--list]
```

- `--model_dir` optional: the directory all models are found in
- `--load` the model to be converted. 
    - Should be a local file `[flux_model].safetensors`, and a full 16 bit verion (ideally). 
    - Default is `flux1-dev.safetensors`, in the directory specified by `--model_dir`
- `--config` used one or more times to specify the configurations to be used. 
    - Each will be saved in a file `[flux_model]_mx[config].gguf`
    - `--config all` will generate every available configuration
- `--list` prints a list of available configurations and exits
- `--verbose` for n = 0,1,2 controls the output verbosity
    - Default is 1
- `--patch` one of the required patch file (optional, if required but not provided convert will try to work it out)

The output files will be in the same directory as the file loaded, and named `[base_name]_mxX_Y.gguf`
They should be placed in `models/unet` and can be loaded with [GGUF Loader Node](https://github.com/city96/ComfyUI-GGUF).

## Configurations

Configs are of the form `Y_Z` (Y and Z being digits) for an average number of bits per parameter of `Y.Z`. 

At time of writing, `python convert.py --list` shows:
```
   3_8 Good for 8Gb card? (requires patch file(s) for Q4_K_S, Q3_K_S)
   5_1 full Q4_1 quantization. 
   5_3 pretty small (requires patch file(s) for Q3_K_S)
   5_9 should work on 12GB card.
   6_6 Comfortable on 12 GB cards
   6_9 Good choice for 12GB cards
   7_3 Aiming for 12GB cards (requires patch file(s) for Q6_K, Q5_K_S, Q4_K_S)
   7_4 roughly same size as 8bit model.
   7_6 Too big for 12GB cards (requires patch file(s) for Q6_K, Q5_K_S, Q4_K_S)
   8_2 Added using --gb 11.0 --q all (requires patch file(s) for Q6_K, Q5_K_S, Q4_K_S)
   8_4 comfortable for 16GB card.
   8_6 Added using --gb 10.0 --q all (requires patch file(s) for Q6_K, Q5_K_S, Q4_K_S)
   9_2 Best for 16GB cards (requires patch file(s) for Q6_K, Q5_K_S, Q4_K_S)
   9_6 might just fit on a 16GB card.
```

Some configurations require one or more 'patch files'. This is because only some
quantizations (Q8_0, Q5_1, Q4_1) can be done natively by convert.py.

A patch file is a version of the original model that has been converted to gguf using other tools.
The process (in summary) is:

- convert the model to gguf in full precision 
    - (use city96/ComfyUI-GGUF) `python tools/convert.py --src flux1-dev.safetensors --dst flux1-dev-BF16.ggufq`
- compile llama-quantize (see city96/ComfyUI-GGUF/tools on how to do this)
- convert the full precision model to another quant 
    - `llama-quantize flux1-dev-BF16.gguf Q3_K`
    - `mv xxx flux1-dev-Q3_K.gguf`

Otherwise, just use the configurations that don't need a patch!

## Creating new configurations

If you want to try a different model reductions, you can add new configurations with `optimization.py`.

```
python optimization.py [-h] --gb GB [--q Q] [--add] [--add_as ADD_AS]
```

- `--gb gb` Attempt to remove apprpximately gb Gb from the full (22GB) model.
- `--q Q` by default, `optimization.py` will not assume there are patches available. If you have some, you can indicate that they are available with `--q Q4_K_S` (for instance). `--q all` will allow any quantization to be used.
- `--add` add this configuration to the database (`configurations.yaml`) so `convert.py` can use it
- `--add_as` use this name for the configuration (optional, default is `NEW`)

Normal workflow would be:
- try optimization.py a few times with different values of `--gb`
- run it with `--add`
- run `convert.py` with `--config NEW`
- at the end of the `convert.py` run it will tell you the actual bits per parameter, and suggest using `configurations.py` to give `NEW` a better name
- do that, and rename the output file, and you are good to go!

## Managing configurations

```
python configurations.py [-h] [--rename RENAME | --remove REMOVE | --sort | --notes NOTES]
```

- `--rename OLD:NEW` to rename a configuraion
- `--remove NAME` to remove one
- `--sort` to sort them into size order (generally gets done automatically)
- `--notes NAME:notes` to change the notes associated with the configuration

You can also just edit configurations.yaml directly!

## How it works...

The script works by calculating a sequence of possible quantizations, sorting them from the lowest to highest values of `error_induced / bits_saved` (from the costs directory), and then applies them in order until the desired number of GB have been removed.
