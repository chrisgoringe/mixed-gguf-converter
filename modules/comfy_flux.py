import sys, os
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','..','ComfyUI_windows_portable','ComfyUI'))
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','..','ComfyUI'))

try:
    from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
except:
    print("The mixed-gguf-converter folder needs to be next to (in the same directory as) the ComfyUI or ComfyUI_windows_portable folder.")
    exit()