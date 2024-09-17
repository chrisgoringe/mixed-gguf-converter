import requests
import os, shutil
from image_grid import image_grid
from PIL import Image
from tqdm import tqdm

QUANTS = ["default",] + [ f"model_{q}" for q in ["9_6", "9_2", "8_4", "7_6", "7_4", "6_9", "5_9", "5_1" ] ]
PIDS = list(range(8))

def localname(to_dir, quant, prompt_id): 
    return os.path.join(to_dir,f"{quant}-{prompt_id}.png")

def download(to_dir="images"):
    if not os.path.exists(to_dir): os.makedirs(to_dir)
    for prompt_id in PIDS:
        for quant in QUANTS:
            rpath = f"https://huggingface.co/ChrisGoringe/MixedQuantFlux/resolve/main/examples/{quant}-prompt_{prompt_id}.png"
            lpath = localname(to_dir, quant, prompt_id)
            r = requests.get(rpath, stream=True)
            if r.status_code == 200:
                with open(lpath, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)

def make_grid(from_dir="images"):
    for prompt_id in tqdm(PIDS):
        ig = image_grid(inp=[localname(from_dir, quant, prompt_id) for quant in QUANTS], 
                        rows=3, columns=3, sort=False)
        Image.fromarray(ig).save(f"comparisons/prompt_{prompt_id}.png")

if __name__=='__main__':
    make_grid()