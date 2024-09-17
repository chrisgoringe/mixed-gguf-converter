import requests
import os, shutil
from image_grid import image_grid
from PIL import Image
from tqdm import tqdm

QUANTS = ["default",] + [ f"model_{q}" for q in ["9_6", "9_2", "8_4", "7_6", "7_4", "6_9", "5_9", "5_1" ] ]
PIDS = list(range(8))

def image_remoteurl(quant, prompt_id): f"https://huggingface.co/ChrisGoringe/MixedQuantFlux/resolve/main/examples/{quant}-prompt_{prompt_id}.png"
def image_localname(quant, prompt_id): return os.path.join(os.path.dirname(__file__),"images",f"{quant}-{prompt_id}.png")
def grid_localname(prompt_id):         return os.path.join(os.path.dirname(__file__),         f"prompt_{prompt_id}.png")

def download():
    for prompt_id in PIDS:
        for quant in QUANTS:
            r = requests.get(image_remoteurl(quant, prompt_id), stream=True)
            if r.status_code == 200:
                with open(image_localname(quant, prompt_id), 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)

def make_grid():
    for prompt_id in tqdm(PIDS):
        ig = image_grid(inp=[image_localname(quant, prompt_id) for quant in QUANTS], 
                        rows=3, columns=3, sort=False)
        Image.fromarray(ig).save(grid_localname(prompt_id))

if __name__=='__main__':
    make_grid()