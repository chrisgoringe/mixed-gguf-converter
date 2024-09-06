from huggingface_hub import HfFileSystem
from functools import partial
import os

fs = HfFileSystem()

rpath = lambda a: "/".join(( "ChrisGoringe", "MixedQuantFlux", a))

class Callback:
    def __init__(self, f):
        self.filename = f
        self.size = None
        self.done = None

    def set_size(self, s): 
        self.size = s
        self.done = 0

    def relative_update(self, n):
        self.done += n
        print(f"\r{self.filename} {self.done}/{self.size}", end='')

for f in os.listdir("d:/MixedQuantFlux"):
    if f.endswith('gguf'):
        local = os.path.join("d:/MixedQuantFlux", f)
        remote = rpath(f)
        fs.put_file(lpath=local, rpath=remote, callback=Callback(f))