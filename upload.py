from huggingface_hub import HfFileSystem
import os
import argparse

remote = lambda a : f"ChrisGoringe/MixedQuantFlux/{a}"

def upload(local_filepath):
    remote_filepath = remote( os.path.basename(local_filepath) )
    hf = HfFileSystem()
    hf.put_file(lpath=local_filepath, rpath=remote_filepath)
 
def main():
    a = argparse.ArgumentParser()
    b = a.add_mutually_exclusive_group(required=True)
    b.add_argument('--dir', help="Upload all gguf files in this directory")
    b.add_argument('--file', help="Upload this file")

    a = a.parse_args()

    if a.dir:
        files = [os.path.join(a.dir,f) for f in os.listdir(a.dir) if os.path.splitext(f)[-1]=='.gguf']
        if not files: raise Exception(f"No .gguf files in {a.dir}")
    else:
        files = [a.file,]
        if not os.path.exists(a.file): raise Exception(f"{a.file} not found")
        if not os.path.splitext(a.file)[-1]=='.gguf': raise Exception(f"{a.file} not a .gguf file")

    for file in files:
        upload(file)

if __name__=='__main__': main()