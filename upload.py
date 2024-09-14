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
    a.add_argument('--model_dir', help="base directory for all models")

    a = a.parse_args()

    if a.dir:
        directory = os.path.join(a.model_dir, a.dir) if a.model_dir else a.dir
        files = [os.path.join(directory,f) for f in os.listdir(directory) if os.path.splitext(f)[-1]=='.gguf']
        if not files: raise Exception(f"No .gguf files in {directory}")
    else:
        file = os.path.join(a.model_dir, a.file) if a.model_dir else a.file
        if not os.path.exists(file): raise Exception(f"{file} not found")
        if not os.path.splitext(file)[-1]=='.gguf': raise Exception(f"{file} not a .gguf file")
        files = [file,]

    for file in files:
        upload(file)

if __name__=='__main__': main()