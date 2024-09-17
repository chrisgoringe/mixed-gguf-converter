import yaml, os, re, argparse

class PatchFileException(Exception): pass

class PathHandler:
    def __init__(self):
        self.base_dir:str     = None
        self.base_patcher:str = None
    
    def modelpath(self, model): return os.path.join(self.base_dir, model) if self.base_dir else model
    
    def patchpath(self, cast):
        if self.base_patcher is None: 
            raise PatchFileException(f"--patch is required for {cast}")
        this_patcher = re.sub("Q[0-9_KLMS]*", cast, self.base_patcher)
        path = self.modelpath(this_patcher)
        if not os.path.exists(path):
            raise PatchFileException(f"File {path}, required for {cast}, is missing (check --patch points to it, or to another patch file alongside it)")
        return path
    
    def list_patch_problems(self, casts:list[str]) -> list[str]:
        probs = []
        for c in casts:
            if not self.available_natively(c):
                try: self.patchpath(c)
                except PatchFileException as e: probs.append(e.args[0])
        return probs

class Configurations(PathHandler):
    _instance = None
    @classmethod
    def instance(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance

    def __init__(self):
        super().__init__()
        with open('configurations.yaml', 'r') as f:
            loaded = yaml.safe_load(f)
            self.configurations = loaded['configurations']
            self.metadata = loaded['metadata']
            self.natives = [q.strip() for q in self.metadata['native'].split(",")]

    def __iter__(self):        yield from self.configurations
    def __contains__(self, k): return k in self.configurations
    def __getitem__(self, k):  return self.configurations[k]
    def pop(self, k):          return self.configurations.pop(k)

    def sort(self):
        keys = [k for k in self.configurations]
        try:
            keys.sort(key = lambda a:float(a.replace('_','.')))
        except ValueError:
            print("Failed to sort configurations into order")
        self._configurations = { k:self.configurations[k] for k in keys }
        self.configurations = self._configurations

    def save(self):
        self.sort()
        tosave = { 'metadata' : self.metadata, 'configurations': self.configurations }
        with open('configurations.yaml', 'w') as f:
            yaml.safe_dump(tosave, f)

    def rename(self, on, nn):         self.configurations[nn] = self.pop(on)
    def remove(self, on):             self.pop(on)
    def edit_notes(self, key, notes): self[key]['notes'] = notes

    def add(self, key:str, casts:list[dict[str,str]], notes:str, allow_bad_key=False, **kwargs):
        if key in self:
            print(f"{key} already in configuration list - not adding")
            return
        try: float(key.replace('_','.'))
        except ValueError: 
            print(f"{key} not in the form X_X, XX_X, or similar")
            if allow_bad_key: print(f"Adding it, but you will want to use configuration.py --rename {key}:X_X to rename it")
            else: return
        self.configurations[key] = { "casts" : casts, "notes" : notes }
        for k in kwargs: self.configurations[key][k] = kwargs[k]
    
    def as_string_with_notes(self, k):
        string = f"{k:>6} {self[k]['notes']}"
        patches_needed = [q['castto'] for q in self.configurations[k]['casts'] if q['castto'] not in self.natives]
        if patches_needed: string += f" (requires patch file(s) for {', '.join(patches_needed)})"
        return string

    @property
    def all_as_string_with_notes(self): return "\n".join(self.as_string_with_notes(k) for k in self.configurations)
    
    @property
    def as_list(self): return [k for k in self.configurations]
    
    def available_natively(self, q): return q in self.natives

configurations = Configurations.instance()

def main():
    a = argparse.ArgumentParser(description="List or edit the configuration list")
    b = a.add_mutually_exclusive_group()
    b.add_argument('--rename', help="--rename FROM:TO renames the configuration 'FROM' to 'TO")
    b.add_argument('--remove', help="Remove a configuration")
    b.add_argument('--sort', action="store_true", help="Sort the configuration list")
    b.add_argument('--notes', help="--notes KEY:notes to chenge the notes on a configuration item" )
    
    a = a.parse_args()
    
    if a.rename:
        names = tuple(x.strip() for x in a.rename.split(':'))
        if len(names)==2:
            oldname, newname = names
            if oldname in configurations:
                if newname not in configurations:
                    configurations.rename(oldname, newname)
                else: print(f"{newname} already exists")
            else: print(f"{oldname} not a configuration")
        else: print(f"Usage: configuration.py --rename from:to")
        
    elif a.remove:
        if a.remove in configurations:
            configurations.remove(a.remove)
        else: print(f"{a.remove} not in configurations")

    elif a.sort:
        configurations.sort()

    elif a.notes:
        bits = tuple(x.strip() for x in a.rename.split(':'))
        if len(bits)==2:
            key, notes = bits
            if key in configurations:
                configurations.edit_notes(key, notes)
            else: print(f"{key} not a configuration")
        else: print(f"Usage: configuration.py --notes key:text")

    else:
        print(configurations.all_as_string_with_notes)


if __name__=='__main__': 
    main()
    configurations.save()