import yaml, os, re

class Configurations:
    _instance = None
    @classmethod
    def instance(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance

    def __init__(self):
        with open('configurations.yaml', 'r') as f:
            loaded = yaml.safe_load(f)
            self.configurations = loaded['configurations']
            self.metadata = loaded['metadata']
            self.natives = [q.strip() for q in self.metadata['native'].split(",")] + ["BF16",]
        self.base_dir:str     = None
        self.base_patcher:str = None

    def __iter__(self):
        for key in self.configurations: yield key

    def modelpath(self, model): 
        return os.path.join(self.base_dir, model) if self.base_dir else model
    
    def patchpath(self, cast):
        if self.base_patcher is None: 
            raise Exception(f"--patch is required for any casts not in {self.natives}")
        this_patcher = re.sub("Q[0-9_KLMS]*", cast, self.base_patcher)
        return self.modelpath(this_patcher)

    def configuration(self, key):
        return self.configurations[key]
    
    def as_string_with_notes(self, k):
        string = f"{k:>6} {self.configurations[k]['notes']}"
        patches_needed = [q['castto'] for q in self.configurations[k]['casts'] if q['castto'] not in self.natives]
        if patches_needed: string += f" (requires patch file(s) for {', '.join(patches_needed)})"
        return string

    @property
    def all_as_string_with_notes(self):
        return "\n".join(self.as_string_with_notes(k) for k in self.configurations)
    
    @property
    def as_list(self):
        return [k for k in self.configurations]
    
    def available_natively(self, q): return q in self.natives

configurations = Configurations.instance()