import os, math, yaml, json
from dataclasses import dataclass
from argparse import ArgumentParser

casts_and_bits = {
    '16bit':16,
    'Q8_0':8.5,
    'bf8':8,
    'Q6_K':6.5,
    'Q5_1':6,
    'Q5_K_S':5.5, 
    'Q4_1':5, 
    'Q4_1':5, 
    'Q4_K_S':4.5, 
    'Q4_0':4.5, 
    'Q3_K_S':3.4375,
    'Q2_K':2.625
}

available_casts = [ '16bit','Q8_0','Q5_1','Q4_1', ]

@dataclass
class CastAndCost:
    cast:str
    cost:float

    @property
    def bits(self): return casts_and_bits[self.cast]

@dataclass
class CastingStep:
    from_cast:str
    to_cast:str
    bits_saved:float
    error_increase:float
    layer:int = None

    @property
    def bits_per_error(self):
        return self.bits_saved / (self.error_increase + 1e-9)
    
    @classmethod
    def from_two_cast_costs(cls, cc1:CastAndCost, cc2:CastAndCost):
        return CastingStep( cc1.cast, cc2.cast, cc1.bits-cc2.bits, cc2.cost-cc1.cost )
    
    @classmethod
    def merge(cls, s1, s2):
        assert s1.to_cast == s2.from_cast and s1.layer == s2.layer
        return CastingStep( s1.from_cast, s2.to_cast, s1.bits_saved+s2.bits_saved, s1.error_increase+s2.error_increase, s1.layer )
    
def get_costs(use_only=None) -> dict[str,dict[str,float]]:
    casting_cost_yaml_file = os.path.join(os.path.dirname(__file__), "costs", "casting_cost.yaml")
    with open(casting_cost_yaml_file, 'r') as f:
        costs = yaml.safe_load(f)
    if use_only:
        for layer in costs:
            costs[layer] = { cast:costs[layer][cast] for cast in available_casts if cast in costs[layer] and cast in use_only }
    for layer in costs: costs[layer]['16bit'] = 0
    return costs

def list_good_steps_for_layer(costs_for_layer:dict[str,float], layer:int) -> list[CastingStep]:
    cast_cost_list = [ CastAndCost(cast, costs_for_layer[cast]) for cast in available_casts if cast in costs_for_layer ]
    initial_discards = []
    for i in range(len(cast_cost_list)-1):
        # if a later step is cheaper, this one is useless
        cc = cast_cost_list[i]
        next_cc = cast_cost_list[i+1]
        if next_cc.cost < cc.cost: initial_discards.append(cc.cast)  
        elif i>0:
            # if the previous step had the same number of bits and was cheaper, this one is useless
            last_cc = cast_cost_list[i-1]
            if casts_and_bits[cc.cast] == casts_and_bits[last_cc.cast]:
                if last_cc.cost <= cc.cost: initial_discards.append(cc.cast)
    
    cast_cost_list = [ cc for cc in cast_cost_list if cc.cast not in initial_discards ]
    steps = [ CastingStep.from_two_cast_costs(cast_cost_list[i], cast_cost_list[i+1]) for i in range(len(cast_cost_list)-1) ]

    while True:
        still_valid_steps:list[CastingStep] = []
        skip_next = False
        for i in range(len(steps)-1):
            if skip_next:
                skip_next = False
                if (i==len(steps)-2):
                    still_valid_steps.append(steps[i+1])
            else:
                if steps[i].bits_per_error > steps[i+1].bits_per_error: 
                    still_valid_steps.append(steps[i])
                    if (i==len(steps)-2):
                        still_valid_steps.append(steps[i+1])
                else: 
                    still_valid_steps.append(CastingStep.merge(steps[i], steps[i+1]))
                    skip_next =True
        if len(still_valid_steps) == len(steps): 
            break
        steps = [ s for s in still_valid_steps]

    for s in still_valid_steps: s.layer = layer
    return still_valid_steps

def get_sorted_steps() -> list[CastingStep]:
    all_costs = get_costs()
    all_steps:list[CastingStep] = []
    for layer in all_costs:
        all_steps.extend(list_good_steps_for_layer(all_costs[layer], layer))
    for step in all_steps: step.bits_saved = step.bits_saved * (339738624 if step.layer<19 else 141557760) 
    all_steps.sort( key=lambda a:a.bits_per_error, reverse=True )
    return all_steps

def select_steps(all_steps:list[CastingStep], bits_wanted) -> list[CastingStep]:
    bits = 0
    cost = 0
    selected = []
    for step in all_steps:
        selected.append(step)
        bits += step.bits_saved
        cost += step.error_increase
        if bits >= bits_wanted: break
    return selected, bits, cost

def merge(applied_steps:list[CastingStep]) -> list[CastingStep]:
    merged = [None]*57
    for step in applied_steps:
        if merged[step.layer] is None:
            merged[step.layer] = step
        else:
            merged[step.layer] = CastingStep.merge(merged[step.layer], step)
    return merged

total_bits = 189347659776
bits_per_gigabyte = 8*math.pow(2,30)
def bits_to_gbytes(bits)->float: return bits/bits_per_gigabyte
def gybtes_to_bits(gb): return gb*bits_per_gigabyte

def get_optimised_casting(gb_wanted) -> list[CastingStep]:
    selected, bits_saved, cost = select_steps(get_sorted_steps(), bits_wanted=gybtes_to_bits(gb_wanted))  
    merged = merge(selected)
    #for selection in merged:
    #    if selection:
    #        print(f"Cast layer {selection.layer} to {selection.to_cast}")
    print(f"Full model is {bits_to_gbytes(total_bits):>6.3f} GB")
    print(f"{bits_to_gbytes(bits_saved):>6.3f} GB saved at cost of {cost:<8.3f}")
    return merged, bits_to_gbytes(bits_saved)

def to_comma_list(values:list[int]):
    strings = []
    sequence_start = None
    previous       = None
    for value in values + [99999,]:
        if previous is None:
            sequence_start = value
        elif value == previous+1:
            pass
        else:
            if previous == sequence_start: strings.append(str(sequence_start))
            else: strings.append(f"{sequence_start}-{previous}")
            sequence_start = value
        previous       = value
    return ", ".join(strings)

def convert_to_config(casting:list[CastingStep]):
    cast_to_layer_map:dict[str,list[int]] = {}

    for layer, selection in enumerate(casting):
        cast = selection.to_cast if selection is not None else 'BF16'
        if cast not in cast_to_layer_map: cast_to_layer_map[cast] = []
        cast_to_layer_map[cast].append(layer)

    casting_list = list( {'layers':to_comma_list(cast_to_layer_map[cast]), 'castto':cast} for cast in cast_to_layer_map )
    return casting_list

if __name__=='__main__': 
    a = ArgumentParser()
    a.add_argument('--gb', type=float, help="Approximate number of GB to remove")
    args = a.parse_args()
    try: gb = args.gb
    except: 
        print("--gb must be specified")
        exit()
    casting, gbs = get_optimised_casting(args.gb)

    y = convert_to_config(casting)
    #tenths = math.floor( 10*gbs + 0.5 )
    #name = f"{tenths//10}_{tenths%10}"
    print(f"\n    \"CONFIG_NAME\" : " + "{")
    print("        'casts': [")
    for cast in y: print("            {'layers': '" + cast['layers'] + "', 'castto': '" + cast['castto'] + "'},")
    print("        ],")
    print("        'notes': 'replace this with a comment!'")
    print("    },")
    
