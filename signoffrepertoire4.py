'''a script that generates a repertoire of sign-off paradigms
they are parsed into saved into a pickle file'''

import numpy as np
from siggen_utils import herzfeld_block
# import json
import pickle
from dsp_utils import parse_samples

Pplus=1
Pminus=-1
P0=0
Pchannel=np.nan

TsN=60
TsA=120
TsB=20
TsC=20


TaN=150
TaB=120
playlist = {}
playlist.update( {'savings': 2 * [(P0, TsN), (Pplus, TsA), (Pminus, TsB), (Pchannel, TsC)],
'AB0':[(P0, TaN),(Pminus, TaB)],
'AB1':[(P0, TaN),(Pplus,13),(Pminus, TaB)],
'AB2':[(P0, TaN),(Pplus,41),(Pminus, TaB)],
'AB3':[(P0, TaN),(Pplus,112),(Pminus, TaB)],
'AB4':[(P0, TaN),(Pplus,230),(Pminus, TaB)],
'AB5':[(P0, TaN),(Pplus,410),(Pminus, TaB)]})


def generate_herzfeld_scenarios(z_list=None, n_blocks=None, Tflips=None, suffix='', probe_first=False): 
    out_dict = {}
    for z in z_list:
        scenario_name =  f'herzfeld,z={z}{suffix}'
        pert_per_z = []
        for n in range(n_blocks):
            hz = herzfeld_block(z, P1=Pplus,P2=Pminus,P0=P0, tau=1, probe_first=probe_first)
            pert_per_z.append((hz,len(hz)))
        out_dict.update({scenario_name:pert_per_z})
    return out_dict

hrz_params = {'z_list': [0.1,0.5,0.9], 'n_blocks': 25}
for iter in range(20):
    hrz_playlist =  generate_herzfeld_scenarios(**hrz_params, suffix=f'${iter}', probe_first=True)
    playlist.update(hrz_playlist)

parsed_playlist = {k: parse_samples(v) for k, v in playlist.items()}
print(f'generated {len(parsed_playlist)} paradigms: {list(parsed_playlist.keys())}')
with open('signoffrepertoire4.pkl', 'wb') as f:
    pickle.dump(parsed_playlist, f)