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

TfN1 = 136
TfA = 240
TfN2 = 96
aug_frog_const = 2.0

TaN=160
TaB=120
playlist = {}
playlist.update( {'savings': 2 * [(P0, TsN), (Pplus, TsA), (Pminus, TsB), (Pchannel, TsC)],
'AB0':[(P0, TaN),(Pminus, TaB)],
'AB1':[(P0, TaN),(Pplus,13),(Pminus, TaB)],
'AB2':[(P0, TaN),(Pplus,41),(Pminus, TaB)],
'AB3':[(P0, TaN),(Pplus,112),(Pminus, TaB)],
'AB4':[(P0, TaN),(Pplus,230),(Pminus, TaB)],
'AB5':[(P0, TaN),(Pplus,410),(Pminus, TaB)],
'spontaneous': [(P0, 50),
                        (Pplus, 125),
                        (Pminus, 15),
                        (Pchannel, 150)],
'evoked': [(P0, 50),
            (Pplus, 125),
            (Pminus, 15),
            (Pchannel, 2),
            (Pplus, 2),
            (Pchannel, 146)],
'visw1p2': [(P0, 250),(Pplus,300), (Pchannel, 600)],            
'visw1p1': [(Pplus,300), (Pchannel, 600)],            
'visw1p3': [(0.5*Pminus, 250),(Pplus,300), (Pchannel, 600)], 
'frogs': [(P0,TfN1),((P0,Pplus),TfA),(P0,TfN2)],
'anti_frogs': [(P0,TfN1),(Pplus,TfA),(P0,TfN2)],

'frogs_aug': [(P0,TfN1),((P0,Pplus*aug_frog_const),TfA),(P0,TfN2)],
'anti_frogs_aug': [(P0,TfN1),(Pplus*aug_frog_const,TfA),(P0,TfN2)],

'frogs_long': [(P0,TfN1),((P0,Pplus),2*TfA),(P0,TfN2)],
'anti_frogs_long': [(P0,TfN1),(Pplus,2*TfA),(P0,TfN2)],

'overlearning_baseline': [(P0, 100), (Pplus, 200), (Pminus, 15), (Pchannel, 150)],
'overlearning': [(P0, 100), (Pplus, 600), (Pminus, 15), (Pchannel, 150)],

'pretrained_sr_baseline': [(P0, 192), (Pplus, 384), (Pminus, 20), (Pchannel, 364)],
'pretrained_sr': [(P0, 192), (Pminus, 384), (Pplus, 384),  (Pminus, 20), (Pchannel, 364)],

'wm_sr_baseline': [(P0, 192), (Pplus, 384), (Pminus, 20), (Pchannel, 192)],
'wm_sr_manipulation': [(P0, 192), (Pplus, 384), (Pminus, 19), (P0, 1), (Pchannel, 192)],
'wmE2_sr_baseline': [(P0, 192), (Pminus, 20), (Pchannel, 150)],
'wmE3_sr_baseline': [(P0, 192), (Pplus, 384), (Pchannel, 192)],

'pretrained_sr_baseline0p5': [(P0, 192), (Pplus, 384//2), (Pminus, 20), (Pchannel, 364)],
'pretrained_sr0p5': [(P0, 192), (Pminus, 384//2), (Pplus, 384//2),  (Pminus, 20), (Pchannel, 364)],
})


def albert_block(enable_noise=True, Pplus=14, Pnoise=6, Pnormalisation=15):
    '''Generates an Albert et al 2020, experiment 2'''
    tnull = 10*4
    texp = 80*4
    tchannel = 10*4
    
    if not enable_noise:
        Pexp = Pplus
    else:
        Pexp = Pplus + Pnoise*np.random.normal(size=texp)

    return [(P0, tnull), (Pexp/Pnormalisation, texp), (Pchannel, tchannel)]



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

'''add 1 noiseless and 10 noisy albert blocks'''
for i in range(101):
    enable_noise = (i != 0)
    playlist.update( {f'albert_block_{i}': albert_block(enable_noise=enable_noise)} )

hrz_params = {'z_list': [0.1,0.5,0.9], 'n_blocks': 25}
for iter in range(5):
    hrz_playlist =  generate_herzfeld_scenarios(**hrz_params, suffix=f'${iter}', probe_first=True)
    playlist.update(hrz_playlist)


parsed_playlist = {k: parse_samples(v) for k, v in playlist.items()}
print(f'generated {len(parsed_playlist)} paradigms: {list(parsed_playlist.keys())}')
with open('signoffrepertoire4.3.pkl', 'wb') as f:
    pickle.dump(parsed_playlist, f)