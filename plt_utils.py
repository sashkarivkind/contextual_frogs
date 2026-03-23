import matplotlib.pyplot as plt
import numpy as np

def plot_by_key(results, keys=None, colors = ['tab:blue', 'tab:green'], x0=0,
                visu_offsets = [0,0.1],  #y_offsets used for visualisation
                align_end = False,
                parse=lambda foo:foo.u_lp,
                plot_opts= {'linewidth':2}
                ):

  for i,key in enumerate(keys):
    y_data = np.array(parse(results[key]))
    x_data = np.arange(len(y_data)) + x0
    x_data -= x_data.max() if align_end else 0
    plt.plot(x_data, y_data+visu_offsets[i],colors[i],**plot_opts)

    
def plot_segments(result , colors = ['tab:red', 'tab:orange', 'tab:pink'], 
                t_start=None, 
                t_increment=None,
                t_segment=None,
                n_segments=None,
                x0=0,
                visu_offsets = [0,0.0,0.0],  #y_offsets used for visualisation
                plot_opts= {'linewidth':2}
                ):

  t_segment = t_segment if t_segment is not None else t_increment
  for i in range(n_segments):
    y_data = result[t_start+i*t_increment:t_start+i*t_increment+t_segment]
    y_data = np.array(y_data)
    y_data = np.array(y_data)
    x_data = np.arange(len(y_data)) + x0
    plt.plot(x_data, y_data+visu_offsets[i],colors[i],**plot_opts)

def parse_herzfeld_data(ooo, n_blocks=25, pooling_fun=np.mean):
  '''a function tailored for working with 'part2' edition of the model/results/etc.'''
  deltas_by_super_scenario = {}
  for i, this_data in enumerate([ooo]):

      for iz, z in enumerate(ooo.keys()):
          if 'herzfeld' not in z:
              continue
          else:
              scenario = z
              super_scenario = z.split('$')[0]
          deltas = []
          this_result = pooling_fun(this_data[scenario],axis=1)
          pointer = 0
          for bb in range(n_blocks):
              block_length = len(this_result)//n_blocks
              deltas.append(this_result[pointer+2]-this_result[pointer])
              pointer += block_length

          if super_scenario not in deltas_by_super_scenario:
              deltas_by_super_scenario[super_scenario] = []
          deltas_by_super_scenario[super_scenario].append(deltas)
  return deltas_by_super_scenario