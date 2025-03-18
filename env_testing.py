'''
verifies pytorch installation and GPU use
'''
import torch
# basic test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
