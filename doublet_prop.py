"""Propagation through doublet metasurface"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from band_limit_ASM import band_limit_ASM
from tools import *
import time

"""# Get cpu or gpu device for training."""
device = 'cpu'
"""# ** initial parameters **"""
lambda0 = 1.55
k0 = 2*np.pi/lambda0
period = 5
N_atom =  401
mesh = 5
f = 20000
N_slm= 49
slm_pitch = 5 # pixels
n_mesh = int(N_atom*period/mesh)
x_lens = torch.tensor([i*mesh for i in range(n_mesh)])
y_lens = torch.tensor([i*mesh for i in range(n_mesh)])
x_lens-= torch.median(x_lens)
y_lens-= torch.median(y_lens)
xw = torch.tensor([(i)*mesh for i in range(3*n_mesh)])
yw = torch.tensor([(i)*mesh for i in range(3*n_mesh)])
xw -= torch.median(xw)
yw -= torch.median(yw)

phase1 = np.loadtxt('results/optimized_mask1_49.txt')
phase2 = np.loadtxt('results/optimized_mask2_49.txt')

phase1 = torch.tensor(phase1)
phase2 = torch.tensor(phase2)

phase1 -= torch.min(phase1)
phase2 -= torch.min(phase2)

# phase1 = norPhase(phase1)
# phase2 = norPhase(phase2)

plot2Field(phase1,phase2,x_lens,y_lens,'Optimized phase mask 1','Optimized phase mask 2')
"""# ** Generate initial amp **"""

initAmp = line_SLM_source(N_slm,x_lens,y_lens,2,slm_pitch,lambda0)

fake1 = torch.rand((401,401))
fake2 = torch.ones((401,401))
fake_amp = initAmp[0]+initAmp[12]+initAmp[24]+initAmp[36]+initAmp[48]

plotField(fake_amp,x_lens,y_lens,'imageplane')


E_before_mask1,_,_ = band_limit_ASM(fake_amp,2000,mesh,1,1.55,device='cpu',cut=True)
inter_E,_,_ = band_limit_ASM(E_before_mask1*torch.exp(1j*fake2),1000,mesh,1,1.55/1.44,device='cpu',cut=True) 

final_E, xi, yi = band_limit_ASM(inter_E*torch.exp(1j*fake2),f,mesh,1,1.55,device='cpu') 
final_I = abs(final_E)**2
final_I /= torch.max(final_I)

plotField(final_I,xw,yw,'imageplane')

plt.show()


