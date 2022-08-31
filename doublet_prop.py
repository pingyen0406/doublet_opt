"""Propagation through doublet metasurface"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from band_limit_ASM import band_limit_ASM
from tools import *
import time
from one_to_two import Model
from PIL import Image

"""Plot loss"""
loss_file = np.loadtxt('results/loss_record_220801.txt')
number = loss_file[0]
loss = loss_file[1]
#plt.figure()
#plt.plot(number,loss,linewidth=2)

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

phase1 = np.loadtxt('results/optimized_mask1_220801.txt')
phase2 = np.loadtxt('results/optimized_mask2_220801.txt')

phase1 = torch.tensor(phase1)
phase2 = torch.tensor(phase2)

phase1 -= torch.min(phase1)
phase2 -= torch.min(phase2)

#phase1 = norPhase(phase1)
#phase2 = norPhase(phase2)

#plt.figure()
#plt.imshow(phase2,cmap='twilight')

#plot2Field(phase1,phase2,x_lens,y_lens,'Optimized phase mask 1','Optimized phase mask 2')

"""# ** Generate indices of the input amplitude **"""
initAmp_index = np.empty((N_slm,4))
c_index = int(N_atom/2)
for i in range(N_slm):
    initAmp_index[i] = np.array([c_index,c_index+slm_pitch*(i-int(N_slm/2)),2,2])
initAmp_index = initAmp_index.astype(int)

initAmp = torch.empty((N_slm,len(phase1),len(phase1)))
for i in range(N_slm):
    # Generate initial amplitude with given position and size
    initAmp[i] = rect(phase1.shape,initAmp_index[i,0:2],initAmp_index[i,2:4])
    
"""# ** Generate indices of the target image **"""
target_I_index = np.empty((N_slm,4))
count=0
for i in range(7):
    for j in range(7):
        target_I_index[count] = np.array([451+50*j,451+50*i,50,50],dtype=int)
        count+=1
target_I_index = target_I_index.astype(int)
target_I = torch.empty((N_slm,1203,1203))

effi = torch.empty((49,4))

for i in range(49):
    incident_pixel=[i]
    
    amp0 = initAmp[i]
    int0 = abs(amp0)**2

    E_before_mask1,_,_ = band_limit_ASM(amp0,2000,mesh,1,1.55,device='cpu',cut=True)
    i0 = torch.sum(abs(E_before_mask1)**2)
    effi[i,0] = i0/torch.sum(int0)
    
    inter_E,_,_ = band_limit_ASM(E_before_mask1*torch.exp(1j*phase1),1000,mesh,1,1.55/1.44,device='cpu',cut=True) 
    i1 = torch.sum(abs(inter_E)**2)
    effi[i,1] = i1/torch.sum(int0)
    
    final_E, xi, yi = band_limit_ASM(inter_E*torch.exp(1j*phase2),f,mesh,1,1.55,device='cpu') 
    final_I = abs(final_E)**2
    i2 = torch.sum(final_I)
    effi[i,2] = i2/torch.sum(int0)
    target_I_now = rect((1203,1203),target_I_index[i,0:2],target_I_index[i,2:4])
    
    effi[i,3] = torch.sum(final_I*target_I_now)/torch.sum(int0)

    final_I /= torch.max(final_I)

    #plotField(final_I,xw,yw,'imageplane')
    #plt.savefig(str(i)+'.png')

plt.figure()
plt.plot(effi[:,0],linewidth=2,label='i0')
plt.plot(effi[:,1],linewidth=2,label='i1')
plt.plot(effi[:,2],linewidth=2,label='i2')
plt.ylim([0.9,1])
plt.legend()
plt.figure()
plt.plot(effi[:,3],linewidth=2,label='final')
plt.legend()
plt.show()


# Save gif file
# imag_list=[]
# for i in range(49):
#     tmpimag = Image.open(str(i)+'.png')
#     imag_list.append(tmpimag)
# imag_list[0].save('scan.gif',save_all=True,append_images=imag_list,duration=200)
#plt.show()


