from one_to_two import *
from tools import * 


"""# ** load input config **"""
input_cfg = 'config.yaml'
cfg = cfg_class(input_cfg)
lambda0 = 1.55
k0 = 2*np.pi/lambda0
period = cfg.period
N_atom =  cfg.N_atom
mesh = cfg.period
N_slm = cfg.N_slm
slm_pitch = cfg.slm_pitch # pixels
distance = cfg.distance


"""Load best model"""
best_model = Model(N_atom,distance,mesh,lambda0,N_slm)
best_model = torch.load('best_model.pth')
print(type(best_model))


phase1 = best_model['phi1']
print(phase1.shape)