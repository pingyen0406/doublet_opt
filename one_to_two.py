# Optimization of 1D SLM to 2D pixels
"""********** Use tensor instead of ndarray in the script **********"""
from copyreg import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from band_limit_ASM import band_limit_ASM
from tools import *
import time
import yaml

"""# Get cpu or gpu device for training."""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


# Model class
class Model(nn.Module):
    def __init__(self,input_dim,distance,mesh,lambda0,N_slm):
        super().__init__()
        self.mesh = mesh
        self.d1 = distance['d1']
        self.d2 = distance['d2']
        self.t = distance['t']
        self.lambda0 = lambda0
        self.N_slm = N_slm
        # initialize the phase mask
        self.phi0 = torch.rand((int(input_dim),int(input_dim)))
        self.phi1 = nn.Parameter(torch.rand((int(input_dim),int(input_dim))))
        self.phi2 = nn.Parameter(torch.rand((int(input_dim),int(input_dim))))
    # Forward propagation     
    def forward(self,amp):
        E_before_mask1,_,_ = band_limit_ASM(amp,self.d1,self.mesh,1,self.lambda0,cut=True,device=device)
        E_inter, _,_ = band_limit_ASM(E_before_mask1*torch.exp(1j*self.phi1*2*torch.pi),self.t,self.mesh,1,self.lambda0/1.44,cut=True,device=device)
        E_imag,_,_ = band_limit_ASM(E_inter*torch.exp(1j*self.phi2*2*torch.pi),self.d2,self.mesh,1,self.lambda0,device=device)
        I_image = abs(E_imag)**2
        I_image = I_image/torch.max(I_image)
        return I_image
    
    # Loss calculation
    def cal_loss(self,pred,target):
        return -torch.log(torch.sum(pred*target))


"""# Training function    """
def train(model,config,initAmp_index,target_I_index,device):
    #print("Matrix size = ", str(target.shape))
    n_loop = config['n_loops'] # Number of the optimization loops
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    loss_record = {'N':[], 'Loss':[]}
    min_mse = 10000
    iteration = 0
    early_stop_cnt = 0    
    while iteration < n_loop:
        for i in range(model.N_slm):
            # Generate initial amplitude with given position and size
            initAmp = rect(model.phi0.shape,initAmp_index[i,0:2],initAmp_index[i,2:4])
            initAmp = initAmp.to(device)
            
            # Put initial ampitude into forward propagation and obtain image plane result
            pred = model(initAmp)
            
            # Generate target intensity with given position and size
            target_I = rect(pred.shape,target_I_index[i,0:2],target_I_index[i,2:4])
            target_I = target_I.to(device)

            # Calaulation loss
            
            if i==0:
                loss = model.cal_loss(pred,target_I)
            else:
                # Note: use 'float' instead of 'tensor', or memory will blow up
                loss = loss + float(model.cal_loss(pred,target_I))
        # Check if the loss is improved
        if loss < min_mse:
            min_mse = loss
            torch.save(model.state_dict(),'best_model.pth') # Save best model if it improves.
            early_stop_cnt=0 # reset early stop count
        else: 
            early_stop_cnt+=1
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('Current loop number:',iteration,' Loss= ',loss.detach().cpu().item())
        loss_record['N'].append(iteration)
        loss_record['Loss'].append(loss.detach().cpu().item())
        iteration+=1
        # Stop optimizing if loss is not improved
        if early_stop_cnt>config['early_stop_n']:
            print('Early stop triggered!!')
            break
    return model.phi1, model.phi2, loss_record


# Input config class
class cfg_class:
    sectionName='one_to_two_config'
    options={'N_slm': (int,True),
             'slm_pitch': (int,True),
             'N_atom': (int,True),
             'period': (int or float,True),
             'distance': (dict,True),
             'training': (dict,True)
             }

    def __init__(self,inFileName):
        inf = open(inFileName,'r')
        yamlcfg = yaml.safe_load(inf)
        inf.close()
        cfg = yamlcfg.get(self.sectionName)
        if cfg is None: raise Exception('Missing one_to_two_config section in yaml file')
        #iterate over options
        for opt in self.options:
            if opt in cfg:
                optval=cfg[opt]
 
                #verify parameter type
                if type(optval) != self.options[opt][0]:
                    raise Exception('Parameter "{}" has wrong type'.format(opt))
                 
                #create attributes on the fly
                setattr(self,opt,optval)
            else:
                if self.options[opt][1]:
                    raise Exception('Missing mandatory parameter "{}"'.format(opt))
                else:
                    setattr(self,opt,None)
            def __str__(self):
                return str(yaml.dump(self.__dict__,default_flow_style=False))

# Main function
def main():
    input_cfg = 'config.yaml'
    cfg = cfg_class(input_cfg)
    """# ** global parameters **"""
    lambda0 = 1.55
    k0 = 2*np.pi/lambda0
    period = cfg.period
    N_atom =  cfg.N_atom
    mesh = cfg.period
    N_slm = cfg.N_slm
    slm_pitch = cfg.slm_pitch # pixels
    distance = cfg.distance
    """# ** Generate indices of the input amplitude **"""
    initAmp_index = np.empty((N_slm,4))
    c_index = int(N_atom/2)
    for i in range(N_slm):
        initAmp_index[i] = np.array([c_index,c_index+slm_pitch*(i-int(N_slm/2)),2,2])
    initAmp_index = initAmp_index.astype(int)

    """# ** Generate indices of the target inensity**"""
    target_I_index = np.empty((N_slm,4))
    count=0
    for i in range(int(np.sqrt(N_slm))):
        for j in range(int(np.sqrt(N_slm))):
            target_I_index[count] = np.array([401+50*j,401+50*i,50,50],dtype=int)
            count+=1
    target_I_index = target_I_index.astype(int)
    """# Model """
    focusOpt = Model(N_atom,distance,mesh,lambda0,N_slm).to(device)
    focusOpt = focusOpt.to(device)
    initPhase = focusOpt.phi0

    """# ********************Start Training!****************************"""
    training_cfg = cfg.training
    start = time.time()
    phase1, phase2, record = train(focusOpt,training_cfg,initAmp_index,target_I_index,device)
    np.savetxt('results/optimized_mask1_xx.txt',phase1.cpu().detach().numpy()*2*np.pi)
    np.savetxt('results/optimized_mask2_xx.txt',phase2.cpu().detach().numpy()*2*np.pi)
    loss_record = np.array([record['N'],record['Loss']])
    np.savetxt('results/loss_record_xx.txt',loss_record)
    end = time.time()
    print('Elapsed time in training: ',end-start,'s')
    """# ********************End Training!****************************"""

    config = {**config,'totel elapsed time':end-start}
    with open('one_to_two.log','w') as log:
        pickle.dump(config,log)

    plt.figure()
    plt.plot(record['N'],record['Loss'])
    plt.title('Loss')
    plt.show()
    # 
if __name__ == '__main__':
    main()