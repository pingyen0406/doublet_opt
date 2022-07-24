# Optimization of 1D SLM to 2D pixels
"""********** Use tensor instead of ndarray in the script **********"""
from hypothesis import target
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from band_limit_ASM import band_limit_ASM
from tools import *
import time

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
    def __init__(self,input_dim,propZ,mesh,lambda0,N_slm):
        super().__init__()
        self.mesh = mesh
        self.propZ = propZ
        self.lambda0 = lambda0
        self.N_slm = N_slm
        # initialize the phase mask
        self.phi0 = torch.rand((int(input_dim),int(input_dim)))
        self.phi1 = nn.Parameter(torch.rand((int(input_dim),int(input_dim))))
        self.phi2 = nn.Parameter(torch.rand((int(input_dim),int(input_dim))))
    # Forward propagation     
    def forward(self,amp):
        E_before_mask1,_,_ = band_limit_ASM(amp,2000,self.mesh,1,self.lambda0,cut=True,device=device)
        E_inter, _,_ = band_limit_ASM(E_before_mask1*torch.exp(1j*self.phi1*2*torch.pi),1000,self.mesh,1,self.lambda0/1.44,cut=True,device=device)
        E_imag,_,_ = band_limit_ASM(E_inter*torch.exp(1j*self.phi2*2*torch.pi),self.propZ,self.mesh,1,self.lambda0,device=device)
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
                loss += float(model.cal_loss(pred,target_I))
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


"""# **Setup Hyper-parameters** """
config={
    'optimizer': 'Adam',
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.01,                  # learning rate 
        #'betas': (0.1,0.5),          
        #'momentum': 0.5     w         # momentum for SGD
    },
    'n_loops': 100,
    'early_stop_n': 20
}

def main():
    """# ** global parameters **"""
    lambda0 = 1.55
    k0 = 2*np.pi/lambda0
    period = 5
    N_atom =  401
    mesh = 5
    f = 20000
    N_slm = 49
    slm_pitch = 5 # pixels

    """# ** Generate indices of the input amplitude **"""
    initAmp_index = np.empty((N_slm,4))
    c_index = int(N_atom/2)
    for i in range(N_slm):
        initAmp_index[i] = np.array([c_index,c_index+slm_pitch*(i-int(N_slm/2)),2,2])
    initAmp_index = initAmp_index.astype(int)

    """# ** Generate indices of the target inensity**"""
    target_I_index = np.empty((N_slm,4))
    count=0
    for i in range(7):
        for j in range(7):
            target_I_index[count] = np.array([401+50*j,401+50*i,50,50],dtype=int)
            count+=1
    target_I_index = target_I_index.astype(int)
    """# Model """
    focusOpt = Model(N_atom*period/mesh,f,mesh,lambda0,N_slm).to(device)
    focusOpt = focusOpt.to(device)
    initPhase = focusOpt.phi0

    """# ********************Start Training!****************************"""
    start = time.time()
    phase1, phase2, record = train(focusOpt,config,initAmp_index,target_I_index,device)
    np.savetxt('results/optimized_mask1_x.txt',phase1.cpu().detach().numpy()*2*np.pi)
    np.savetxt('results/optimized_mask2_x.txt',phase2.cpu().detach().numpy()*2*np.pi)
    loss_record = np.array([record['N'],record['Loss']])
    np.savetxt('results/loss_record_x.txt',loss_record)
    end = time.time()
    print('Elapsed time in training: ',end-start,'s')
    """# ********************End Training!****************************"""

    plt.figure()
    plt.plot(record['N'],record['Loss'])
    plt.title('Loss')
    plt.show()
    # 
if __name__ == '__main__':
    main()