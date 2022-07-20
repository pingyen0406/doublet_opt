# Simple optimizer to obtain a focusing wavefront
"""********** Use tensor instead of ndarray in the script **********"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from band_limit_ASM import band_limit_ASM
from tools import *
import time

"""# Get cpu or gpu device for training."""
device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
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
        for i in range(self.N_slm):
            E_before_mask1,_,_ = band_limit_ASM(amp[i],2000,self.mesh,1,self.lambda0,cut=True,device=device)
            E_inter, _,_ = band_limit_ASM(E_before_mask1*torch.exp(1j*self.phi1*2*torch.pi),1000,self.mesh,1,self.lambda0/1.44,cut=True,device=device)
            E_imag,_,_ = band_limit_ASM(E_inter*torch.exp(1j*self.phi2*2*torch.pi),self.propZ,self.mesh,1,self.lambda0,device=device)
            I_image = abs(E_imag)**2
            I_image = I_image/torch.max(I_image)
            if i==0:
                I_image_list = torch.empty((self.N_slm,I_image.shape[0],I_image.shape[1]))
                I_image_list[0] = I_image
            else:
                I_image_list[i] = I_image
        return I_image_list
    
    # Loss calculation
    def cal_loss(self,pred,target):
        loss = 0
        for i in range(self.N_slm):
            tmp_loss = torch.log(torch.sum(pred[i]*target[i]))
            loss = loss + tmp_loss
        return -loss
        #loss = nn.CrossEntropyLoss()
        #target = target.double()
        #return loss(pred,target)

"""# Training function    """
def train(model,config,initAmp,target,device):
    #print("Matrix size = ", str(target.shape))
    n_loop = config['n_loops'] # Number of the optimization loops
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    loss_record = {'N':[], 'Loss':[]}
    #initAmp.to(device)
    #target = target.to(device)
    min_mse = 10000
    iteration = 0
    early_stop_cnt = 0    
    while iteration < n_loop:
        optimizer.zero_grad()
        pred = model(initAmp)
        #pred = pred.to(device)
        loss = model.cal_loss(pred,target)
        # Check if the loss is improved
        if loss < min_mse:
            min_mse = loss
            torch.save(model.state_dict(),'best_model.pth') # Save best model if it improves.
            early_stop_cnt=0 # reset early stop count
        else: 
            early_stop_cnt+=1
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
    n_mesh = int(N_atom*period/mesh)
    x_lens = torch.tensor([i*mesh for i in range(n_mesh)])
    y_lens = torch.tensor([i*mesh for i in range(n_mesh)])
    x_lens-= torch.median(x_lens)
    y_lens-= torch.median(y_lens)

    #init_phase = np.random.uniform(0,2*np.pi,inputA.shape)
    xw = torch.tensor([(i)*mesh for i in range(3*n_mesh)])
    yw = torch.tensor([(i)*mesh for i in range(3*n_mesh)])
    xw -= torch.median(xw)
    yw -= torch.median(yw)

    """# ** Generate initial amp **"""
    initAmp = line_SLM_source(N_slm,x_lens,y_lens,2,slm_pitch,lambda0)
    """# ** Define target E-field**"""
    target_E = torch.empty(N_slm,len(xw),len(yw))
    count=0
    for i in range(7):
        for j in range(7):
            target_E[count] = rect(xw,yw,[401+50*j,401+50*i],[50,50])
            count+=1
    target_I = target_E
    """# Model """
    focusOpt = Model(N_atom*period/mesh,f,mesh,lambda0,N_slm).to(device)
    initPhase = focusOpt.phi0
    initPhase = initPhase.to(device)
    initAmp = initAmp.to(device)
    """# ********************Start Training!****************************"""
    start = time.time()
    phase1, phase2, record = train(focusOpt,config,initAmp,target_I,device)
    np.savetxt('optimized_mask1_49.txt',phase1.cpu().detach().numpy()*2*np.pi)
    np.savetxt('optimized_mask2_49.txt',phase2.cpu().detach().numpy()*2*np.pi)
    loss_record = np.array([record['N'],record['Loss']])
    np.savetxt('loss_record.txt',loss_record)
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