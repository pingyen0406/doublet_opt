# Simple optimizer to obtain a focusing wavefront
"""********** Use tensor instead of ndarray in the script **********"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from tools import *
from band_limit_ASM import band_limit_ASM

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
    def __init__(self,input_dim,propZ,mesh,lambda0):
        super().__init__()
        # initializing the phase mask with random phase
        self.phi0 = torch.ones((int(input_dim),int(input_dim)))
        self.phi = nn.Parameter(torch.ones((int(input_dim),int(input_dim))))
        self.mesh = mesh
        self.lambda0 = lambda0
        self.propZ = propZ
         
    def forward(self,amp):
        E_imag, _,_ = band_limit_ASM(amp*torch.exp(1j*self.phi*2*torch.pi),self.propZ,self.mesh,1,self.lambda0)
        I_image = abs(E_imag)**2
        I_image = I_image/torch.max(I_image)
        return I_image

    def cal_loss(self,pred,target):
        return -torch.sum(pred*target)
        loss = nn.CrossEntropyLoss()
        target = target.double()
        return loss(pred,target)

# Optimization    
def train(model,config,initAmp,target,device):
    print("Matrix size = ", str(target.shape))
    n_loop = config['n_loops'] # Number of the optimization loops
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    loss_record = {'N':[], 'Loss':[]}
    initAmp.to(device)
    target = target.to(device)
    min_mse = 10000
    iteration = 0
    early_stop_cnt = 0    
    while iteration < n_loop:
        optimizer.zero_grad()
        pred = model(initAmp)
        #pred = pred.to(device)
        loss = model.cal_loss(pred,target)
        #loss = F.cross_entropy(pred.double(),target.double())
        # Check if the loss is improved
        if loss < min_mse:
            min_mse = loss
            torch.save(model.state_dict(),'results/best_model.pth') # Save best model if it improves.
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
    return model.phi, loss_record

def main():
    """# **Setup Hyper-parameters** """
    config={
        'optimizer': 'Adam',
        'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
            'lr': 0.01,                  # learning rate 
            #'betas': (0.1,0.5),          
            #'momentum': 0.5              # momentum for SGD
        },
        'n_loops': 100,
        'early_stop_n': 100
    }

    """# ** initial parameters **"""
    lambda0=1.55
    k0 = 2*np.pi/lambda0
    period = 1
    N_atom =  1001
    mesh = 1
    f = 3000
    n_mesh = int(N_atom*period/mesh)

    # Create grids for the phase mask
    x_lens = torch.tensor([i*mesh for i in range(n_mesh)])
    y_lens = torch.tensor([i*mesh for i in range(n_mesh)])
    x_lens-= torch.median(x_lens)
    y_lens-= torch.median(y_lens)

    # Create grids for image plane
    xw = torch.tensor([i*mesh for i in range(3*n_mesh)])
    yw = torch.tensor([i*mesh for i in range(3*n_mesh)])
    xw -= torch.median(xw)
    yw -= torch.median(yw)


    """# ** Generate initial amp **"""
    #initAmp = torch.rand((len(x_lens),len(y_lens)))
    initAmp = gaussian(lambda0,300,0.001,x_lens,y_lens)
    # for i in range(9):
    #     initAmp += gaussian(1.55,2,200,x_lens,y_lens-25*(i-4))
    initAmp /= torch.max(initAmp)
    #initAmp = torch.ones((n_mesh,n_mesh))

    """# ** Define target E-field**"""
    from PIL import Image
    target_image = Image.open('figures/testZ.png')
    target_image = target_image.resize((len(xw),len(yw)))
    target_image = target_image.convert('L')
    target_I = np.array(target_image,dtype=float)
    target_I /= np.max(target_I)
    target_I = torch.tensor(target_I)

    # target_E = gaussian(lambda0,3,0.001,xw,yw)
    # target_E = rect(xw,yw,[1500,1800],[100,100])
    # target_E = 0
    # for i in range(3):
    #     for j in range(3):
    #        target_E += gaussian(lambda0,5,0.0001,xw+(i-1)*30,yw+(j-1)*30)
    # target_I = abs(target_E)**2
    # target_I /= torch.max(target_I)

    """#********************** Model ***********************************"""
    focusOpt = Model(N_atom*period/mesh,f,mesh,lambda0).to(device)
    initPhase = focusOpt.phi0
    initPhase = initPhase.to(device)
    initAmp = initAmp.to(device)
    """# ******************** Start Training!****************************"""
    start = time.time()
    phase, record = train(focusOpt,config,initAmp,target_I,device)
    np.savetxt('results/optimized_mask.txt',phase.cpu().detach().numpy()*2*np.pi)
    end = time.time()
    print('Elapsed time in training: ',end-start,'s')
    """# ******************** End Training!******************************"""

    """ Use optimized phase mask and calculate the image plane intensity """
    phase1 = phase*2*torch.pi
    finalE, xi, yi = band_limit_ASM(initAmp*torch.exp(1j*phase1),f,mesh,1,lambda0) 
    final_I = abs(finalE)**2
    final_I /= torch.max(final_I)

    """ Calculate image plane intensity using initial random phase"""
    init_imagE ,xi0 ,yi0 = band_limit_ASM(initAmp*torch.exp(1j*initPhase*2*torch.pi),f,mesh,1,lambda0) 
    init_imagI = abs(init_imagE)**2
    init_imagI /= torch.max(init_imagI)

    """ Compare optimized phase and intensity with initial phase and target intensity """
    fig, axs= plt.subplots(2,3)
    im1 = axs[0,0].imshow(detach(initPhase),origin='lower',extent=[x_lens[0],x_lens[-1],y_lens[0],y_lens[-1]],cmap='jet',aspect='equal')
    axs[0,0].set_title('Initial phase mask')
    fig.colorbar(im1,ax=axs[0,0])
    im2 = axs[1,0].imshow(detach(phase),origin='lower',extent=[x_lens[0],x_lens[-1],y_lens[0],y_lens[-1]],cmap='jet',aspect='equal')
    axs[1,0].set_title('Optimized phase mask')
    fig.colorbar(im2,ax=axs[1,0])
    im3 = axs[0,1].imshow(detach(init_imagI),origin='lower',extent=[xw[0],xw[-1],yw[0],yw[-1]],cmap='jet',aspect='equal')
    axs[0,1].set_title('Initial intensity on image plane')
    fig.colorbar(im3,ax=axs[0,1])
    im4 = axs[1,1].imshow(detach(final_I),origin='lower',extent=[xw[0],xw[-1],yw[0],yw[-1]],cmap='jet',aspect='equal')
    axs[1,1].set_title('Optimized intensity on image plane')
    fig.colorbar(im4,ax=axs[1,1])
    im5 = axs[0,2].imshow(detach(target_I),origin='lower',extent=[xw[0],xw[-1],yw[0],yw[-1]],cmap='jet',aspect='equal')
    axs[0,2].set_title('Target intensity')
    fig.colorbar(im5,ax=axs[0,2])
    im6 = axs[1,2].plot(record['N'],record['Loss'],linewidth=2)
    axs[1,2].set_title('Loss')


    """ Plot the 1D intensity cross section"""
    final_I_max, final_I_row, final_I_col = findMax2D(final_I)
    final_I_slice = final_I[final_I_row].cpu().detach().numpy()
    target_I_slice = target_I[int(len(target_I)/2)].cpu().detach().numpy()
    xData = xw.cpu().detach().numpy()
    fig = plt.figure(figsize=[7.5,6])
    ax1 = fig.add_subplot(111)
    plotData1 = plt.plot(xData,target_I_slice,linewidth=2,label='target')
    plotData2 = plt.plot(xData,final_I_slice,linewidth=2,label='optimized')
    plt.title('1D intensity comparison')
    plt.legend()

    plt.show()
    # 

if __name__ == '__main__':
    main()