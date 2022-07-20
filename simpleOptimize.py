# Simple optimizer to obtain a focusing wavefront
"""********** Use tensor instead of ndarray in the script **********"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.functional import F
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

# Create a Gausian beam profile
def gaussian(lambda0,w0,z,x,y):
    """
    ****Generating a gaussion function****

    lambda0: operating wavelength
    w0: beam waist
    z: distance from waist
    x: x grid
    y: y grid
    """

    gX,gY = torch.meshgrid(x,y)
    k0 = 2*np.pi/lambda0
    zr = np.pi*w0**2/lambda0
    Rz = z*(1+(zr/z)**2)
    wz = w0*np.sqrt(1+(z/zr)**2)
    Ez = w0/wz*torch.exp(-(gX**2+gY**2)/wz**2)#*torch.exp(1j*(k0*z+k0*(gX**2+gY**2)/2/Rz))
    return Ez

# Create a rect function
def rect(x,y,center,size):
    """
    ****Generating a rectangle region has value 1****

    lambda0: operating wavelength
    x: x grid
    y: y grid
    center: list, center of the rectangle (index number)
    size: list, size of the rectangle (index number)
    """
    gX,gY = torch.meshgrid(x,y)
    rect = torch.zeros(gX.shape)
    rect[center[1]-int(size[1]/2):center[1]+int(size[1]/2),center[0]-int(size[0]/2):center[0]+int(size[0]/2)]=1
    return rect

# create a band-limited transfer matrix(k-domain)
def band_limit_ASM(source, prop_z, mesh, padRatio=1,lambda0=1.55,device=device,cut=False):
    """
    ****Band-limited angular spectrum method****

    source: u0 at input plane, must in a "square matrix"
    propz: propagation distance
    mesh: mesh of the input source
    padRatio: zero padding size with respect to the input source
    lambda0: operating wavelength
    device: 'cuda' or 'cpu'
    """
    # read input source and create its meshgrid
    ny,nx = list(source.size())
    if ny != nx:
        raise Exception('The input source is not a square matrix!')
    xs = torch.tensor([mesh*i for i in range(nx)]) 
    ys = torch.tensor([mesh*i for i in range(ny)])
    xs -= torch.median(xs)
    ys -= torch.median(ys)
    Xs,Ys = torch.meshgrid(xs,ys)
    
    # Padding zero to the source aperture
    # Create simulation window meshgrid
    pad_nx = int(padRatio*nx)
    pad_ny = int(padRatio*ny)
    if pad_nx%2 ==0:
        pad_nx+=1
    if pad_ny%2 ==0:
        pad_ny+=1
    x_width_w = (2*pad_nx+nx-1)*mesh
    y_width_w = (2*pad_ny+ny-1)*mesh
    xw = torch.tensor([mesh*i for i in range(int(2*pad_nx+nx))]) 
    yw = torch.tensor([mesh*i for i in range(int(2*pad_ny+ny))])
    xw -= torch.median(xw)
    yw -= torch.median(yw)
    Xw,Yw = torch.meshgrid(xw,yw)
    Ny,Nx = list(Xw.size())
    padding = nn.ZeroPad2d((pad_nx,pad_ny,pad_nx,pad_ny))
    window = padding(source)
    # Create corersponding meshgrid in freq-domain for simulation window
    # See FT_sampling.pdf in the folder for reference
    Ny,Nx = list(window.size())
    Fx_max = 1/(2*mesh)
    Fy_max = 1/(2*mesh)
    dFx = 1/x_width_w
    dFy = 1/y_width_w
    fx = np.linspace(-Fx_max,Fx_max,Nx)
    fy = np.linspace(-Fy_max,Fy_max,Ny)
    fX,fY = np.meshgrid(fx,fy)
    alpha = lambda0*fX
    beta = lambda0*fY
    # Create band-limited matrix
    ux_lim = 1/np.sqrt(4*dFx**2*prop_z**2+1)/lambda0
    uy_lim = 1/np.sqrt(4*dFy**2*prop_z**2+1)/lambda0    
    ux_lim_n = np.argwhere(abs(fx)<ux_lim)
    uy_lim_n = np.argwhere(abs(fy)<uy_lim)
    band_matrix = torch.ones((len(uy_lim_n),len(ux_lim_n)))
    padding_band = nn.ZeroPad2d(int((Nx-len(ux_lim_n))/2))
    band_matrix = padding_band(band_matrix)

    gamma = np.sqrt(1-alpha**2-beta**2,dtype=complex) # Assume alpha**2+beta**2 < 1
    gamma = torch.tensor(gamma)
    gamma*=band_matrix # Combine band-lmited matrix and gamma (z-direction)
    prop_matrix = torch.exp(1j*2*torch.pi/lambda0*prop_z*gamma)
    window, prop_matrix = window.to(device), prop_matrix.to(device)
    # Propagation!
    target = plane_prop(window,prop_matrix)
    # Cut out the region used for zero-padding if needed
    if cut==True:
        target = target[int((Ny-ny)/2):int((Ny+ny)/2),int((Nx-nx)/2):int((Nx+nx)/2)]
        return target, xs, ys
    else:
        return target, xw, yw
# ASM Propagation 
def plane_prop(source, prop_matrix):
    source_f = torch.fft.fftshift(torch.fft.fft2(source))
    target = torch.fft.ifft2(torch.fft.fftshift(source_f*prop_matrix)) 
    return target

# Find the maximum value and its index (row and column)
def findMax2D(data):
    max_tuple = torch.max(data, dim=1)
    max_row_index = torch.argmax(max_tuple[0])
    max_col_index = torch.argmax(data[max_row_index])
    return max_tuple[0], max_row_index, max_col_index

# 2D image plot
def plotField(data,xData,yData,title=None):
    """Plot the field distribution"""
    data = data.cpu().detach().numpy()
    xData = xData.cpu().detach().numpy()
    yData = yData.cpu().detach().numpy()
    fig = plt.figure(figsize=[7.5,6])
    ax1 = fig.add_subplot(111)
    plotData = plt.imshow(data,origin='lower',extent=[xData[0],xData[-1],yData[0],yData[-1]],
                            cmap='jet',aspect='auto')
    plt.title(title)
    plt.colorbar()
    return fig
# 2 image plots share same coordinates
def plot2Field(data1,data2,xData,yData,title1=None,title2=None):
    """Compare 2 field, and they should share the same x,y coordinates"""
    data1 = data1.cpu().detach().numpy()
    data2 = data2.cpu().detach().numpy()
    xData = xData.cpu().detach().numpy()
    yData = yData.cpu().detach().numpy()
    fig, axs= plt.subplots(1,2)
    im1 = axs[0].imshow(data1,origin='lower',extent=[xData[0],xData[-1],yData[0],yData[-1]],cmap='jet',aspect='auto')
    axs[0].set_title(title1)
    fig.colorbar(im1,ax=axs[0])
    im2 = axs[1].imshow(data2,origin='lower',extent=[xData[0],xData[-1],yData[0],yData[-1]],cmap='jet',aspect='auto')
    axs[1].set_title(title2)
    fig.colorbar(im2,ax=axs[1])
    return fig

def detach(data):
    return data.cpu().detach().numpy()

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
    target_image = Image.open('testZ.png')
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
    np.savetxt('optimized_mask.txt',phase.cpu().detach().numpy()*2*np.pi)
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