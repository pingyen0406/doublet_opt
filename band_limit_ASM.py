from colorsys import yiq_to_rgb
from tkinter import image_names
import torch.nn as nn
import torch
import numpy as np
from matplotlib import pyplot as plt
from tools import gaussian

# create a band-limited transfer matrix(k-domain)
def band_limit_ASM(source, prop_z, mesh, padRatio=1,lambda0=1.55,device='cpu',cut=False):
    """
    ****Band-limited angular spectrum method****

    source: u0 at input plane, must in a "square matrix"
    propz: propagation distance
    mesh: mesh of the input source
    padRatio: zero padding size with respect to the input source
    lambda0: operating wavelength
    device: 'cuda' or 'cpu'

    output: (E,x,y)
    E: complex E_field on the image plane
    x: x coordinate tensor
    y: y coordinate tensor
    """
    # read input source and create its meshgrid
    ny,nx = list(source.size())
    if ny != nx:
        raise Exception('The input source is not a square matrix!')
    xs = torch.tensor([mesh*i for i in range(nx)]) 
    ys = torch.tensor([mesh*i for i in range(ny)])
    xs = xs - torch.median(xs)
    ys = ys - torch.median(ys)
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
    xw = xw - torch.median(xw)
    yw = yw - torch.median(yw)
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
    gamma = torch.tensor(gamma,dtype=torch.complex128)
    gamma = gamma * band_matrix # Combine band-lmited matrix and gamma (z-direction)
    prop_matrix = torch.exp(1j*2*torch.pi/lambda0*prop_z*gamma)
    window, prop_matrix = window.to(device), prop_matrix.to(device)
    # Propagation!
    window_f = torch.fft.fftshift(torch.fft.fft2(window))
    
    image_E = torch.fft.ifft2(torch.fft.fftshift(window_f*prop_matrix))
    # Cut out the region used for zero-padding if needed
    if cut==True:
        image_E = image_E[pad_ny:-pad_ny,pad_nx:-pad_nx]
        return image_E, xs, ys
    else:
        return image_E, xw, yw

def main():
    # Check ASM with gaussian beam
    lambda0 = 1.55
    mesh = 1    
    prop_z = 1000
    w0 = 3
    x = torch.tensor([i for i in range(501)])
    y = torch.tensor([i for i in range(501)])
    x = x - torch.median(x)
    y = y - torch.median(y)

    # Using ASM
    Ez0 = gaussian(lambda0,w0,0.001,x,y)
    Ez0,xi,yi = band_limit_ASM(Ez0,prop_z,mesh,padRatio=1,cut=True)
    Iz0 = torch.abs(Ez0)**2
    Iz0_slice = Iz0[int(len(x)/2)]
    Iz0_slice = Iz0_slice / torch.max(Iz0)
    Iz0_slice = Iz0_slice.cpu().detach().numpy()

    # Analytical gaussian
    Ez1 = gaussian(lambda0,w0,prop_z,x,y)
    Iz1 = torch.abs(Ez1)**2
    Iz1_slice = Iz1[int(len(x)/2)]
    Iz1_slice = Iz1_slice / torch.max(Iz1)
    Iz1_slice = Iz1_slice.cpu().detach().numpy()

    plt.figure()
    plt.plot(x,Iz0_slice,linewidth=2)
    plt.plot(x,Iz1_slice,linewidth=2)
    plt.show()

if __name__ == '__main__':
    main()