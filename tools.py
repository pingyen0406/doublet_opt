"""Some common used functions"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
    Ez = w0/wz*torch.exp(-(gX**2+gY**2)/wz**2)*torch.exp(1j*(k0*z+k0*(gX**2+gY**2)/2/Rz))
    return Ez

# Create a rect function
def rect(all_size,center,rect_size):
    """
    ****Generating a rectangle region has value 1****

    lambda0: operating wavelength
    x: x grid
    y: y grid
    center: list, center of the rectangle (index number)
    size: list, size of the rectangle (index number)
    """
    center = center.astype(int)
    rect_size = rect_size.astype(int)
    rect = torch.zeros(all_size)
    rect[ center[1]-int(rect_size[1]/2) : center[1]+int(rect_size[1]/2),
          center[0]-int(rect_size[0]/2) : center[0]+int(rect_size[0]/2) ]=1
    return rect

# 2D image plot
def plotField(data,xData,yData,title=None):
    """Plot the field distribution"""
    data = data.cpu().detach().numpy()
    xData = xData.cpu().detach().numpy()
    yData = yData.cpu().detach().numpy()
    fig = plt.figure(figsize=[7.5,6])
    ax1 = fig.add_subplot(111)
    plotData = plt.imshow(data,origin='lower',extent=[xData[0],xData[-1],yData[0],yData[-1]],
                            cmap='hot',aspect='auto')
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
# Find the maximum value and its index (row and column)
def findMax2D(data):
    max_tuple = torch.max(data, dim=1)
    max_row_index = torch.argmax(max_tuple[0])
    max_col_index = torch.argmax(data[max_row_index])
    return max_tuple[0], max_row_index, max_col_index
# Convert tensor to ndarray 
def detach(data):
    return data.cpu().detach().numpy()

# Generate 1D SLM source
def line_SLM_source(N,all_size,beam_w,pitch,lambda0=1.55):
    """
    Generate a uniform distributed 1D SLM light source seperately

    N: number of SLM pixels
    all_size: shape of the matrix contains the SLM
    beam_w: beam size (pixel)
    pitch: pitch between beams (pixel)

    output: (N*len(x)*len(y))complex tensor with N-th SLM pixel on 
    """
    slm_source = torch.empty((N,all_size[1],all_size[0]))
    c_index = int(len(all_size)/2)

    for i in range(N):
        slm_source[i] = rect(all_size,(c_index,c_index+pitch*(i-int(N/2))),(beam_w,beam_w))
        #slm_source[i] = gaussian(1.55,5,0.001,x-pitch*5*(i-int(N/2)),y)
    return slm_source



# Normailze phase between [0,2pi]
def norPhase(phaseData):
    phaseData -=  torch.min(phaseData)
    size = phaseData.shape
    for i in range(size[0]):
        for j in range(size[1]):
            while phaseData[i,j] > 2*torch.pi:
                phaseData[i,j] -= 2*torch.pi
            while phaseData[i,j] < 0:
                phaseData[i,j] += 2*torch.pi
    return phaseData
