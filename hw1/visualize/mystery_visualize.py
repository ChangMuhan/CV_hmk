#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb


def colormapArray(X, colors):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap. See the Bewares
    """
    X_copy=np.copy(X)
    vmin=np.nanmin(X)
    vmax=np.nanmax(X)
    N=colors.shape[0]
    
    if vmin==vmax:
        middle=colors[(N-1)//2]
        color_unit8=(middle*(N-1).astype(np.uint8)
        return np.tile(color_unit8,(X_copy.shape[0],X_copy.shape[1],1))
    
    X_copy[np.isnan(X_copy)]=vmin
    X_copy[np.isinf(X_copy)]=vmax
    
    index=(N-1)*(X_copy-vmin)/(vmax-vmin)
    index=index.astype(np.int32)

    color_image=colors[index]
    final=(color_image*(N-1).astype(np.uint8)
    return final


if __name__ == "__main__":
    colors = np.load("mysterydata/colors.npy")
    data = np.load("mysterydata/mysterydata.npy")
    
    X=np.load("mysterydata/mysterydata2.npy")
    Y=np.load("mysterydata/mysterydata3.npy")
    Z=np.load("mysterydata/mysterydata4.npy")

    
    X_transformed1=np.log1p(X)
    

    channel1=X_transformed1[:,:,0]
    plt.imsave("channel1.png",channel1)
   

    channel2=X_transformed1[:,:,1]
    plt.imsave("channel2.png",channel2)
    
    if np.isnan(Y).any():
        nan_count = np.isnan(X).sum()
        print(f"there are {nan_count} color value problems")
    else:
        print("No problem")
    
    channel3=Y[:,:,2]
    channel3_min=np.nanmin(channel3)
    channel3_max=np.nanmax(channel3)
    print("min value in channel3:",channel3_min)
    print("max value in channel3:",channel3_max)
    channel3_fixed=np.copy(channel3)
    channel3_fixed[np.isnan(channel3_fixed)]=channel3_min
    plt.imsave("channel3.png",channel3_fixed,vmin=channel3_min,vmax=channel3_max)

    channel4=Y[:,:,3]
    channel4_min=np.nanmin(channel4)
    channel4_max=np.nanmax(channel4)
    print("min value in channel4:",channel4_min)
    print("max value in channel4:",channel4_max)
    channel4_fixed=np.copy(channel4)
    channel4_fixed[np.isnan(channel4_fixed)]=channel4_min
    plt.imsave("channel4.png",channel4_fixed,vmin=channel4_min,vmax=channel4_max)

    
    fig,axes=plt.subplots(3,3,figsize=(15,15))
    num_channels=Z.shape[2]

    for i in range(num_channels):
        z_channel=Z[:,:,i]
        color_image=colormapArray(z_channel,colors)
        
        row=i//3
        col=i%3

        ax=axes[row,col]
        ax.imshow(color_image)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("z_all_channels.png")

    pdb.set_trace()
