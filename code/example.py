# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:52:33 2022

@author: Delinte Nicolas

"""

import os
import TIME
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

if __name__=='__main__':
    
    os.chdir('..')
    
    trk_file='data/sampleSubject_cc_bundle_mid_ant.trk'
    MF_dir='data/'
    Patient='sampleSubject'

    fixelWeights,_,_,voxelStreams,_=TIME.tractToMFpop(trk_file, MF_dir, Patient,sNum=560)

    metricMapList=[nib.load('data/sampleSubject_mf_fvf_f0.nii.gz').get_fdata(),
                   nib.load('data/sampleSubject_mf_fvf_f1.nii.gz').get_fdata()]

    microMap=TIME.getMicrostructureMap(trk_file, fixelWeights, metricMapList)
    
    # Plotting results --------------------------------------------------------
    
    background=nib.load('data/sampleSubject_T1_diffusionSpace.nii.gz').get_fdata()
    totalSegmentLength=np.sum(fixelWeights,axis=3)
    totalSegmentLengthTransparency=totalSegmentLength/np.max(totalSegmentLength)
    tSL=totalSegmentLength.copy()
    tSL[totalSegmentLength>0]=1

    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(np.rot90(TIME.getMainFixel(fixelWeights)[:,71,:]),cmap='gray')
    axs[0].set_title('Most aligned fixel')
    axs[1].imshow(np.rot90(totalSegmentLength[:,71,:]),cmap='inferno',clim=[0,80])
    axs[1].set_title('Total segment length')
    axs[2].imshow(np.rot90(background[:,71,:]),cmap='gray')
    axs[2].imshow(np.rot90(tSL[:,71,:]),alpha=np.rot90(totalSegmentLengthTransparency[:,71,:]),cmap='Wistia')
    axs[2].set_title('Total segment length')
    axs[3].imshow(np.rot90(microMap[:,71,:]),cmap='gray')
    axs[3].set_title('Fiber volume fraction \n (axonal density) map')
    
    TIME.plotStreamlineMetrics(voxelStreams,metricMapList)
