# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:52:33 2022

@author: Delinte Nicolas

An example pyhton script using the main methods and outputs of TIME. A
tractogram of the middle anterior section of the corpus callosum is used as
tractography input.

"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from TIME.core import *

if __name__ == '__main__':

    os.chdir('..')

    trk_file = 'data/sampleSubject_cc_bundle_mid_ant.trk'
    MF_dir = 'data/'
    Patient = 'sampleSubject'

    fixelWeights, _, _, voxelStreams, _ = get_fixel_weight_MF(
        trk_file, MF_dir, Patient, streamList=[0])

    metricMapList = [nib.load('data/sampleSubject_mf_fvf_f0.nii.gz').get_fdata(),
                     nib.load('data/sampleSubject_mf_fvf_f1.nii.gz').get_fdata()]

    microMap = get_microstructure_map(fixelWeights, metricMapList)

    weightedMean, weightedDev, _, [Min, Max] = weighted_mean_dev(
        metricMapList, [fixelWeights[:, :, :, 0], fixelWeights[:, :, :, 1]])

    # Printing means ----------------------------------------------------------

    print('The fiber volume fraction estimation of '+Patient+' in the middle '
          + 'anterior bundle of the corpus callosum are \n'
          + 'Weighted mean : '+str(weightedMean)+'\n'
          + 'Weighted standard deviation : '+str(weightedDev)+'\n'
          + 'Min/Max : '+str(Min), str(Max)+'\n')

    # Plotting results --------------------------------------------------------

    background = nib.load(
        'data/sampleSubject_T1_diffusionSpace.nii.gz').get_fdata()
    totalSegmentLength = np.sum(fixelWeights, axis=3)
    totalSegmentLengthTransparency = totalSegmentLength / \
        np.max(totalSegmentLength)
    tSL = totalSegmentLength.copy()
    tSL[totalSegmentLength > 0] = 1

    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(np.rot90(main_fixel_map(
        fixelWeights)[:, 71, :]), cmap='gray')
    axs[0].set_title('Most aligned fixel')
    axs[1].imshow(np.rot90(totalSegmentLength[:, 71, :]),
                  cmap='inferno', clim=[0, 80])
    axs[1].set_title('Total segment length')
    axs[2].imshow(np.rot90(background[:, 71, :]), cmap='gray')
    axs[2].imshow(np.rot90(tSL[:, 71, :]), alpha=np.rot90(
        totalSegmentLengthTransparency[:, 71, :]), cmap='Wistia')
    axs[2].set_title('Total segment length')
    axs[3].imshow(np.rot90(microMap[:, 71, :]), cmap='gray')
    axs[3].set_title('Fiber volume fraction \n (axonal density) map')

    plot_streamline_metrics(voxelStreams, metricMapList)
