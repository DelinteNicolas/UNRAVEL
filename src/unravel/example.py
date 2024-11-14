# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:52:33 2022

@author: Delinte Nicolas

An example python script using the main methods and outputs of UNRAVEL. A
tractogram of the middle anterior section of the corpus callosum is used as
tractography input.

"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from dipy.io.streamline import load_tractogram
from unravel.core import (get_fixel_weight, get_microstructure_map,
                          get_weighted_mean, main_fixel_map,
                          plot_streamline_metrics, total_segment_length)
from unravel.utils import plot_streamline_trajectory, get_streamline_density


if __name__ == '__main__':

    os.chdir('../..')

    data_dir = 'data/'
    sub = 'sampleSubject'
    trk_file = data_dir+sub+'_cc_bundle_mid_ant.trk'
    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()

    # Maps and means ----------------------------------------------------------

    peaks = np.stack((nib.load(data_dir+sub+'_mf_peak_f0.nii.gz').get_fdata(),
                      nib.load(data_dir+sub+'_mf_peak_f1.nii.gz').get_fdata()),
                     axis=4)

    fixel_weights = get_fixel_weight(trk, peaks)

    metric_maps = np.stack((nib.load(data_dir+sub+'_mf_fvf_f0.nii.gz').get_fdata(),
                            nib.load(data_dir+sub+'_mf_fvf_f1.nii.gz').get_fdata()),
                           axis=3)

    micro_map = get_microstructure_map(fixel_weights, metric_maps)

    weightedMean, weightedDev = get_weighted_mean(micro_map, fixel_weights)

    # Colors ------------------------------------------------------------------

    fList = np.stack((nib.load(data_dir+sub+'_mf_fvf_f0.nii.gz').get_fdata(),
                      nib.load(data_dir+sub+'_mf_fvf_f1.nii.gz').get_fdata()),
                     axis=3)

    rgb = get_streamline_density(trk, color=True)

    # Total segment length ----------------------------------------------------

    tsl = total_segment_length(fixel_weights)

    # Printing means ----------------------------------------------------------

    print('The fiber volume fraction estimation of '+sub+' in the middle '
          + 'anterior bundle of the corpus callosum are \n'
          + 'Weighted mean : '+str(weightedMean)+'\n'
          + 'Weighted standard deviation : '+str(weightedDev)+'\n')

    # Plotting results --------------------------------------------------------

    slice_num = 71

    background = nib.load(
        'data/sampleSubject_T1_diffusionSpace.nii.gz').get_fdata()
    roi = np.where(tsl > 0, .99, 0)
    non_roi = np.where(tsl == 0, .99, 0)
    alpha_tsl = tsl[:, slice_num, :]/np.max(tsl)*2
    alpha_tsl[alpha_tsl > 1] = 1

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(np.rot90(background[:, slice_num, :]), cmap='gray')
    axs[0, 0].imshow(np.rot90(main_fixel_map(fixel_weights)[:, slice_num, :]),
                     cmap='Wistia', alpha=np.rot90(roi[:, slice_num, :]))
    axs[0, 0].set_title('Most aligned fixel')
    axs[0, 1].imshow(np.rot90(rgb[:, slice_num, :]))
    axs[0, 1].imshow(np.rot90(background[:, slice_num, :]), cmap='gray',
                     alpha=np.rot90(non_roi[:, slice_num, :]))
    axs[0, 1].set_title('Angular weighted \n direction')
    axs[1, 0].imshow(np.rot90(background[:, slice_num, :]), cmap='gray')
    axs[1, 0].imshow(np.rot90(roi[:, slice_num, :]), cmap='Wistia',
                     alpha=np.rot90(alpha_tsl))
    axs[1, 0].set_title('Total segment length')
    axs[1, 1].imshow(np.rot90(background[:, slice_num, :]), cmap='gray')
    fvf = axs[1, 1].imshow(np.rot90(micro_map[:, slice_num, :]), cmap='autumn',
                           alpha=np.rot90(roi[:, slice_num, :]))
    fig.colorbar(fvf, ax=axs[1, 1])
    axs[1, 1].set_title('Fiber volume fraction \n (axonal density) map')

    # Along streamline metric --------------------------------------------------

    stream_num = 500

    plot_streamline_trajectory(trk, resolution_increase=3,
                               streamline_number=stream_num, axis=1)

    plot_streamline_metrics(trk, peaks, metric_maps,
                            method_list=['vol', 'cfo', 'ang'],
                            streamline_number=stream_num, ff=fList)
