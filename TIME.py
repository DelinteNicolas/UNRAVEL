# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:05:25 2022

@author: DELINTE  Nicolas

"""

import os
import numpy as np
import nibabel as nib
from dipy.io.image import load_nifti, save_nifti
from dipy.io.streamline import load_tractogram

def deltasToD(dx,dy,dz):
    
    e=np.array([[dx, -dz-dy, dy*dx-dx*dz],
                [dy, dx, -dx**2-(dz+dy)*dz],
                [dz, dx, dx**2+(dy+dz)*dy]])
            
    try:
        e_1=np.linalg.inv(e)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError
    
    lamb=np.diag([1,0,0])           # Arbitrary
    
    D=(e.dot(lamb)).dot(e_1)/500    # Change value to change vector length
    
    return D

def voxelDistance(position1:tuple,position2:tuple):
    
    dis=abs(np.floor(position1)-np.floor(position2))
    
    return dis

def getVoxelsFromSegment(position1:tuple,position2:tuple,subparts:int=10)->dict:
    '''
    

    Parameters
    ----------
    position1 : tuple
        DESCRIPTION.
    position2 : tuple
        DESCRIPTION.
    subparts : int, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    dict
        DESCRIPTION.

    '''
    
    voxDis=sum(voxelDistance(position1,position2))
    
    voxList={}
    
    if voxDis==0:                        #Trying to gain time
        voxList[tuple(np.floor(position1))]=1
        
        return voxList
    
    subseg=np.linspace(position1,position2,subparts)

    for i in subseg:
        xyz=tuple(np.floor(i))
        if xyz not in voxList:
            voxList[xyz]=1/subparts
        else:
            voxList[xyz]+=(1/subparts)
            
    return voxList

def compute_subsegments(start, finish, vox_size=[1,1,1], offset=[0,0,0], return_nodes=False):
    '''
    Parameters
    ----------
    start : 1-D array of shape (d,)
        Starting point of segment.
    finish : 1-D array of shape (d,)
        End point of segment.
    vox_size : 1-D array of shape (d,)
        vox_size[i] is the (strictly positive) voxel size along the i-th axis,
        with i=1,...,d.
    offset : 1-D array of shape (d,)
        offset[i] is the start of the first voxel along the i-th axis, with
        i=1,...,d.
        In 3 dimensions (d=3), voxel (0, 0, 0) would be defined as
        [off_x, off_x + s_x[ x [off_y, off_y + s_y[ x [off_z, off_z + s_z[,
        where off_x = offset[0], s_x = vox_size[0], off_y = offset[1], etc.
    return_nodes : boolean, optional
        If True, the nodes making the sub-segment(s) are returned.
        The default is False.

    Returns
    -------
    tuple
        subseg_lengths: list of N-1 scalar(s) of type float, with N>=2.

        visited_voxels: list of N-1 1-D array(s) of shape (d,) and type int,
            with N>=2.

        nodes: only if return_nodes=True. List of N arrays of shape (d,) and
            type float, with N>=2.

    '''
    
    start=np.asarray(start)
    finish=np.asarray(finish)
    
    tol_lam = 1e-10

    # Extension of substep to move away from voxel boundaries and avoid
    # numerical round-off errors. Chosen such that will never leave to a voxel
    # crossing when starting from a boundary
    step_ext = np.min(vox_size) * 0.01

    step = finish - start
    Lres = np.sqrt(np.sum(step**2))  # residual step length
    cur = start  # current node

    # Output quantities
    subseg_lengths = []
    visited_voxels = []
    nodes = [start]

    while Lres > 0:
        vox_ID_cur = np.floor((cur-offset)/vox_size).astype(int)
        vox_start = offset + vox_ID_cur * vox_size

        # Case cur at voxel start but step is negative. Decrement voxel ID,
        # consider curr at the end of previous voxel
        decr_vox_ID = (cur == vox_start) & (step < 0)
        if np.any(decr_vox_ID):
            vox_ID_cur[decr_vox_ID] = vox_ID_cur[decr_vox_ID] - 1
            vox_start = offset + vox_ID_cur * vox_size

        # We now work in a centered voxel, i.e. a domain defined by
        # [0, vox_size[0]] x [0, vox_size[1]] x [0, vox_size[2]]
        # We search for smallest lambda in ]0, 1] such that one component i of
        # (cur-vox_start) + lambda * step
        # hits 0 or vox_size[i]
        step_nnz = step.copy()
        step_nnz[step == 0] = 1  # avoid division by 0
        # Lambda to reach start of voxel at 0
        lam_0 = (0.0 - (cur-vox_start))/step_nnz
        lam_0[step == 0] = 1
        # Lambda to reach end of voxel at vox_size[i] for all i
        lam_vs = (vox_size - (cur-vox_start))/step_nnz
        # note: if cur artificially considered at end of previous voxel (see
        # above), ignore small displacement to outer edge
        lam_vs[(step == 0) | decr_vox_ID] = 1
        # Find minimum lambda in ]0, 1]. If all > 1, it means the current step
        # can be completely performed => lambda = 1
        lam_cand = np.concatenate((lam_0, lam_vs))
        lam_cand[(lam_cand <= 0) | (lam_cand > 1)] = 1
        lam = np.min(lam_cand)

        old = cur
        # New location, either at finish or at voxel boundary
        if np.abs(lam - 1) < tol_lam:
            # avoid numerical round-off errors at last sub-step. Typically
            # happens when a step is along a voxel boundary.
            lam = 1
            cur = finish
        else:
            cur = cur + lam * step

        # Use mid-point between new and old node to get voxel in which
        # sub-segment lies:
        midpoint = (old + cur)/2
        vox_ID = np.floor((midpoint-offset)/vox_size).astype(int)

        if lam > tol_lam:
            # Update history (ignore if substep is too small and same voxel
            # visited twice, might happen at corners)
            subseg_lengths.append(lam * Lres)
            visited_voxels.append(vox_ID)
            nodes.append(cur)

        # Residual step
        step = (1-lam) * step  #  equivalent to finish - cur
        Lres = (1-lam)*Lres  #  or np.sqrt(np.sum(step**2))

    out = [subseg_lengths, visited_voxels]
    if return_nodes:
        out.append(nodes)
        
    voxList={}
    for i,sub in enumerate(subseg_lengths):
        voxList[tuple(visited_voxels[i])]=sub
        
    return voxList
    
def angleBetweenVectors(v1,v2):
    v1n=v1/np.linalg.norm(v1)
    v2n=v2/np.linalg.norm(v2)
    
    if (v1n==v2n).all():
        return 0
    
    ang=np.arccos(sum(v1n*v2n))*180/np.pi
    
    if ang > 90:
        ang=180-ang
    
    return ang
    
def angularWeight(vs,vList,nList):
    '''
    Parameters
    ----------
    vs : segment vector
    vList : list of the k vectors corresponding to each fiber population

    Returns
    -------
    ang_coef: list of the k coefficients

    '''
    
    if len(vList)==1:
        return [1]
    
    if len(vList)-sum(nList)<=1:
        return [1-i for i in list(map(int,nList))]
    
    angle_diffList=[]
    
    for i,v in enumerate(vList):
        if nList[i]:
            angle_diffList.append(0)
        else:
            angle_diffList.append(angleBetweenVectors(vs,v))
        
    sum_diff=np.sum(angle_diffList)
    
    ang_coef=[]
    
    for i,angle_diff in enumerate(angle_diffList):
        if nList[i]:
            ang_coef.append(0)
        else:
            coef=1-angle_diff/sum_diff
            coef=coef/(len(vList)-1-sum(nList))
            ang_coef.append(coef)
    
    return ang_coef

def t6ToMFpeak(t):
    '''
    (6,17,1,15) with info on 0,2,5 to (17,1,15,3)
    '''
    
    new_t=np.transpose(t,(1,2,3,0))
    new_t[:,:,:,0]=new_t[:,:,:,0]
    new_t[:,:,:,1]=new_t[:,:,:,2]
    new_t[:,:,:,2]=new_t[:,:,:,5]
    new_t=new_t[:,:,:,:3]
    
    t1=nib.load('C:/users/nicol/Documents/Doctorat/Data/Phantom/Diamond/LUCFRD_diamond_t0.nii.gz')
    
    
    out=nib.Nifti1Image(new_t,np.eye(4))#,t1.header)
    out.to_filename('C:/users/nicol/Desktop/LUCFRD_peak_f.nii.gz')
    
    return new_t

def MFpeakToDIAMONDpop(peaks):
    
    t=np.zeros(peaks.shape[:3]+(1,6))
    
    for xyz in np.ndindex(peaks.shape[:3]):
        
        if peaks[xyz].all()==0:
            continue
    
        dx,dy,dz=peaks[xyz]
        
        try:
            D=deltasToD(dx,dy,dz)
        except np.linalg.LinAlgError:
            continue
        
        t[xyz+(0,0)]=D[0,0]
        t[xyz+(0,1)]=D[0,1]
        t[xyz+(0,2)]=D[1,1]
        t[xyz+(0,3)]=D[0,2]
        t[xyz+(0,4)]=D[1,2]
        t[xyz+(0,5)]=D[2,2]
    
    return t

def DIAMONDpopToMFpeak(t):
    '''
    Too slow for now
    
    Parameters
    ----------
    t : DIAMOND t array

    Returns
    -------
    peaks : peak array

    '''
    
    if len(t.shape)==4:
        t=t[...,np.newaxis]
        t=np.transpose(t,(1,2,3,4,0))
        
        D_t=np.array([[t[:,:,:,0,0],t[:,:,:,0,1],t[:,:,:,0,2]],
                      [t[:,:,:,0,1],t[:,:,:,0,3],t[:,:,:,0,4]],
                      [t[:,:,:,0,2],t[:,:,:,0,4],t[:,:,:,0,5]]])
        
    else:
    
        D_t=np.array([[t[:,:,:,0,0],t[:,:,:,0,1],t[:,:,:,0,3]],
                      [t[:,:,:,0,1],t[:,:,:,0,2],t[:,:,:,0,4]],
                      [t[:,:,:,0,3],t[:,:,:,0,4],t[:,:,:,0,5]]])
    
    D_t=np.transpose(D_t,(2,3,4,0,1))
    
    val_t,vec_t=np.linalg.eig(D_t)
    
    vol_shape=t.shape[0]*t.shape[1]*t.shape[2]
    
    vec_t=vec_t.reshape((vol_shape,3,3))
    vec_t=np.transpose(vec_t, (0, 2, 1))
    idx=np.argmax(val_t.reshape((vol_shape,3)),axis=1)
    
    peaks=vec_t[range(vol_shape),idx].reshape(t.shape[:3]+(3,)).real
        
    return peaks

def tractToMFpop(trk_file:str,MF_dir:str,K:int=2,binary:bool=False,sNum:int=80):
    '''
    

    Parameters
    ----------
    trk_file : str
        DESCRIPTION.
    MF_dir : str
        DESCRIPTION.
    K : int, optional
        DESCRIPTION. The default is 2.
    binary : bool, optional
        DESCRIPTION. The default is False.
    sNum : int, optional
        DESCRIPTION. The default is 80.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    # Tract -----------------
    
    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    
    # MF peaks --------------
    
    tList=[]
    for k in range(K):
        # !!!
        img=nib.load(MF_dir+'_peak_f'+str(k)+'.nii.gz')
        t=img.get_fdata()
    
        # If finger peaks organised as ...
        if not t.shape[3]==3 and t.shape[0]==6:
            t=t6ToMFpeak(t) 
        
        tList.append(t)
    
    return tractToFiberPop(trk, tList, binary,sNum)

def tractToDIAMONDpop(trk_file:str,DIAMOND_dir:str,K:int=2,binary:bool=False,sNum:int=80):
    '''
    

    Parameters
    ----------
    trk_file : str
        DESCRIPTION.
    DIAMOND_dir : str
        DESCRIPTION.
    K : int, optional
        DESCRIPTION. The default is 2.
    binary : bool, optional
        DESCRIPTION. The default is False.
    sNum : int, optional
        DESCRIPTION. The default is 80.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    # Tract -----------
    
    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    
    # t0 & t1 ---------------
    
    tList=[]
    for k in range(K):
        img=nib.load(DIAMOND_dir+'_diamond_t'+str(k)+'.nii.gz')
        t=img.get_fdata()
        
        # Removes tensor k where frac_k == 0
        if os.path.isfile(DIAMOND_dir+'_diamond_fractions.nii.gz'):
            
            f=nib.load(DIAMOND_dir+'_diamond_fractions.nii.gz').get_fdata()
            
            ft=f[:,:,:,0,k]
            
            t[ft==0,:,:]=[[0,0,0,0,0,0]]
        
        tList.append(DIAMONDpopToMFpeak(t))
        
    # !!! Add filter to remove peaks where frac==0
    
    return tractToFiberPop(trk, tList, binary,sNum)
    
    
def tractToFiberPop(trk,tList:list,binary:bool=False,sNum:int=80):
    '''
    

    Parameters
    ----------
    trk : TYPE
        DESCRIPTION.
    tList : list
        DESCRIPTION.
    binary : bool, optional
        DESCRIPTION. The default is False.
    sNum : int, optional
        DESCRIPTION. The default is 80.

    Returns
    -------
    ROI_select : TYPE
        DESCRIPTION.
    ROI_quantity : TYPE
        DESCRIPTION.
    phi_maps : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    outputVoxelStream : TYPE
        DESCRIPTION.

    '''
    
    phi_maps={}
    K = len(tList)
    #t10=np.zeros(tList[0].shape)
    ROI_select=np.zeros(tList[0].shape[0:3]) 
    ROI_quantity=np.zeros(tList[0].shape[0:3]+(K,))
    
    sList=tractToStreamlines(trk)
    
    outputVoxelStream=[]
    
    for h,streamline in enumerate(sList):
        
        voxelStream={}
        
        #!!!
        previous_point=streamline[0,:]+.5
    
        for i in range(1,streamline.shape[0]):
            
            point=streamline[i,:]
            
            #!!!
            point+=.5    # Is correct I think
            # TODO: compare to edges of spatial domain
                
            # voxList=getVoxelsFromSegment(point,previous_point)
            voxList=compute_subsegments(previous_point,point)
            
            vs=(point-previous_point)   # Tract deltas
            
            for x,y,z in voxList:
            
                x,y,z=(int(x),int(y),int(z))
                
                vList=[]
                nList=[]    # Null list, boolean
                
                for t in tList:
                    
                    v=t[x,y,z,:]
                    vList.append(v)
                    
                    # Fingerprint : null vector = [0,0,0]
                    # Diamond : null vector = [1,0,0]
                    nList.append(all(v == 0 for v in v[1:]))
                
                if all(nList):       # If no tensor in voxel
                    ROI_select[x,y,z]=3   
                    #t10[x,y,z,:]=np.zeros(3)
                    continue          
                
                if (x,y,z) not in phi_maps:       # Never been to this voxel
                    phi_maps[(x,y,z)]=[[],[]]
                    
                    
                aList=[]    # angle list
                for k,v in enumerate(vList):
                    if nList[k]:
                        aList.append(1000)
                    else:
                        aList.append(angleBetweenVectors(vs,v))
                
                min_k=np.argmin(aList)
                phi_maps[(x,y,z)][0].append(aList[min_k])
                
                if binary:
                    
                    ROI_quantity[x,y,z,min_k]+=voxList[(x,y,z)]
                    phi_maps[(x,y,z)][1].append(voxList[(x,y,z)])
                        
                else:
                    
                    coefList=angularWeight(vs,vList,nList)
                    for k,coef in enumerate(coefList):
                        ROI_quantity[x,y,z,k]+=voxList[(x,y,z)]*coef
                    phi_maps[(x,y,z)][1].append(voxList[(x,y,z)]*coefList[min_k])
    
                if h%sNum==0:
                    
                    if (x,y,z) not in voxelStream:
                        voxelStream[(x,y,z)]=[]
                        for k,coef in enumerate(coefList):
                            voxelStream[(x,y,z)].append(voxList[(x,y,z)]*coef)
                    else:
                        for k,coef in enumerate(coefList):
                            voxelStream[(x,y,z)][k]+=voxList[(x,y,z)]*coef
    
    
            previous_point=point
            
        if h%sNum==0:
            
            outputVoxelStream.append(voxelStream)
    
    ROI_select[ROI_quantity[:,:,:,0]>ROI_quantity[:,:,:,1]]=1
    ROI_select[ROI_quantity[:,:,:,0]<ROI_quantity[:,:,:,1]]=2
        
    # img=nib.load(DIAMOND_dir+'_diamond_t0.nii.gz')
    # t=MFpeakToDIAMONDpop(t10)
    # save_nifti(output_dir+'afToTensor.nii.gz', t,img.affine,img.header)
    
    return ROI_select, ROI_quantity, phi_maps, len(sList), outputVoxelStream

def tractToStreamlines(trk)->list:
    '''
    

    Parameters
    ----------
    trk : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    '''
    
    streams=trk.streamlines
    streams_data=trk.streamlines.get_data()
    b = np.float64(streams_data)
    
    sList=[]
    
    for i,offset in enumerate(streams._offsets):
        
        sList.append(b[offset:offset+streams._lengths[i],:])
    
    return sList

def plotStreamlineMetrics(voxelStreams:list,metric_maps:list):
    '''
    

    Parameters
    ----------
    voxelStreams : list
        DESCRIPTION.
    metric_maps : list
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    import matplotlib.pyplot as plt
    
    for i,voxelStream in enumerate(voxelStreams):
            mList=[]
            mcfoList=[]
            vList=[]
            qList0=[]
            qList1=[]
            cfoList0=[]
            cfoList1=[]
            qListT0=[]
            qListT1=[]
            
            for voxel in voxelStream:
                #vList.append(str(voxel))
                vList.append(str((voxel[0],voxel[2])))
                qList0.append(voxelStream[voxel][0])
                qList1.append(voxelStream[voxel][1])
                qListT0.append(voxelStream[voxel][0]/(voxelStream[voxel][0]+voxelStream[voxel][1]))
                qListT1.append(voxelStream[voxel][1]/(voxelStream[voxel][0]+voxelStream[voxel][1]))
                
                if voxelStream[voxel][0]>voxelStream[voxel][1]:
                    cfoList0.append(voxelStream[voxel][0]+voxelStream[voxel][1])
                    cfoList1.append(0)
                    
                    mcfoList.append(metric_maps[0][voxel])
                    
                else:
                    cfoList1.append(voxelStream[voxel][0]+voxelStream[voxel][1])
                    cfoList0.append(0)
                    
                    mcfoList.append(metric_maps[1][voxel])
                
                mList.append((voxelStream[voxel][0]*metric_maps[0][voxel]+
                             voxelStream[voxel][1]*metric_maps[1][voxel])/
                             (voxelStream[voxel][0]+voxelStream[voxel][1]))
                
            fig, axs = plt.subplots(3, 1)
            axs[0].bar(vList,qList0,label='Pop 1')
            axs[0].bar(vList,qList1,bottom=qList0,label='Pop 2')
            axs[0].legend()
            axs[0].set_ylabel('Fixel weight \n (angular weighting)')
            axs[1].bar(vList,cfoList0,label='Pop 1')
            axs[1].bar(vList,cfoList1,bottom=cfoList0,label='Pop 2')
            axs[1].set_ylabel('Fixel weight \n (closest fixel only)')
            axs[2].plot(vList,mList,label='Angular weighting')
            axs[2].plot(vList,mcfoList,label='Closest fixel only')
            if i==4:
                #axs[2].plot(vList,[.5,.48,.48,.46,.44,.42,.42,.4,.38,.38,.38,.38,.36,.34,.34,.32,.32,.32,.3,.3],label='Ground thruth')
                axs[2].plot(vList,[.88,.87,.87,.86,.85,.83,.83,.81,.80,.80,.80,.80,.78,.76,.76,.73,.73,.73,.71,.71],label='Ground thruth')
            #axs[2].set_ylim([0.2,.8])
            axs[2].set_ylim([0.6,1])
            axs[2].legend()
            fig.suptitle(i)
    

def plotStreamlineDetails(voxelStreams:list):
    '''
    

    Parameters
    ----------
    voxelStreams : list
        DESCRIPTION.

    Returns
    -------
    None.

    '''
      
    import matplotlib.pyplot as plt
    
    for i,voxelStream in enumerate(voxelStreams):
            vList=[]
            qList0=[]
            qList1=[]
            qListT0=[]
            qListT1=[]
            
            for voxel in voxelStream:
                vList.append(str(voxel))
                qList0.append(voxelStream[voxel][0])
                qList1.append(voxelStream[voxel][1])
                qListT0.append(voxelStream[voxel][0]/(voxelStream[voxel][0]+voxelStream[voxel][1]))
                qListT1.append(voxelStream[voxel][1]/(voxelStream[voxel][0]+voxelStream[voxel][1]))
                
            fig, axs = plt.subplots(2, 1)
            axs[0].bar(vList,qList0)
            axs[0].bar(vList,qList1,bottom=qList0)
            axs[1].bar(vList,qListT0,label='Pop 1')
            axs[1].bar(vList,qListT1,bottom=qListT0,label='Pop 2')
            axs[1].plot(qListT0,c='black')
            axs[1].legend()
            fig.suptitle(i)
            

def streamlineCount(quantity_map)->int:
    '''
    Parameters
    ----------
    quantity_map :  [x,y,z,k] volume containing the quantity of streamline 
                    belonging to each population k

    Returns
    -------
    sc : int

    '''
    
    sc=np.sum(quantity_map,axis=3)
    
    return sc

def volumetricAgreement(quantity_map):
    '''
    Parameters
    ----------
    quantity_map :  [x,y,z,k] volume containing the quantity of streamline 
                    belonging to each population k

    Returns
    -------
    index : volumetric agreement index
    index_map : voxel-wise volumetric agreement index

    '''
    
    index_max=np.amax(quantity_map, axis=3)
    
    index=np.sum(index_max)/np.sum(quantity_map)
    
    index_map=index_max/np.sum(quantity_map,axis=3)
    
    return index, index_map

def angularAgreement(phi_maps,volume_shape):
    '''
    Parameters
    ----------
    phi_maps : dictionnary of the volumes of phi and the corresponding contribution

    Returns
    -------
    phi : angular agreement index
    phi_map : voxel-wise angular agreement index

    '''
    
    phi_sum=np.zeros(volume_shape[:3])
    phi_map=np.zeros(volume_shape[:3])
    w=np.zeros(volume_shape[:3])
    
    for (x,y,z) in phi_maps:
        lista=np.array(phi_maps[(x,y,z)][0])*np.array(phi_maps[(x,y,z)][1])
        phi_sum[x,y,z]=round(np.sum(lista),3)
        w[x,y,z]=sum(phi_maps[(x,y,z)][1])
    
    phi=np.sum(phi_sum)/np.sum(w)
    
    phi_map=phi_sum/w
    
    return phi, phi_map

def tractToROI(trk_file:str):
    '''
    

    Parameters
    ----------
    trk_file : str
        DESCRIPTION.

    Returns
    -------
    ROI : TYPE
        DESCRIPTION.

    '''
    
    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    
    streams_data=trk.streamlines.get_data()
    
    b = np.float64(streams_data)
    ROI=np.zeros(trk._dimensions) 
    
    for i in range(b.shape[0]):
        
        # !!!
        ROI[(int(b[i,0]+.5),int(b[i,1]+.5),int(b[i,2]+.5))]=1
        
    return ROI