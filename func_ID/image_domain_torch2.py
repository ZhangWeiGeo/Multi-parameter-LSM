#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 20:07:51 2023

@author: zhangw0c
"""





import torch
import numpy as np


def NonStationaryConvolve2D_torch(x,psfs,coe,nx,nz,nnx,nnz,wx,wz,f_b=0):

    size1,size2,_,_ = x.shape
    
    # output=torch.zeros((size1,size2,nnx,nnz),requires_grad=True)
    output=torch.zeros((size1,size2,nnx,nnz),device=(x.device))

    for i1 in range(0,size1):
        for i2 in range(0,size2):
            
            ref=x[i1,i2,:,:];
            ref=ref.view(nnx,nnz);
            
            for ix in range(0,nx):
                for iz in range(0,nz):
                    # print("ix:",ix);print("iz:",iz);
 
#the PSF is bigger than migrtion image, up:wz,left:wx,back:wy
                    sx=ix+wx
                    # sy=iy+wy
                    sz=iz+wz
 
#find which 4/8 coe for interpolation //obtain interplolation coefficent
                    coe_idx=(sx-wx//2) % wx;
                    # coe_idy=(sy-wy//2) % wy;
                    coe_idz=(sz-wz//2) % wz;
 
                    coe1=coe[coe_idx,coe_idz,0];
                    coe2=coe[coe_idx,coe_idz,1];
                    coe3=coe[coe_idx,coe_idz,2];
                    coe4=coe[coe_idx,coe_idz,3];

# find which 4/8 psfs for interpolation
                    idx=np.int(np.floor( (sx-wx//2) //wx));
                    # idy=np.int(np.floor( (sy-wy//2) //wy));
                    idz=np.int(np.floor( (sz-wz//2) //wz));
 
  
                    input1 = psfs[:,:,idx,idz]
                    input2 = psfs[:,:,idx+1,idz]
                    input3 = psfs[:,:,idx+1,idz+1]
                    input4 = psfs[:,:,idx,idz+1]
 
 
 
                    interp_psf = coe1*input1 + coe2*input2 + coe3*input3 + coe4*input4;
                    
                    #forward
                    # in_idx=iz*wx + ix;
        			# psf_idx = (sz-wz/2+iz)*nnx + ix+sx-wx/2;
        			# scale_idx = sz*nnx + sx;
        			# output1_d[psf_idx] = output1_d[psf_idx] + 1.0*psf1_wxwywz_d[in_idx] * input2_d[scale_idx];
                    
                    psf_idx_beg = sx - wx//2;
                    psf_idx_end = sx - wx//2 + wx;
                    psf_idz_beg = sz - wz//2;
                    psf_idz_end = sz - wz//2 + wz;
 
                    if f_b==0:
                        scale_psf = 1.0*ref[sx,sz]*interp_psf;
                        output[i1,i2,psf_idx_beg:psf_idx_end,psf_idz_beg:psf_idz_end] +=  scale_psf;
                    else:
                        # adjoint
                        # cuda_get_wxwywz_from_nnxnnynnz_2D
                        # cal_mul_a_b_to_c
                        # cuda_sum_along_both
                        # cuda_set_value_to_nnxnnynnz
                        ref_arr = ref[psf_idx_beg:psf_idx_end,psf_idz_beg:psf_idz_end]
                        sum_arr = torch.mul(ref_arr,interp_psf)
                        sum_value = torch.sum(sum_arr);
                        output[i1,i2,sx,sz] = sum_value;

                    # print("output.grad_fn:",output.grad_fn);
    return output