#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 21:54:05 2023

@author: zhangw0c
"""

import os
import sys
sys.path.append("/home/zhangjiwei/pyfunc/lib")

from lib_sys import *



from numba import cuda
import cupy as cp
from numba import jit, prange
from pylops.utils.backend import get_array_module



def obtain_nnx_nny_nnz_from_nxnynz_wxwywz_3D(nx,ny,nz,wx,wy,wz):
    
    w1=wx;
    w2=wy;
    w3=wz;
    half_w1 = np.floor(w1/2);
    half_w2 = np.floor(w2/2);
    half_w3 = np.floor(w3/2);
    if (w1%2)==0 | (w2%2)==0  | (w3%2)==0:
        print("The width/height/y must be odd,w1=%d,w2=%d,w3=%d",w1,w2,w3);
        exit(0);
		
    s_n1 = np.floor((nx-half_w1)/w1);
    s_n2 = np.floor((ny-half_w2)/w2);
    s_n3 = np.floor((nz-half_w3)/w3);
    
    if((nx-half_w1)%w1==0):
        s_n1=s_n1;
    else:
        s_n1=s_n1+1;

    if((ny-half_w2)%w2==0):
        s_n2=s_n2;
    else:
        s_n2=s_n2+1;
			
    if((ny-half_w3)%w3==0):
        s_n3=s_n3;
    else:
        s_n3=s_n3+1;
    
    s_n1=np.int(s_n1+2);
    s_n2=np.int(s_n2+2);
    s_n3=np.int(s_n3+2);
    
    nnx=np.int(s_n1*	w1);
    nny=np.int(s_n2*	w2);
    nnz=np.int(s_n3*	w3);
    
    bbbl=w1;
    bbbf=w2;
    bbbu=w3;
    
    bbbr=nnx-nx-bbbl;
    bbbb=nny-ny-bbbf;
		
    bbbd=nnz-nz-bbbu;
    
    return nnx,nny,nnz,s_n1,s_n2,s_n3,bbbl,bbbf,bbbu,bbbr,bbbb,bbbd

# nnx,nny,nnz,s_n1,s_n2,s_n3,bbbl,bbbf,bbbu,bbbr,bbbb,bbbd=obtain_nnx_nny_nnz_from_nxnynz_wxwywz_3D(500,1,240,23,1,23);

            
def form_the_weight_coeffiecent_3D(nx,ny,nz,output_bool=0):

    coe_h=np.zeros((nx,ny,nz,8));
    for ix in range(0,nx):
        for iy in range(0,ny):
            for iz in range(0,nz):
                
                # id1=iz*nx*ny+iy*nx+ix;

                sx=1.0*ix/nx;
                sy=1.0*iy/ny;
                sz=1.0*iz/nz;

                coe_h[ix,iy,iz,0]=1.0*(1-sx)*(1-sy)*(1-sz);
                coe_h[ix,iy,iz,1]=1.0*  sx  *(1-sy)*(1-sz);
                coe_h[ix,iy,iz,2]=1.0*  sx  *(1-sy)*  sz  ;
                coe_h[ix,iy,iz,3]=1.0*(1-sx)*(1-sy)*  sz  ;
                coe_h[ix,iy,iz,4]=1.0*(1-sx)*  sy  *(1-sz);
                coe_h[ix,iy,iz,5]=1.0*  sx  *  sy  *(1-sz);
                coe_h[ix,iy,iz,6]=1.0*  sx  *  sy  *  sz  ;
                coe_h[ix,iy,iz,7]=1.0*(1-sx)*  sy  *  sz  ;
    
    if output_bool:
        
        PF.mkdir("./psf-eps/");
        for imark in range(0,8):
            tmp=1.0*coe_h[:,0,:,imark];
            tmp=tmp.reshape(nx,nz)
            eps_name='./psf-eps/coe-' + str(imark) + '.eps'
            PF.imshow(tmp.T,nx,nz,1,1,0,nx,10,0,nz,10,'x(grid)','y(grid)',tmp.min(),tmp.max(),1,'gray',(5,5),eps_name);
            
            PF.mkdir("./psf-eps/coe/");
            file_name='./psf-eps/coe/' + str(imark) + '.bin'
            PF.fwrite_file_2d(file_name,tmp.T,nx,nz);
            
    return coe_h

def expand_boundary_2D_zeros(psf_h,nx,nz,wx,wz,nnx,nnz,reverse=0):
    
    if reverse==0:
        wf_nxnynz1_h=1.0*np.zeros((nnx,nnz));
        wf_nxnynz1_h[wx:nx+wx,wz:nz+wz] = 1.0*psf_h[:,:];
    else:
        wf_nxnynz1_h = 1.0*psf_h[wx:nx+wx,wz:nz+wz];
    
    return wf_nxnynz1_h

def expand_boundary_2D_psf(psf_h,nx,nz,wx,wz,nnx,nnz):
    
    wf_nxnynz1_h=np.zeros((nnx,nnz));
    wf_nxnynz1_h[wx:nx+wx,wz:nz+wz] = 1.0*psf_h[:,:];

##up
    wf_nxnynz1_h[:,0:wz] = 1.0*wf_nxnynz1_h[:,wz:2*wz];
##down
    wf_nxnynz1_h[:,nnz-wz:nnz] = 1.0*wf_nxnynz1_h[:,nnz-2*wz:nnz-wz];
##left    
    wf_nxnynz1_h[0:wx,:] = 1.0*wf_nxnynz1_h[wx:2*wx,:];
##right    
    wf_nxnynz1_h[nnx-wx:nnx,:] = 1.0*wf_nxnynz1_h[nnx-2*wx:nnx-wx,:]
    
    return wf_nxnynz1_h

def reshape_2D_psf(mig,n1,n2,wx,wz,output_bool=0):
    
    output=np.zeros((wx,wz,n1,n2))
    
    for ix in range(0,n1):
        for iz in range(0,n2):

            begx=(ix+0)*wx
            endx=(ix+1)*wx
            begz=(iz+0)*wz
            endz=(iz+1)*wz
            output[:,:,ix,iz] = 1.0*mig[begx:endx,begz:endz];
            
            if output_bool:
                tmp=1.0*mig[begx:endx,begz:endz];
                PF.mkdir("./psf-eps/");
                eps_name='./psf-eps/' + str(ix) + '-' + str(iz) + '.eps'
                PF.imshow(tmp.T,wx,wz,1,1,0,wx,10,0,wz,10,'x(grid)','y(grid)',tmp.min(),tmp.max(),1,'gray',(5,5),eps_name);
                
                PF.mkdir("./psf-eps/psf/");
                file_name='./psf-eps/psf/' + str(ix) + '-' + str(iz) + '.bin'
                PF.fwrite_file_2d(file_name,tmp.T,wx,wz);

                nx,nz = mig.shape
                file_name='./psf-eps/psf/' + "psf-" + str(nx) + "-" + str(nz) + '.bin'
                PF.fwrite_file_2d(file_name,mig.T,nx,nz);
    return output 


def cal_angle_ref_func(vel_arr, den_arr, angle_start=0, angle_num=90, dangle=1.0):
    
    (nx,nz) = vel_arr.shape
    angle_ref_arr = np.zeros((angle_num, nx, nz));###no dtype for float 64, it is better for use to use the float 64 for this function. Because there is an error/nan for vel_arr.dtype
   
    for ig in range(0, angle_num):
        for iz in range(0, nz-1):
            
            gama0 	  =   (angle_start + ig*dangle) /180.0*np.pi;
            c0        =   vel_arr[:,iz];
            c1        =   vel_arr[:,iz+1];
            rou0      =   den_arr[:,iz];
            rou1      =   den_arr[:,iz+1];
            
            radian 	  =  c1/c0*np.sin( gama0 );
            
            radian    = np.clip(radian, -1.0, 1.0) ;
            
            gama1 	=  ( 1.0*np.arcsin( radian )  );
            
            tmp0 	= rou1*c1*np.cos(1.0*gama0) - rou0*c0*np.cos(1.0*gama1) ; 

            tmp1 	= rou1*c1*np.cos(1.0*gama0) + rou0*c0*np.cos(1.0*gama1) ;

            final   = (1.0*tmp0 / tmp1).reshape(nx);

            mask = np.isnan(final)
            final[mask] = 0;            #print("mask is nan",mask);

            angle_ref_arr[ig,:,iz] =  final[:];
    
    return angle_ref_arr.astype(np.float32)



def expand_psfs_6D(psfs):
    (nx,ny,nz,wx,wy,wz) = psfs.shape
    output = np.zeros((nx+2, ny+2, nz+2, wx, wy, wz), dtype=psfs.dtype);
    
    output[1:nx+1, 1:ny+1, 1:nz+1, :, :, :] = 1.0*psfs[:, :, :, :, :, :];
    #x direction
    output[0:1, 1:ny+1, 1:nz+1, :, :, :]        = 1.0*psfs[0, :, :, :, :, :];
    output[nx+1:nx+2, 1:ny+1, 1:nz+1, :, :, :]  = 1.0*psfs[nx-1, :, :,  :, :, :];
    
    #y direction
    output[:,  0, :, :, :, :]    = 1.0*output[:,  1, :,  :, :, :]
    output[:, -1, :, :, :, :]    = 1.0*output[:, -2, :, :, :, :].reshape(nx+2, nz+2, wx, wy, wz);
    
    #z direction
    output[:, :,  0, :, :, :]      = 1.0*output[:, :,  1, :, :, :]
    output[:, :, -1, :, :, :]      = 1.0*output[:, :, -2, :, :, :].reshape(nx+2, ny+2, wx, wy, wz);
    
    return output

def find_psf_value_id(psfx, a):
    return_px = 0
    return_ipx   = 0
    for ipx, px in enumerate(psfx):
        if ipx<len(psfx)-1:
            if a>=px and a< psfx[ipx+1]:
                return_ipx   = ipx
                return_px = px
    
    if a>=psfx[-1]:
        return_ipx   = len(psfx)-1
        return_px    = psfx[-1]
    
    return return_ipx, return_px

def get_psf_id(total_psfx, part_psfx, multiple_psf_wx):
    
    psfx  = []
    t_idx = []
    
    for ipx, px in enumerate(part_psfx):
        idx, value = find_psf_value_id(total_psfx, px);
        if px<=total_psfx[-1]:
            psfx.append(value - px + ipx*multiple_psf_wx);
            t_idx.append(idx);
        else:
            psfx.append(  psfx[-1] + multiple_psf_wx);
            t_idx.append(t_idx[-1]);

    return psfx, t_idx

def convert_5D_psfs_to_3D_array(psfs):
    
    shape_list = list(psfs.shape);
    print("shape_list is ", list(shape_list));
    
    angle_num = shape_list[0]
    nx_number = shape_list[1]
    nz_number = shape_list[2]
    wx        = shape_list[3]
    wz        = shape_list[4]
    
    nx = wx *  nx_number
    nz = wz *  nz_number
    
    output = np.zeros((angle_num, nx, nz),dtype=psfs.dtype);
    
    for ix in range(0, nx_number):
        for iz in range(0, nz_number):
            output[:, ix*wx:(ix+1)*wx, iz*wz:(iz+1)*wz] = psfs[:,ix,iz,:,:].reshape(angle_num, wx, wz);
            
    return output

def convert_4D_psfs_to_2D_array(psfs):
    
    shape_list = list(psfs.shape);
    print("shape_list is ", list(shape_list));
    
    nx_number = shape_list[0]
    nz_number = shape_list[1]
    wx        = shape_list[2]
    wz        = shape_list[3]
    
    nx = wx *  nx_number
    nz = wz *  nz_number
    
    output = np.zeros((nx, nz),dtype=psfs.dtype);
    
    for ix in range(0, nx_number):
        for iz in range(0, nz_number):
            output[ix*wx:(ix+1)*wx, iz*wz:(iz+1)*wz] = psfs[ix,iz,:,:].reshape(wx, wz);
            
    return output

def compute_illumination_for_psfs_2D(mig_ones, psfx, psfz, illumination_wx, illumination_wz):
    (nx, nz) = mig_ones.shape
    output = np.zeros((len(psfx),len(psfz)),dtype=mig_ones.dtype);

    for ipx, px in enumerate(psfx):
        for ipz, pz in enumerate(psfz):
            
            begx = px   - illumination_wx
            if begx < 0:
                begx=0; 
                endx = begx + 2*illumination_wx
            else:
                endx = begx + illumination_wx
            if endx >nx-1:
                endx=nx-1;
                begx=endx-2*illumination_wx


            begz = pz   - illumination_wz
            if begz < 0:
                begz=0;
                endz = begz + 2*illumination_wz
            else:
                endz = begz + illumination_wz   
            if endz >nz-1:
                endz=nz-1;
                begz=endz-2*illumination_wz
            
            arr = mig_ones[begx:endx, begz:endz];
            
            mean_v = np.mean(arr);
            
            output[ipx,ipz] = 1.0*mean_v;
        
    return output

def generate_mul_reg_para(multi_para, inv_angle_start, inv_angle_end, dangle):
    
    num_max = int( max(abs(inv_angle_start), abs(inv_angle_end)) //dangle);
    
    num_length = int( (inv_angle_end - inv_angle_start) //dangle);
    
    if multi_para!=1:
        step    = 1.0 * (multi_para-1.0) / num_max ; 
        part1   = np.arange(1, 1 + num_max * step,     step)
    else:
        part1   = np.ones(num_max);
    
    total_2num   = np.concatenate((np.flip(part1), part1), axis=0);
    ###the length of total_2num is 2*num_max
    
    if abs(inv_angle_start) >= abs(inv_angle_end):
        multi_para_arr = total_2num[0:num_length]
    else:
        start = int( (inv_angle_start-0) //dangle);
        multi_para_arr = total_2num[num_max + start: 2*num_max]
    
    return multi_para_arr.astype( np.float32 )



class interpolate_4D_psfs():
    
    def __init__(self, dims, hs, ihx, ihz, dtype, dhx=0, dhz=0):
        
        self.hs = hs
        self.hshape = hs.shape[2:]
        self.ohx, self.dhx, self.nhx = ihx[0], ihx[1] - ihx[0], len(ihx)
        self.ohz, self.dhz, self.nhz = ihz[0], ihz[1] - ihz[0], len(ihz)
        self.ehx, self.ehz = ihx[-1], ihz[-1]
        self.dims = dims
        self.dtype = dtype
        
        ''' I will introduce these parameters for block of hessian
            see the svd of hessian
            since gridx  may be [0,0,0,0,0,]
            since gridx  may be [nx,nx,nx,nx,nx]
        '''
        if dhx!=0:
            self.dhx = dhx;
        if dhz!=0:
            self.dhz = dhz;
    
    @staticmethod
    def _matvec_rmatvec(x, y, hs, hshape, xdims, ohx, ohz, dhx, dhz, nhx, nhz):
        for ix in prange(hshape[0]//2, hshape[0]//2+xdims[0]):
            for iz in range(hshape[1]//2, hshape[1]//2+xdims[1]):
                # find closest filters and interpolate h
                ihx_l = int(np.floor((ix - ohx) / dhx)) if dhx != 0 else 0
                ihz_t = int(np.floor((iz - ohz) / dhz)) if dhz != 0 else 0
                dhx_r = (ix - ohx) / dhx - ihx_l if dhx != 0 else 0
                dhz_b = (iz - ohz) / dhz - ihz_t if dhz != 0 else 0

                if ihx_l < 0:
                    ihx_l = ihx_r = 0
                    dhx_l = dhx_r = 0.5
                elif ihx_l >= nhx - 1:
                    ihx_l = ihx_r = nhx - 1
                    dhx_l = dhx_r = 0.5
                else:
                    ihx_r = ihx_l + 1
                    dhx_l = 1.0 - dhx_r

                if ihz_t < 0:
                    ihz_t = ihz_b = 0
                    dhz_t = dhz_b = 0.5
                elif ihz_t >= nhz - 1:
                    ihz_t = ihz_b = nhz - 1
                    dhz_t = dhz_b = 0.5
                else:
                    ihz_b = ihz_t + 1
                    dhz_t = 1.0 - dhz_b

                h_tl = hs[ihx_l, ihz_t]
                h_bl = hs[ihx_l, ihz_b]
                h_tr = hs[ihx_r, ihz_t]
                h_br = hs[ihx_r, ihz_b]

                h = (
                    dhz_t * dhx_l * h_tl
                    + dhz_b * dhx_l * h_bl
                    + dhz_t * dhx_r * h_tr
                    + dhz_b * dhx_r * h_br
                )
                
                y[ix-hshape[0]//2, iz-hshape[1]//2, :, :] = h[:, :];
                
        return y
    
    def _matvec(self, x):
        ncp = get_array_module(x)
        y = ncp.zeros([self.dims[0], self.dims[1], self.hshape[0], self.hshape[1]], dtype=self.dtype)
        
        y = self._matvec_rmatvec(
            x,
            y,
            self.hs,
            self.hshape,
            self.dims,
            self.ohx,
            self.ohz,
            self.dhx,
            self.dhz,
            self.nhx,
            self.nhz
        )
        return y
    
            
        
def convert_4D_psfs_to_2D_hessian(psfs_inter):
    
    [nx, nz, wx, wz] = psfs_inter.shape

    hessian_matrix = np.zeros((nx*nz, nx*nz), dtype=np.float32)
    
    for ix in prange(0, nx):
        for iz in prange(0, nz):
            
            # psf_padding   = np.zeros((nx, nz), dtype=np.float32) 
            
            # psf           = psfs_inter[ix, iz, :, :];
            
            # psf           = np.reshape(psf,[wx,wz]);
            
            # begx          = ix - wx//2;
            # endx          = begx + wx;
            # if begx < 0:
            #     begx      = 0;
            # if endx > nx:
            #     endx      = nx
            
            # begz          = iz - wz//2;
            # endz          = begz + wz;
            # if begz < 0:
            #     begz      = 0;
            # if endz > nz:
            #     endz      = nz;

            # print("begx=", begx); print("endx=", endx);
            # print("endx=", endx); print("endz=", endz);
            # # psf2           = ;
            
            # psf_padding[begx:endx, begz:endz]   = psf[begx:endx, begz:endz];
            
            # psf_padding   =  np.reshape(psf_padding,[nx*nz,1]);
            
            # hessian_matrix[:, ix*nz+iz] =  psf_padding[:,0];           
            
            
            
            psf_padding   = np.zeros((nx*nz, 1), dtype=np.float32) 
            
            # id1 = ix * nz + iz;
            
            for i in prange(0, wx):
                for j in prange(0, wz):
                    
                    mm =  (ix - wx//2 + i) * nz +  iz - wz//2 + j;
                    
                    if mm>0 and mm<nx*nz:
                        psf_padding[mm,:] = psfs_inter[ix, iz, i, j];
            hessian_matrix[:, ix*nz+iz] =  psf_padding[:,0];
            # hessian_matrix[ix*nz+iz, :] =  np.reshape(psf_padding, (1,nx*nz));           

    return  hessian_matrix                  

@cuda.jit(max_registers=40)  
def cuda_reshape_wxwz_into_nxnz_kernel(psfs, psf_padding, wx, wz, nx, nz, ix):
    
    i, j, iz = cuda.grid(3)

    if i < wx and j < wz and iz < nz:
        
        mm =  (ix - wx//2 + i) * nz +  iz - wz//2 + j;
    
        if mm>0 and mm<nx*nz:
            psf_padding[mm, iz] = psfs[iz, i, j];
            

def convert_4D_psfs_to_2D_hessian_cupy(psfs_inter):
    '''
    default, the input 4D psf :  [nx, nz, wx, wz]
    if transpose=True:
    the input should be [nz, nx, wz, wx]
    
    '''
    module = WR.get_module_type(psfs_inter);
    
    if module != "cupy":
        psfs_inter = TF.array_to_cupy(psfs_inter);
    
    psfs_inter     = TF.array_to_float32_contiguous(psfs_inter)
    
    [nx, nz, wx, wz] = psfs_inter.shape

    hessian_matrix = cp.zeros((nx*nz, nx*nz), dtype=cp.float32)
    
    
    dims_block =(32, 16, 1)
    dim_grid   = (wx//dims_block[0]+1, wz//dims_block[1]+1, nz)
    
    for ix in range(0, nx):
        # for iz in range(0, nz):        
            # for i in range(0, wx):
            #     for j in range(0, wz):
            psfs          = cp.asarray( psfs_inter[ix, :, :, :] );
            psf_padding   = cp.zeros((nx*nz, nz), dtype=cp.float32); 
            
            cuda_reshape_wxwz_into_nxnz_kernel[dim_grid, dims_block](psfs, psf_padding, wx, wz, nx, nz, ix);

            hessian_matrix[:, ix*nz : (ix+1)*nz] =  1.0*psf_padding;

    return  hessian_matrix


def convert_4D_psfs_to_2D_hessian_np(psfs_inter):
    '''
    default, the input 4D psf :  [nx, nz, wx, wz]
    if transpose=True:
    the input should be [nz, nx, wz, wx]
    
    '''
    
    psfs_inter     = TF.array_to_float32_contiguous(psfs_inter)
    
    [nx, nz, wx, wz] = psfs_inter.shape

    hessian_matrix = np.zeros((nx*nz, nx*nz), dtype=np.float32)
    

    for ix in range(0, nx):
        for iz in range(0, nz):        
            
            column_slice  = slice(ix*nz + iz , ix*nz + iz +1);
            
            psf_padding   = np.zeros((nx*nz, 1), dtype=np.float32);
            
            for i in range(0, wx):
                for j in range(0, wz):
                    
                    mm =  (ix - wx//2 + i) * nz +  iz - wz//2 + j;
                    
                    if mm>0 and mm<nx*nz:
                        psf_padding[mm, 0]          = psfs_inter[ix, iz, i, j];
            
            hessian_matrix[:, column_slice] =  1.0*psf_padding;

    return  hessian_matrix


def convert_4D_psfs_to_2D_hessian_np_tranpose(psfs_inter):
    '''
    default, the input 4D psf :  [nx, nz, wx, wz]
    if transpose=True:
    the input should be [nz, nx, wz, wx]
    
    '''

    psfs_inter     = TF.array_to_float32_contiguous(psfs_inter)
    
    [nz, nx, wz, wx] = psfs_inter.shape

    hessian_matrix = np.zeros((nx*nz, nx*nz), dtype=np.float32)


    for iz in range(0, nz):
        for ix in range(0, nx):
            
            column_slice  = slice(iz*nx + ix, iz*nx + ix +1);
            
            psf_padding   = np.zeros((nx*nz, 1), dtype=np.float32);
            
            for j in range(0, wz):
                for i in range(0, wx):
                    
                    mm =  (iz - wz//2 + j) * nx +  ix - wx//2 + i;
                    
                    if mm>0 and mm<nx*nz:
                        psf_padding[mm, 0]          = psfs_inter[iz, ix, j, i];
            
            hessian_matrix[:, column_slice] =  1.0*psf_padding;

    return  hessian_matrix

def compute_illumination_for_psfs_2D(mig_ones, psfx, psfz, illumination_wx, illumination_wz):
    (nx, nz) = mig_ones.shape
    output = np.zeros((len(psfx),len(psfz)),dtype=mig_ones.dtype);

    for ipx, px in enumerate(psfx):
        for ipz, pz in enumerate(psfz):
            
            begx = px   - illumination_wx
            if begx < 0:
                begx=0; 
                endx = begx + 2*illumination_wx
            else:
                endx = begx + illumination_wx
            if endx >nx-1:
                endx=nx-1;
                begx=endx-2*illumination_wx


            begz = pz   - illumination_wz
            if begz < 0:
                begz=0;
                endz = begz + 2*illumination_wz
            else:
                endz = begz + illumination_wz   
            if endz >nz-1:
                endz=nz-1;
                begz=endz-2*illumination_wz
            
            arr = mig_ones[begx:endx, begz:endz];
            
            mean_v = np.mean(arr);
            
            output[ipx,ipz] = 1.0*mean_v;
        
    return output



##pacthing

def generate_tapering_2D_matrix(shape, overlap):
    '''
    generate a 
    生成一个线性衰减(1-0)的矩阵，矩阵内部为1，边界长度为overlap
    -----------------------------------
    -----------------------------------
    return weight_matrix
    '''
    rows, cols = shape
    
    nnx = rows
    nnz = cols
    
    weight_matrix = np.ones((nnx, nnz), dtype=np.float32)

    # Define the overlapping regions
    taper1  = np.linspace(0, 1, overlap)
    taper2  = np.linspace(1, 0, overlap)


    # Apply the weights for the left and right overlaps
    for i in range(overlap-1):
        weight_matrix[:, i]    = taper1[i]
        weight_matrix[:, -i-1] = taper2[overlap-i-1]
    
    for i in range(overlap-1):
        weight_matrix[i, :]    *= taper1[i]
        weight_matrix[-i-1,:]  *= taper2[overlap-i-1]

    return weight_matrix


def trans_2D_to_4D_block_info(input_2D, windows, tapers):
    '''
    给定nx nz，计算一共有多少个带有taper_overlap的窗口，以及上下左右需要补多长的边界，默认上和左边补长度为taper_overlap
    默认  tapers[0] = tapers[1]
    -----------------------------------
    -----------------------------------
    return [nnx, nnz, nx, nz, up, down, left, right], [num1, num2]
    '''
    [nx, nz] = input_2D.shape ;
    ## set the figures!!!!!!,
    ##step1,  how many windows   (windows[0]-tapers[0])
    numberx  = np.int(np.round( ( nx + 2*tapers[0]) / (windows[0]-tapers[0]) + 0.5 )  )
    numberz  = np.int(np.round( ( nz + 2*tapers[1]) / (windows[1]-tapers[1]) + 0.5 )  )  
    
    ##step2, compute nnx and nzz length, according to my jpg figure in paper, we must add tapers[0]
    nnx       = numberx * (windows[0]-tapers[0]) + 1*tapers[0]
    nnz       = numberz * (windows[1]-tapers[1]) + 1*tapers[1]
    
    ####step3, length: up, down, left, right
    left      = tapers[0]
    right     = nnx - nx - left
    up        = tapers[1]
    down      = nnz - nz - up
    
    ####step4,
    num1 = numberx
    num2 = numberz
    
    return [nnx, nnz, nx, nz, up, down, left, right], [num1, num2]

def trans_2D_to_4D_block_extends(input_2D, extends_info, forward=True, mode='edge'):
    '''
    forward=True 扩展边界  
    forward=False 去掉边界
    mode='edge' or mode='constant'
    ------------------------
    return 返回 扩展边界的数组 或者 去掉边界的数组
    '''
    if forward:
        [nnx, nnz, nx, nz, up, down, left, right] = extends_info
        if mode=='edge':
            extended_array = np.pad(input_2D, ((left, right), (up, down)), mode='edge')
        else:
            extended_array = np.pad(input_2D, ((left, right), (up, down)),  mode='constant', constant_values=0);
    
    else:
        [nnx, nnz, nx, nz, up, down, left, right] = extends_info
        extended_array = np.zeros([nx, nz], dtype=input_2D.dtype);
        extended_array[:,:] = input_2D[left:left+nx, up:up+nz];
        
    return extended_array    


def trans_2D_to_4D_block_func(input_2D, output_4D, windows, tapers, nums, tapering_2D=0, forward=True):
    '''
    input_2D:   2d arrary,
    output_4D:  4d arrary,
    windows:    window size with boundary tappering,
    tapers:     tapering length,
    steps:      moved step,
    dims:       how many steps,
    '''
    
    if forward:
        output_4D           = np.zeros((nums[0], nums[1], windows[0], windows[1]), dtype=np.float32);
    else:
        output_2D           = np.zeros_like(input_2D)
        
    for ix in range(0, nums[0]):
        for iz in range(0, nums[1]):
            
            x_beg = ix * (windows[0]-tapers[0])
                
            x_end = x_beg + windows[0]
            
            z_beg = iz * (windows[1]-tapers[1])
                
            z_end = z_beg + windows[1]
            
            print("ix",ix); print("iz",iz)
            print("nums[0]",nums[0]); print("nums[1]",nums[1])
            print("x_beg",x_beg); print("x_end",x_end)
            print("z_beg",z_beg); print("z_end",z_end)
            
            if forward:
                output_4D[ix, iz, :, :] = input_2D[x_beg:x_end, z_beg:z_end];
                
        
            
            else:
                output_2D[x_beg:x_end, z_beg:z_end]+= output_4D[ix, iz, :, :] * tapering_2D;
                # output_2D[x_beg:x_end, z_beg:z_end]+= tapering_2D;
            
    if forward:
        return output_4D
    else:
        return output_2D


def trans_4D_psfs_to_6D_psfs(psfs, psfx, psfz, shape_4D, extends_info, windows, tapers, extend_pos=1):
    '''
    psfs_6D = C.trans_4D_psfs_to_6D_psfs(psfs, psfx, psfz, mig_arr_4D.shape, extends_info, windows, tapers);
    reference trans_2D_to_4D_block_func,  since there is a overlap shift
    
    extend_pos: where we want to expand the position psfs, we remend that set 1, since, psfs[0] may contains many zeros!!!!
    '''
    [nnx, nnz, nx, nz, up, down, left, right] = extends_info
    
    hx = psfx[1] - psfx[0]
    hz = psfz[1] - psfz[0]
    
    
    ###vertical extend
    [psf_numx, psf_numz, wx, wz] = psfs.shape
    
    up_num     = np.int(np.round( ((up    / hz + 0.5)))); 
    up_arr     = np.zeros((psf_numx, up_num, wx, wz), dtype=psfs.dtype);
    
    down_num   = np.int(np.round( ((down  / hz + 0.5)))); 
    down_arr   = np.zeros((psf_numx, down_num, wx, wz), dtype=psfs.dtype);
   
    for iz in range(0, up_num):
        up_arr[:, iz, :, :]   = 1.0 * psfs[:, extend_pos, :, :]
    for iz in range(0, down_num):
        down_arr[:, iz, :, :] = 1.0 * psfs[:, -1*extend_pos, :, :]
    
    print("up_arr.min() is{}", up_arr.min())
    print("down_arr.min() is{}", down_arr.min())
    
    psfs1 = np.concatenate((up_arr, psfs, down_arr), axis=1) ; 
    
    print("psfs1.shape is{}", psfs1.shape)
    
    
    ###x extend
    [psf_numx, psf_numz, wx, wz] = psfs1.shape
        
    left_num   = np.int(np.round( ((left / hx+ 0.5)))); 
    left_arr  = np.zeros((left_num, psf_numz, wx, wz), dtype=psfs.dtype);
    
    right_num  = np.int(np.round( ((right / hx+ 0.5)))); 
    right_arr = np.zeros((right_num, psf_numz, wx, wz), dtype=psfs.dtype);
    
    for ix in range(0, left_num):
        left_arr[ix, :, :, :]  =  psfs1[extend_pos, :, :, :]
    for ix in range(0, right_num):
        right_arr[ix, :, :, :] =  psfs1[-1*extend_pos, :, :, :]
    
    
    
    psfs2 = np.concatenate((left_arr, psfs1, right_arr), axis=0) ; 
    
    print("psfs2.shape is{}", psfs2.shape);
    # print("psfs2[-1,-1,:,:].min() is{}", psfs2[-1,-1,:,:].min())
    
    #########reshape 4D to 6D for svd, psf2 to psf3
    [psf_numx, psf_numz, wx, wz] = psfs2.shape
    
    # reshape_wx   = np.int(np.round((windows[0]-tapers[0]) // hx))
    # reshape_wz   = np.int(np.round((windows[1]-tapers[1]) // hz))
    # reshape_nx   = np.int(np.round(psf_numx // reshape_wx))
    # reshape_nz   = np.int(np.round(psf_numz // reshape_wz))
    
    reshape_nx   = shape_4D[0]
    reshape_nz   = shape_4D[1] 
    reshape_wx   = np.int(np.round((windows[0]) // hx))
    reshape_wz   = np.int(np.round((windows[1]) // hz))
    
    step_wx      = np.int(np.round((windows[0]-tapers[0]) // hx))
    step_wz      = np.int(np.round((windows[0]-tapers[0]) // hz))
    
    psfs3 = np.zeros((reshape_nx, reshape_nz, reshape_wx, reshape_wz, wx, wz), dtype=psfs.dtype) ;
    for ix in range(0, reshape_nx):
        for iz in range(0, reshape_nz):
            # for iwx in range(0, reshape_wx):
            #     for iwz in range(0, reshape_wz):
            idx_beg = ix * step_wx
            idx_end = idx_beg + reshape_wx
            
            idz_beg = iz * step_wz
            idz_end = idz_beg + reshape_wz
                    
            psfs3[ix, iz, :, :, :, :] = psfs2[idx_beg:idx_end, idz_beg:idz_end, :, :];
                    
            # x_beg = ix * (windows[0]-tapers[0])
                        
            # x_end = x_beg + windows[0]
                    
            # z_beg = iz * (windows[1]-tapers[1])
                        
            # z_end = z_beg + windows[1]
    
    print("psfs3.shape is{}", psfs3.shape);
    
    return psfs3


#############################
#############################
#############################
#############################
#############################
#############################
#############################
def coord_normalize(arr, min_val=None, max_val=None):
    # 如果没有提供 min_val 和 max_val，则使用输入数组的最小值和最大值
    if min_val is None:
        min_val = np.min(arr)
    if max_val is None:
        max_val = np.max(arr)
    
    # 如果最大值和最小值相等，直接返回一个全零数组以避免除以0
    if max_val == min_val:
        return np.zeros_like(arr)
    
    # 归一化到 [-1, 1]
    normalized_arr = 2 * (arr - min_val) / (max_val - min_val) - 1
    return normalized_arr

def coord_denormalize(normalized_arr, min_val, max_val):
    """
    将 [-1, 1] 区间的归一化数组恢复为原始范围。
    
    参数：
    normalized_arr -- 归一化到 [-1, 1] 的数组
    min_val -- 原始数据的最小值
    max_val -- 原始数据的最大值
    
    返回：
    恢复后的原始数据数组
    """
    # 如果最大值和最小值相等，返回一个全零数组，因为数据无法被反归一化
    if max_val == min_val:
        return np.zeros_like(normalized_arr)
    
    # 反向归一化操作
    original_arr = (normalized_arr + 1) / 2 * (max_val - min_val) + min_val
    
    return original_arr

def coord_normalize_and_denormalize_test():
    """
    测试 coord_normalize 和 coord_denormalize 函数的正确性
    """
    # 生成一个测试数组
    test_arr = np.array([0, 10, 20, 30, 40])
    
    min_val = np.min(test_arr)
    max_val = np.max(test_arr)
    
    # 进行归一化操作
    normalized_arr = coord_normalize(test_arr, min_val, max_val)
    
    # 提取原始数组的 min 和 max
    
    
    # 进行反归一化操作
    denormalized_arr = coord_denormalize(normalized_arr, min_val, max_val)
    
    # 打印结果
    print("原始数组：", test_arr)
    print("归一化后的数组：", normalized_arr)
    print("反归一化后的数组：", denormalized_arr)
    
    # 检查反归一化后的数组是否与原始数组接近
    assert np.allclose(test_arr, denormalized_arr), "测试失败：反归一化结果与原始数组不一致"
    
    print("测试通过：反归一化后的数组与原始数组一致")

# coord_normalize_and_denormalize_test()
#############################
#############################
#############################
#############################
#############################
#############################
#############################
def agc_factor_compute(data, window_size=[1, 1, 1], power=2, smooth_type=1, smooth_sigma=[2, 2, 2], radius=[10, 10, 10]):
    """
    
    """
    
    smooth_data   = np.abs(data) ** power
    if smooth_type==1:
        agc_factor = uniform_filter(smooth_data, size=window_size, mode='nearest')
    else:
        agc_factor = sci_gaussian_filter(smooth_data, sigma=smooth_sigma, radius=window_size)


    agc_factor = np.power(agc_factor, 1.0/power)
        
    agc_factor[agc_factor == 0] = 1.0

    return agc_factor

def non_batch_mean_and_var(data):

    module = WR.get_module_type(data)    

    mean = module.mean(data).item()
    var  = module.var(data).item()

    return mean, var

def non_batch_normalize(data, mean, var, epsilon=1e-5):

    module = WR.get_module_type(data)    

    bn_output = (data - mean) / module.sqrt(var + epsilon)
    
    return bn_output

def non_batch_normalize_reverse(bn_output, mean, var, epsilon=1e-5):
    module = WR.get_module_type(bn_output)
    original_data = bn_output * module.sqrt(var + epsilon) + mean
    return original_data
