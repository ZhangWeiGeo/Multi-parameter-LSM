#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:11:07 2023

@author: zhangw0c
"""

import numpy as np
import cupy as cp


from math import floor
from numba import cuda
from numba import jit, prange
from NonStationaryConvolve2D_kernel import  _matvec_rmatvec_call as _Convolve2D_kernel
from NonStationaryConvolve3D_kernel import  _matvec_rmatvec_call as _Convolve3D_kernel

#Cop_cuda = ID.NonStationaryConvolve2D(dims=(nx, nz), hs=cp.asarray(psfs).astype(np.float32), ihx=psfx, ihz=psfz, dtype=np.float32, num_threads_per_blocks=(16,32), engine="cuda")
# dims=(nx, nz);hs=cp.asarray(psfs).astype(np.float32);ihx=psfx; ihz=psfz;dtype=np.float32;num_threads_per_blocks=(16,32); engine="cuda";
class NonStationaryConvolve2D():
    def __init__(
    self,
    dims,
    hs,
    ihx,
    ihz,
    engine= "numpy",
    num_threads_per_blocks= (32, 16),
    dtype= "float32",
    name: str = "C"):
        
        #init
        self.hs = hs
        self.hshape = hs.shape[2:]
        self.ohx, self.dhx, self.nhx = ihx[0], ihx[1] - ihx[0], len(ihx) #beg_x, interval_x,number_x
        self.ohz, self.dhz, self.nhz = ihz[0], ihz[1] - ihz[0], len(ihz) #beg_z, interval_z,number_z
        self.ehx, self.ehz = ihx[-1], ihz[-1] #endx,endz
        self.dims = tuple(dims)
        self.engine = engine
    
        # create additional input parameters for engine=cuda
        self.kwargs_cuda = {}
        if engine == "cuda":
            self.kwargs_cuda["num_threads_per_blocks"] = num_threads_per_blocks
            num_threads_per_blocks_x, num_threads_per_blocks_z = num_threads_per_blocks
            num_blocks_x = (
                self.dims[0] + num_threads_per_blocks_x - 1
            ) // num_threads_per_blocks_x
            num_blocks_z = (
                self.dims[1] + num_threads_per_blocks_z - 1
            ) // num_threads_per_blocks_z
            self.kwargs_cuda["num_blocks"] = (num_blocks_x, num_blocks_z)
        
        self._register_multiplications(engine)
        
    def _register_multiplications(self, engine):
        if engine == "numba":
            numba_opts = dict(nopython=True, fastmath=True, nogil=True, parallel=True)
            self._mvrmv = jit(**numba_opts)(self._matvec_rmatvec)
        elif engine == "cuda":
            self._mvrmv = _Convolve2D_kernel
        else:
            self._mvrmv = self._matvec_rmatvec
    
    @staticmethod
    def _matvec_rmatvec(
        x,
        y,
        hs,
        hshape,
        xdims,
        ohx: float,
        ohz: float,
        dhx: float,
        dhz: float,
        nhx: int,
        nhz: int,
        rmatvec: bool = False,
    ):
        for ix in prange(xdims[0]):
            for iz in range(xdims[1]):
                # find closest filters and interpolate h
                ihx_l = int(np.floor((ix - ohx) / dhx)) #id number of left for hs_arr
                ihz_t = int(np.floor((iz - ohz) / dhz)) #id number of top  for hs_arr
                dhx_r = (ix - ohx) / dhx - ihx_l #weight for right psfs, left 1-ihz_t
                dhz_d = (iz - ohz) / dhz - ihz_t #weight for down psfs,  top 1-dhz_d
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
                    ihz_t = ihz_d = 0
                    dhz_t = dhz_d = 0.5
                elif ihz_t >= nhz - 1:
                    ihz_t = ihz_d = nhz - 1
                    dhz_t = dhz_d = 0.5
                else:
                    ihz_d = ihz_t + 1
                    dhz_t = 1.0 - dhz_d

                h_tl = hs[ihx_l, ihz_t]
                h_bl = hs[ihx_l, ihz_d]
                h_tr = hs[ihx_r, ihz_t]
                h_br = hs[ihx_r, ihz_d]

                h = (
                    dhz_t * dhx_l * h_tl
                    + dhz_d * dhx_l * h_bl
                    + dhz_t * dhx_r * h_tr
                    + dhz_d * dhx_r * h_br
                )

                # find extremes of model where to apply h (in case h is going out of model)
                xextremes = (
                    max(0, ix - hshape[0] // 2),
                    min(ix + hshape[0] // 2 + 1, xdims[0]),
                )
                zextremes = (
                    max(0, iz - hshape[1] // 2),
                    min(iz + hshape[1] // 2 + 1, xdims[1]),
                )
                # find extremes of h (in case h is going out of model)
                hxextremes = (
                    max(0, -ix + hshape[0] // 2),
                    min(hshape[0], hshape[0] // 2 + (xdims[0] - ix)),
                )
                hzextremes = (
                    max(0, -iz + hshape[1] // 2),
                    min(hshape[1], hshape[1] // 2 + (xdims[1] - iz)),
                )
                if not rmatvec:
                    y[xextremes[0] : xextremes[1], zextremes[0] : zextremes[1]] += (
                        x[ix, iz]
                        * h[
                            hxextremes[0] : hxextremes[1], hzextremes[0] : hzextremes[1]
                        ]
                    )
                else:
                    y[ix, iz] = np.sum(
                        h[hxextremes[0] : hxextremes[1], hzextremes[0] : hzextremes[1]]
                        * x[xextremes[0] : xextremes[1], zextremes[0] : zextremes[1]]
                    )
        return y 
    

    def _forward(self, x):
        # ncp = get_array_module(x)
        ncp = cp.get_array_module(x)
        y = ncp.zeros(self.dims, dtype=x.dtype)
        y = self._mvrmv(
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
            self.nhz,
            rmatvec=False,
            **self.kwargs_cuda
        )
        return y


    def _adjoint(self, x):
        # ncp = get_array_module(x)
        ncp = cp.get_array_module(x)
        y = ncp.zeros(self.dims, dtype=x.dtype)
        y = self._mvrmv(
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
            self.nhz,
            rmatvec=True,
            **self.kwargs_cuda
        )
        return y



# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # # # # # # # # 
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
######################################
class NonStationaryConvolve3D():
    def __init__(
    self,
    dims,
    hs,
    ihx,
    ihy,
    ihz,
    engine= "numpy",
    dim_block= (2, 16, 16),
    dtype= "float32",
    name: str = "C"):
        
        #init
        self.hs = hs
        self.hshape = hs.shape[3:]
        self.ohx, self.dhx, self.nhx = ihx[0], ihx[1] - ihx[0], len(ihx) 
        self.ohy, self.dhy, self.nhy = ihy[0], ihy[1] - ihy[0], len(ihy) 
        self.ohz, self.dhz, self.nhz = ihz[0], ihz[1] - ihz[0], len(ihz) 
        self.ehx, self.ehx, self.ehz = ihx[-1], ihy[-1], ihz[-1]
        self.dims = tuple(dims)
        self.engine = engine
    
        # create additional input parameters for engine=cuda
        self.kwargs_cuda = {}
        if engine == "cuda":
            self.kwargs_cuda["dim_block"] = dim_block
            
            gridx = (self.dims[0] + dim_block[0] - 1)//dim_block[0]
            gridy = (self.dims[1] + dim_block[1] - 1)//dim_block[1]
            gridz = (self.dims[2] + dim_block[2] - 1)//dim_block[2]
            
            self.kwargs_cuda["dim_grid"] = (gridx, gridy, gridz)
        
        self._register_multiplications(engine)
        
    def _register_multiplications(self, engine):
        if engine == "numba":
            numba_opts = dict(nopython=True, fastmath=True, nogil=True, parallel=True)
            self._mvrmv = jit(**numba_opts)(self._matvec_rmatvec)
        elif engine == "cuda":
            self._mvrmv = _Convolve3D_kernel
        else:
            self._mvrmv = self._matvec_rmatvec
    
    @staticmethod
    def _matvec_rmatvec(
        x,
        y,
        hs,
        hshape,
        xdims,
        ohx: float,
        ohy: float,
        ohz: float,
        dhx: float,
        dhy: float,
        dhz: float,
        nhx: int,
        nhy: int,
        nhz: int,
        rmatvec: bool = False,
    ):
        for ix in prange(xdims[0]):
            for iy in range(xdims[1]):
                for iz in range(xdims[2]):
                    # find closest filters and interpolate h
                    ihx_l = int(np.floor((ix - ohx) / dhx)) #id number of left for hs_arr
                    ihy_b = int(np.floor((iy - ohy) / dhy)) #id number of back for hs_arr  
                    ihz_t = int(np.floor((iz - ohz) / dhz)) #id number of top  for hs_arr
                    
                    dhx_r = (ix - ohx) / dhx - ihx_l #weight for right psfs, left 1-ihz_t
                    dhy_f = (iy - ohy) / dhy - ihy_b #weight for front psfs, left 1-ihz_t
                    dhz_d = (iz - ohz) / dhz - ihz_t #weight for down psfs,  top 1-dhz_d
                    
                    if ihx_l < 0:
                        ihx_l = ihx_r = 0
                        dhx_l = dhx_r = 0.5
                    elif ihx_l >= nhx - 1:
                        ihx_l = ihx_r = nhx - 1
                        dhx_l = dhx_r = 0.5
                    else:
                        ihx_r = ihx_l + 1
                        dhx_l = 1.0 - dhx_r
                        
                    if ihy_b < 0:
                        ihy_b = ihy_f = 0
                        dhy_b = dhy_f = 0.5
                    elif ihy_b >= nhy - 1:
                        ihy_b = ihy_f = nhy - 1
                        dhy_b = dhy_f = 0.5
                    else:
                        ihy_f = ihy_b + 1
                        dhy_b = 1.0 - dhy_f
    
                    if ihz_t < 0:
                        ihz_t = ihz_d = 0
                        dhz_t = dhz_d = 0.5
                    elif ihz_t >= nhz - 1:
                        ihz_t = ihz_d = nhz - 1
                        dhz_t = dhz_d = 0.5
                    else:
                        ihz_d = ihz_t + 1
                        dhz_t = 1.0 - dhz_d
    
                    h_lbt = hs[ihx_l, ihy_b, ihz_t]
                    h_lbd = hs[ihx_l, ihy_b, ihz_d]
                    h_lft = hs[ihx_l, ihy_f, ihz_t]
                    h_lfd = hs[ihx_l, ihy_f, ihz_d]
                    
                    h_rbt = hs[ihx_r, ihy_b, ihz_t]
                    h_rbd = hs[ihx_r, ihy_b, ihz_d]
                    h_rft = hs[ihx_r, ihy_f, ihz_t]
                    h_rfd = hs[ihx_r, ihy_f, ihz_d]
    
                    h = (
                          dhx_l * dhy_b * dhz_t * h_lbt
                        + dhx_l * dhy_b * dhz_d * h_lbd
                        + dhx_l * dhy_f * dhz_t * h_lft
                        + dhx_l * dhy_f * dhz_d * h_lfd
                        + dhx_r * dhy_b * dhz_t * h_rbt
                        + dhx_r * dhy_b * dhz_d * h_rbd
                        + dhx_r * dhy_f * dhz_t * h_rft
                        + dhx_r * dhy_f * dhz_d * h_rfd
                    )
    
                    # find extremes of model where to apply h (in case h is going out of model)
                    xextremes = (
                        max(0, ix - hshape[0] // 2),
                        min(ix + hshape[0] // 2 + 1, xdims[0]),
                    )
                    yextremes = (
                        max(0, iy - hshape[1] // 2),
                        min(iy + hshape[1] // 2 + 1, xdims[1]),
                    )
                    zextremes = (
                        max(0, iz - hshape[2] // 2),
                        min(iz + hshape[2] // 2 + 1, xdims[2]),
                    )
                    
                    
                    # find extremes of h (in case h is going out of model)
                    hxextremes = (
                        max(0, -ix + hshape[0] // 2),
                        min(hshape[0], hshape[0] // 2 + (xdims[0] - ix)),
                    )
                    hyextremes = (
                        max(0, -iy + hshape[1] // 2),
                        min(hshape[1], hshape[1] // 2 + (xdims[1] - iy)),
                    )
                    hzextremes = (
                        max(0, -iz + hshape[2] // 2),
                        min(hshape[2], hshape[2] // 2 + (xdims[2] - iz)),
                    )
                    
                    if not rmatvec:
                        y[ xextremes[0] : xextremes[1], yextremes[0] : yextremes[1], zextremes[0] : zextremes[1] ] += (
                            x[ix, iy, iz]
                            * h[ hxextremes[0] : hxextremes[1], hyextremes[0] : hyextremes[1], hzextremes[0] : hzextremes[1] ]
                        )
                    else:
                        y[ix, iy, iz] = np.sum(
                            h[ hxextremes[0] : hxextremes[1], hyextremes[0] : hyextremes[1], hzextremes[0] : hzextremes[1] ]
                            * x[ xextremes[0] : xextremes[1], yextremes[0] : yextremes[1], zextremes[0] : zextremes[1] ]
                        )
        return y 
    

    def _forward(self, x):
        # ncp = get_array_module(x)
        ncp = cp.get_array_module(x)
        y = ncp.zeros(self.dims, dtype=x.dtype)
        y = self._mvrmv(
            x,
            y,
            self.hs,
            self.hshape,
            self.dims,
            self.ohx,
            self.ohy,
            self.ohz,
            self.dhx,
            self.dhy,
            self.dhz,
            self.nhx,
            self.nhy,
            self.nhz,
            rmatvec=False,
            **self.kwargs_cuda
        )
        return y


    def _adjoint(self, x):
        # ncp = get_array_module(x)
        ncp = cp.get_array_module(x)
        y = ncp.zeros(self.dims, dtype=x.dtype)
        y = self._mvrmv(
            x,
            y,
            self.hs,
            self.hshape,
            self.dims,
            self.ohx,
            self.ohy,
            self.ohz,
            self.dhx,
            self.dhy,
            self.dhz,
            self.nhx,
            self.nhy,
            self.nhz,
            rmatvec=True,
            **self.kwargs_cuda
        )
        return y
