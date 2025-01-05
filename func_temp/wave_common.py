
import sys
# sys.path.append("/home/zhangjiwei/pyfunc/lib")

from lib_sys import *

# from lib_cuda import *

# from lib_torch import *


# from wave_libary1 import *

# wave_kernel_dict=wave_kernel_dict_func(log_file='wave_kernel_dict.txt');

# import pycuda.driver as cuda
# import pycuda.autoinit
# from pycuda.compiler import Sourcemodule



def combine_x_y(operator, x, y):
    
    y[...] = operator(x);
    
def combine_x_y1y2y3(operator, x, y1, y2, y3):
    
    y1[...], y2[...], y3[...] = operator(x);
    
def combine_x_y1y2(operator, x, y1, y2):
    
    y1[...], y2[...] = operator(x);
    
def allocate_N_Narray(shape_list=[200, 100], module=cp, N=1):
    '''
    ##Note this is a ","
    r_ex_amp_d, = allocate_N_Narray(NxNyNz_list, cp, N=1 );  
    
    A1, A2 = allocate_N_Narray(NxNyNz_list, cp, N=2 );
    '''
    arr_list = []
    
    for inumber in range(0, N):
        p = module.zeros(shape_list, dtype=module.float32);
        arr_list.append(p)

    return arr_list

############################################ 
############################################ all function use define as key words
def define_cupy_derivative_F_space_1D(FD2d_dict, nnx_radius, dx, grid, block, coe_2_d, coe_1_d, radius):
    '''
    Note that all  division is replaced with the multiply!!
    
    
    nnx_radius, dx, coe_2_d, coe_1_d, radius should be int32 or float32.
    '''
    
    ### FD method for derivatives
    
    
    
    dxx_1D      = lambda x, y: FD2d_dict['dxx_1D']( grid, block, (x, y, coe_2_d, cp.int32(nnx_radius), cp.float32(1.0/dx/dx), cp.int32(radius) ), );
    


    dx_1D_forw  = lambda x, y: FD2d_dict['dx_1D_forw']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.float32(1.0/dx), cp.int32(radius) ), );
    
    
    
    
    dx_1D_back  = lambda x, y: FD2d_dict['dx_1D_back']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.float32(1.0/dx), cp.int32(radius) ), );
    



    operator_list = {}
    local_vars    = locals()
    for var_name, var_value in local_vars.items():
       if callable(var_value):
           operator_list[var_name] = var_value
           
    return operator_list


def define_cupy_derivative_F_space_2D(FD2d_dict, nnx_radius, nnz_radius, dx, dz, grid, block, coe_2_d, coe_1_d, radius):
    '''
    Note that all  division is replaced with the multiply!!
    
    
    nnx_radius, nny_radius, nnz_radius, dx, dy, dz, coe_2_d, coe_1_d, radius should be int32 or float32.
    '''
    
    ### FD method for derivatives
    lap_2D      = lambda x, y: FD2d_dict['lap_2D']( grid, block, (x, y, coe_2_d, cp.int32(nnx_radius), cp.int32(nnz_radius), cp.float32(1.0/dx/dx), cp.float32(1.0/dz/dz), cp.int32(radius) ), );
    
    
    dxx_2D      = lambda x, y: FD2d_dict['dxx_2D']( grid, block, (x, y, coe_2_d, cp.int32(nnx_radius), cp.int32(nnz_radius), cp.float32(1.0/dx/dx), cp.int32(radius) ), );
    
    
    
    dzz_2D      = lambda x, y: FD2d_dict['dzz_2D']( grid, block, (x, y, coe_2_d, cp.int32(nnx_radius), cp.int32(nnz_radius), cp.float32(1.0/dz/dz), cp.int32(radius) ), );
    
    
    
    
    
    div_2D_forw = lambda x, y: FD2d_dict['div_2D_forw']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nnz_radius), cp.float32(1.0/dx), cp.float32(1.0/dz), cp.int32(radius) ), );
    div_2D_back = lambda x, y: FD2d_dict['div_2D_back']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nnz_radius), cp.float32(1.0/dx), cp.float32(1.0/dz), cp.int32(radius) ), );
    
    
    
    
    grad_2D_forw = lambda x, y1, y2: FD2d_dict['grad_2D_forw']( grid, block, (x, y1, y2, coe_1_d, cp.int32(nnx_radius), cp.int32(nnz_radius), cp.float32(1.0/dx), cp.float32(1.0/dz), cp.int32(radius) ), );
    grad_2D_back = lambda x, y1, y2: FD2d_dict['grad_2D_back']( grid, block, (x, y1, y2, coe_1_d, cp.int32(nnx_radius), cp.int32(nnz_radius), cp.float32(1.0/dx), cp.float32(1.0/dz), cp.int32(radius) ), );
    
    
    

    dx_2D_forw  = lambda x, y: FD2d_dict['dx_2D_forw']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nnz_radius), cp.float32(1.0/dx), cp.int32(radius) ), );
    
    dz_2D_forw  = lambda x, y: FD2d_dict['dz_2D_forw']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nnz_radius), cp.float32(1.0/dz), cp.int32(radius) ), );
    
    
    dx_2D_back  = lambda x, y: FD2d_dict['dx_2D_back']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nnz_radius), cp.float32(1.0/dx), cp.int32(radius) ), );
    
    dz_2D_back  = lambda x, y: FD2d_dict['dz_2D_back']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nnz_radius), cp.float32(1.0/dz), cp.int32(radius) ), );


    operator_list = {}
    local_vars    = locals()
    for var_name, var_value in local_vars.items():
       if callable(var_value):
           operator_list[var_name] = var_value
           
    return operator_list



def define_derivative_K_space_2D(kx_arr, kz_arr, kx_arr2, kz_arr2, kx_kz_2, dx, dz, x_first=True):
    '''
        The same operator with FD
        
        all the input must be cp.float32 or cp.int32
        
        Note that  I should use  cp.float32(1.0) or cp.int32(1) for the multiply operator
    '''
    
    lap_f        = lambda x: cp.real( cp.fft.ifftn(cp.float32(-1.0)*kx_kz_2 * cp.fft.fftn(x)) );
    lap_2D       = lambda x, y: combine_x_y(lap_f, x, y);
    
    
    
    pre_forw_x    = ( +1j * kx_arr * cp.exp(+1j * kx_arr * dx / 2) ).astype(cp.complex64);
    
    pre_forw_z    = ( +1j * kz_arr * cp.exp(+1j * kz_arr * dz / 2) ).astype(cp.complex64);
    
    
    pre_back_x    = ( +1j * kx_arr * cp.exp(-1j * kx_arr * dx / 2) ).astype(cp.complex64);

    pre_back_z    = ( +1j * kz_arr * cp.exp(-1j * kz_arr * dz / 2) ).astype(cp.complex64);


    if x_first:
        x_axis=1;

        z_axis=0;
    else:
        x_axis=0;

        z_axis=1;
        
        
        
    dxx_f        = lambda x: cp.real( cp.fft.ifft (cp.float32(-1.0) * kx_arr2 * cp.fft.fft (x, axis=x_axis), axis=x_axis)  );
    dxx_2D       = lambda x, y: combine_x_y(dxx_f, x, y);
    
    
    
    dzz_f        = lambda x: cp.real( cp.fft.ifft (cp.float32(-1.0) * kz_arr2 * cp.fft.fft (x, axis=z_axis), axis=z_axis)  );
    dzz_2D       = lambda x, y: combine_x_y(dzz_f, x, y);
    
    
    
    
    
    dx_f        = lambda x: cp.real( cp.fft.ifft ( pre_forw_x * cp.fft.fft (x, axis=x_axis), axis=x_axis)  );
    dx_2D_forw   = lambda x, y: combine_x_y(dx_f, x, y);
    
    
    
    dz_f     = lambda x: cp.real( cp.fft.ifft ( pre_forw_z * cp.fft.fft (x, axis=z_axis), axis=z_axis)  );
    dz_2D_forw   = lambda x, y: combine_x_y(dz_f, x, y);
    
    
    
    
    
    dx_b         = lambda x: cp.real( cp.fft.ifft ( pre_back_x * cp.fft.fft (x, axis=x_axis), axis=x_axis) );
    dx_2D_back   = lambda x, y: combine_x_y(dx_b, x, y);
    
    
    
    dz_b         = lambda x: cp.real( cp.fft.ifft ( pre_back_z * cp.fft.fft (x, axis=z_axis), axis=z_axis) );
    dz_2D_back   = lambda x, y: combine_x_y(dz_b, x, y);
        
    
    



    div_f           = lambda x: dx_f(x) + dz_f(x);
    div_2D_forw     = lambda x, y: combine_x_y(div_f, x, y);
    
    
    div_b           = lambda x: dx_b(x) + dz_b(x);
    div_2D_back     = lambda x, y: combine_x_y(div_b, x, y);
    
    
    
    
    grad_f        = lambda x: [dx_f(x), dz_f(x)];
    grad_2D_forw  = lambda x, y1, y2: combine_x_y1y2(grad_f, x, y1, y2);
    
    
    grad_b        = lambda x: [dx_b(x), dz_b(x)];
    grad_2D_back  = lambda x, y1, y2: combine_x_y1y2(grad_b, x, y1, y2);

    
    
    operator_list = {}
    local_vars    = locals()
    for var_name, var_value in local_vars.items():
       if callable(var_value):
           operator_list[var_name] = var_value
           
    return operator_list


# a = cp.array([1+2j], dtype=cp.complex64) 
# b = cp.float32(2.0) 
# result = a * b  
# real_part = cp.real(result) 
# print(real_part.dtype)  


def define_cupy_derivative_F_space_3D(FD3d_dict, nnx_radius, nny_radius, nnz_radius, dx, dy, dz, grid, block, coe_2_d, coe_1_d, radius):
    '''
    Note that all  division is replaced with the multiply!!
    
    
    nnx_radius, nny_radius, nnz_radius, dx, dy, dz, coe_2_d, coe_1_d, radius should be int32 or float32.
    '''
    
    ### FD method for derivatives
    lap_3D      = lambda x, y: FD3d_dict['lap_3D']( grid, block, (x, y, coe_2_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dx/dx), cp.float32(1.0/dy/dy), cp.float32(1.0/dz/dz), cp.int32(radius) ), );
    
    
    dxx_3D      = lambda x, y: FD3d_dict['dxx_3D']( grid, block, (x, y, coe_2_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dx/dx), cp.int32(radius) ), );
    
    dyy_3D      = lambda x, y: FD3d_dict['dyy_3D']( grid, block, (x, y, coe_2_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dy/dy), cp.int32(radius) ), );
    
    dzz_3D      = lambda x, y: FD3d_dict['dzz_3D']( grid, block, (x, y, coe_2_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dz/dz), cp.int32(radius) ), );
    
    
    
    
    
    div_3D_forw = lambda x, y: FD3d_dict['div_3D_forw']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dx), cp.float32(1.0/dy), cp.float32(1.0/dz), cp.int32(radius) ), );
    div_3D_back = lambda x, y: FD3d_dict['div_3D_back']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dx), cp.float32(1.0/dy), cp.float32(1.0/dz), cp.int32(radius) ), );
    
    
    
    
    grad_3D_forw = lambda x, y1, y2, y3: FD3d_dict['grad_3D_forw']( grid, block, (x, y1, y2, y3, coe_1_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dx), cp.float32(1.0/dy), cp.float32(1.0/dz), cp.int32(radius) ), );
    grad_3D_back = lambda x, y1, y2, y3: FD3d_dict['grad_3D_back']( grid, block, (x, y1, y2, y3, coe_1_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dx), cp.float32(1.0/dy), cp.float32(1.0/dz), cp.int32(radius) ), );
    
    
    

    dx_3D_forw  = lambda x, y: FD3d_dict['dx_3D_forw']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dx), cp.int32(radius) ), );
    dy_3D_forw  = lambda x, y: FD3d_dict['dy_3D_forw']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dy), cp.int32(radius) ), );
    dz_3D_forw  = lambda x, y: FD3d_dict['dz_3D_forw']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dz), cp.int32(radius) ), );
    
    
    dx_3D_back  = lambda x, y: FD3d_dict['dx_3D_back']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dx), cp.int32(radius) ), );
    dy_3D_back  = lambda x, y: FD3d_dict['dy_3D_back']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dy), cp.int32(radius) ), );
    dz_3D_back  = lambda x, y: FD3d_dict['dz_3D_back']( grid, block, (x, y, coe_1_d, cp.int32(nnx_radius), cp.int32(nny_radius), cp.int32(nnz_radius), cp.float32(1.0/dz), cp.int32(radius) ), );


    operator_list = {}
    local_vars    = locals()
    for var_name, var_value in local_vars.items():
       if callable(var_value):
           operator_list[var_name] = var_value
           
    return operator_list



def define_derivative_K_space_3D(kx_arr, ky_arr, kz_arr, kx_arr2, ky_arr2, kz_arr2, kx_ky_kz_2, dx, dy, dz, x_first=True):
    '''
        The same operator with FD
        
        all the input must be cp.float32 or cp.int32
        
        Note that  I should use  cp.float32(1.0) or cp.int32(1) for the multiply operator
    '''
    
    lap_f        = lambda x: cp.real( cp.fft.ifftn( cp.float32(-1.0) * kx_ky_kz_2 * cp.fft.fftn(x)) );
    lap_3D       = lambda x, y: combine_x_y(lap_f, x, y);
    
    
    
    pre_forw_x    = ( +1j * kx_arr * cp.exp(+1j * kx_arr * dx / 2) ).astype(cp.complex64);
    pre_forw_y    = ( +1j * ky_arr * cp.exp(+1j * ky_arr * dy / 2) ).astype(cp.complex64);
    pre_forw_z    = ( +1j * kz_arr * cp.exp(+1j * kz_arr * dz / 2) ).astype(cp.complex64);
    
    
    pre_back_x    = ( +1j * kx_arr * cp.exp(-1j * kx_arr * dx / 2) ).astype(cp.complex64);
    pre_back_y    = ( +1j * ky_arr * cp.exp(-1j * ky_arr * dy / 2) ).astype(cp.complex64);
    pre_back_z    = ( +1j * kz_arr * cp.exp(-1j * kz_arr * dz / 2) ).astype(cp.complex64);


    if x_first:
        x_axis=2;
        y_axis=1;
        z_axis=0;
    else:
        x_axis=0;
        y_axis=1;
        z_axis=2;
        
        
       
        
    dxx_f        = lambda x: cp.real( cp.fft.ifft (cp.float32(-1.0) * kx_arr2 * cp.fft.fft (x, axis=x_axis), axis=x_axis)  );
    dxx_3D       = lambda x, y: combine_x_y(dxx_f, x, y);
    
    dyy_f        = lambda x: cp.real( cp.fft.ifft (cp.float32(-1.0) * ky_arr2 * cp.fft.fft (x, axis=y_axis), axis=y_axis)  );
    dyy_3D       = lambda x, y: combine_x_y(dyy_f, x, y);
    
    dzz_f        = lambda x: cp.real( cp.fft.ifft (cp.float32(-1.0) * kz_arr2 * cp.fft.fft (x, axis=z_axis), axis=z_axis)  );
    dzz_3D       = lambda x, y: combine_x_y(dzz_f, x, y);
    
    
    
    
    
    dx_f         = lambda x: cp.real( cp.fft.ifft ( pre_forw_x * cp.fft.fft (x, axis=x_axis), axis=x_axis)  );
    dx_3D_forw   = lambda x, y: combine_x_y(dx_f, x, y);
    
    dy_f         = lambda x: cp.real( cp.fft.ifft ( pre_forw_y * cp.fft.fft (x, axis=y_axis), axis=y_axis)  );
    dy_3D_forw   = lambda x, y: combine_x_y(dy_f, x, y);
    
    dz_f         = lambda x: cp.real( cp.fft.ifft ( pre_forw_z * cp.fft.fft (x, axis=z_axis), axis=z_axis)  );
    dz_3D_forw   = lambda x, y: combine_x_y(dz_f, x, y);
    
    
    
    
    
    dx_b         = lambda x: cp.real( cp.fft.ifft ( pre_back_x * cp.fft.fft (x, axis=x_axis), axis=x_axis) );
    dx_3D_back   = lambda x, y: combine_x_y(dx_b, x, y);
    
    dy_b         = lambda x: cp.real( cp.fft.ifft ( pre_back_y * cp.fft.fft (x, axis=y_axis), axis=y_axis) );
    dy_3D_back   = lambda x, y: combine_x_y(dy_b, x, y);
    
    dz_b         = lambda x: cp.real( cp.fft.ifft ( pre_back_z * cp.fft.fft (x, axis=z_axis), axis=z_axis) );
    dz_3D_back   = lambda x, y: combine_x_y(dz_b, x, y);
        
    
    



    div_f         = lambda x: dx_f(x) + dy_f(x) + dz_f(x);
    div_3D_forw   = lambda x, y: combine_x_y(div_f, x, y);
    
    
    div_b         = lambda x: dx_b(x) + dy_b(x) + dz_b(x);
    div_3D_back   = lambda x, y: combine_x_y(div_b , x, y);
    
    
    
    
    grad_f        = lambda x: [dx_f(x), dy_f(x), dz_f(x)];
    grad_3D_forw  = lambda x, y1, y2, y3: combine_x_y1y2y3(grad_f, x, y1, y2, y3);
    
    
    grad_b        = lambda x: [dx_b(x), dy_b(x), dz_b(x)];
    grad_3D_back  = lambda x, y1, y2, y3: combine_x_y1y2y3(grad_b, x, y1, y2, y3);

    
    
    operator_list = {}
    local_vars    = locals()
    for var_name, var_value in local_vars.items():
       if callable(var_value):
           operator_list[var_name] = var_value
           
    return operator_list

# a = cp.array([1+2j], dtype=cp.complex64) 
# b = cp.float32(2.0) 
# result = a * b  
# real_part = cp.real(result) 
# print(real_part.dtype)  
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
def cal_grid_reverse( dims=[100, 100, 100], block = (8, 8, 8), forward=False):
    '''
    Please reference the FD operator:
        
    Case 1 see Wave_eq_solver_2D:
        if the input array [nz, ny, nx]
        [nz, ny, nx] = list(input_array.shape)
        dims         = [nz, ny, nx]
        We should use 
        grid = cal_grid_reverse( dims, block, forward=False);

        note that all FD operators use this block and grid.
        Because it is faster than another case
    '''
    if len(dims)==1:
        
        grid = (
                    (dims[0] + block[0] - 1) // block[0],
               ) 
    if forward:
        if len(dims)==2:
            grid = (
                        (dims[0] + block[0] - 1) // block[0],
                        (dims[1] + block[1] - 1) // block[1],
                   ) 
        
            
        if len(dims)==3:
            grid = (
                        (dims[0] + block[0] - 1) // block[0],
                        (dims[1] + block[1] - 1) // block[1],
                        (dims[2] + block[2] - 1) // block[2],
                   ) 
    else:
        if len(dims)==2:
            grid = (
                        (dims[1] + block[0] - 1) // block[0],
                        (dims[0] + block[1] - 1) // block[1],
                   )   ### consistent with C+CUDA
        
            
        if len(dims)==3:
            grid = (
                        (dims[2] + block[0] - 1) // block[0],
                        (dims[1] + block[1] - 1) // block[1],
                        (dims[0] + block[2] - 1) // block[2],
                   )  ### consistent with C+CUDA
    return grid


def expand_ND_with_boundary(input_d, bu=0, bd=0, bb=0, bf=0, bl=0, br=0):
    
    if len(input_d.shape)==3:
        extended_arr    = cp.pad(input_d, pad_width=((bu, bd), (0, 0), (0, 0)), mode='edge')
        extended_arr    = cp.pad(extended_arr, pad_width=((0, 0), (bb, bf), (0, 0)), mode='edge')
        p_vel_arr       = cp.pad(extended_arr, pad_width=((0, 0), (0, 0), (bl, br)), mode='edge')
        
    else:
        extended_arr    = cp.pad(input_d, pad_width=((bu, bd), (0, 0)), mode='edge')
        p_vel_arr       = cp.pad(extended_arr, pad_width=((0, 0), (bl, br)), mode='edge')
        
    return p_vel_arr
    

def com_stablity_condition_acoustic(c_max, coe_2_d,  FD_order,  TIME_order, dx, dy, dz, dt, SP_bool, dims=3, log_file=""):
    
    kx_max = cp.pi/dx;
    ky_max = cp.pi/dy;
    kz_max = cp.pi/dz;
    
    if dims ==3:
        k_max  = cp.sqrt(kx_max * kx_max + ky_max * ky_max + kz_max * kz_max);
        d_d_d_d_recip = 1/dx/dx + 1/dy/dy + 1/dz/dz;
    if dims ==2:
        k_max  = cp.sqrt(kx_max * kx_max + kz_max * kz_max);
        d_d_d_d_recip = 1/dx/dx + 1/dz/dz;
        
      
        
    recommend_time_order  = int( 0.5 * c_max *  k_max * dt) + 1
    
    
    
    if not SP_bool:
        sum_coe_2_d         = 2.0*cp.sum(cp.abs(coe_2_d[1:FD_order])) + cp.abs(coe_2_d[0]);
    else:
        sum_coe_2_d         = cp.pi*cp.pi; 
        ##see lines, A recipe for stability of FD wave equation computations
    
    interval_t_max = 2 * TIME_order / c_max * cp.sqrt(1/d_d_d_d_recip/sum_coe_2_d);
    
    if len(log_file):
        
        file1               = "recommend TIME_order={}\n".format( recommend_time_order ); ##equation 33 at (Edvaldo S. Araujo and Reynam C. Pestana, 2019)
        WR.write_txt(log_file, file1 + file1, "a+");
        
        file1 = "sum_coe_2_d={}\n, c_max={}\n, kx_max={}\n, kz_max={}\n, c_max={}\n, k_max={}\n".format(sum_coe_2_d, c_max, kx_max, kz_max, c_max, k_max);
        file2 = "interval_t_max={}\n".format(interval_t_max);
        
        WR.write_txt(log_file, file1, "a+");
        WR.write_txt(log_file, file2 + file2, "a+");
    
    return interval_t_max
    
    
def com_stablity_condition_vector_acoustic(c_max, coe_1_d,  FD_order,  TIME_order, dx, dy, dz, dt, SP_bool, dims=3, log_file=""):
    
    kx_max = cp.pi/dx;
    ky_max = cp.pi/dy;
    kz_max = cp.pi/dz;
    if dims ==3:
        k_max  = cp.sqrt(kx_max * kx_max + ky_max * ky_max + kz_max * kz_max);
        d_d_d_d_recip = 1/dx/dx + 1/dy/dy + 1/dz/dz;
    if dims ==2:
        k_max  = cp.sqrt(kx_max * kx_max + kz_max * kz_max);
        d_d_d_d_recip = 1/dx/dx + 1/dz/dz;
        
             
    
    
    recommend_time_order  = int( 0.5 * c_max *  k_max * dt) + 1
    
    
    
    
    if not SP_bool:
        sum_coe_1_d         = cp.sum(cp.abs(coe_1_d));
    if SP_bool:
        sum_coe_1_d         = cp.pi/cp.sqrt(4.0); ##it is the same with the second order
    
    interval_t_max          = TIME_order / c_max / cp.sqrt(d_d_d_d_recip) / sum_coe_1_d;
    ##since this is istropic, otherwise it is depedent on C11 C33 C55 and so on
    
    if len(log_file):
        
        file1               = "recommend TIME_order={}\n".format( recommend_time_order ); ##equation 33 at (Edvaldo S. Araujo and Reynam C. Pestana, 2019)
        WR.write_txt(log_file, file1 + file1, "a+");
        
        file1 = "sum_coe_1_d={}\n, c_max={}\n, kx_max={}\n, kz_max={}\n, c_max={}\n, k_max={}\n".format(sum_coe_1_d, c_max, kx_max, kz_max, c_max, k_max);
        file2 = "interval_t_max={}\n".format(interval_t_max);
        
        WR.write_txt(log_file, file1, "a+");
        WR.write_txt(log_file, file2 + file2, "a+");
    
    return interval_t_max


def compute_FD_time_order(c_max: float = 3000, dx_list: list = [10, 10, 10], dt: float = 0.001, log_file = "compute_FD_time_order.txt"):                 

    #####provide the velocity, dx, dt, compute the optimal time order
    k_max               = np.sqrt( sum( (np.pi/dx)**2 for dx in dx_list) )            
    recommend_time_order= int( 0.5 * c_max *  k_max * dt) + 1
    file1               = "c_max={}(m/s)\n".format( c_max );
    file2               = "dx_list={}(m)\n".format( dx_list );
    file3               = "dt={} (s)\n".format( dx_list );
    file4               = "k_max={}( radius )\n".format( k_max );
    file5               = "recommend_time_order={}\n".format( recommend_time_order );
    WR.write_txt(log_file, file1 + file2 + file3 + file4 + file5, "w+");


#####################################
###################################
#################################
##############################
############################
#####################################
###################################
#################################
##############################
############################
def acoustic_model_paramerization_ref(K_arr, D_arr, covering_variable=False, model_paramerization_mark=1):
    '''
    Compute the acoustic_model_paramerization based on the relative pertubation (image) of K and density
    K_arr: the relative pertubation (image) of K
    D_arr: the relative pertubation (image) of denisty
    covering_variable: if or not cover the input variables
    '''
    
    module  =WR.get_module_type(K_arr)
    
    output1 = module.zeros_like(K_arr)
    output2 = module.zeros_like(D_arr)
    
    if   model_paramerization_mark==1:
        print("velocity density paramerization")
        output1[:]  = 2.0*K_arr[:]
        output2[:]  = 1.0*K_arr[:] + 1.0*D_arr[:]
    
    elif model_paramerization_mark==2:    
        print("imepdance-velocity paramerization")
        output1[:]  = 1.0*K_arr[:] + 1.0*D_arr[:]
        output2[:]  = 1.0*K_arr[:] - 1.0*D_arr[:]
        
        
    if covering_variable==True:
        K_arr[:] = output1[:]
        D_arr[:] = output2[:]
    
    
    return output1, output2


def acoustic_model_paramerization_hessian(KK_arr, KD_arr, DK_arr, DD_arr, covering_variable=False, model_paramerization_mark=1):
    '''
    Compute the acoustic_model_paramerization based on the hessian of K and density
    KK_arr: hessian of KK
    KD_arr: hessian of KD
    DK_arr: hessian of DK
    DD_arr: hessian of DD
    covering_variable: if or not cover the input variables
    model_paramerization_mark:
    '''
    
    module  = WR.get_module_type(KK_arr)
    
    output1 = module.zeros_like(KK_arr)
    output2 = module.zeros_like(KK_arr)
    output3 = module.zeros_like(KK_arr)
    output4 = module.zeros_like(KK_arr)
    
    if   model_paramerization_mark==1:
        print("velocity density paramerization")
        output1[:]  = 4.0*KK_arr[:]
        output2[:]  = 2.0*KK_arr[:] + 2.0*KD_arr[:]
        output3[:]  = 2.0*KK_arr[:] + 2.0*DK_arr[:]
        output4[:]  = 1.0*KK_arr[:] + 1.0*KD_arr[:] + 1.0*DK_arr[:] + 1.0*DD_arr[:]
        
    elif model_paramerization_mark==2:
        print("imepdance-velocity paramerization")
        output1[:]  = 1.0*KK_arr[:] + 1.0*KD_arr[:] + 1.0*DK_arr[:] + 1.0*DD_arr[:]
        output2[:]  = 1.0*KK_arr[:] - 1.0*KD_arr[:] + 1.0*DK_arr[:] - 1.0*DD_arr[:]
        
        output3[:]  = 1.0*KK_arr[:] + 1.0*KD_arr[:] - 1.0*DK_arr[:] - 1.0*DD_arr[:]
        output4[:]  = 1.0*KK_arr[:] - 1.0*KD_arr[:] - 1.0*DK_arr[:] + 1.0*DD_arr[:]
        
    if covering_variable==True:
        KK_arr[:] = output1[:]
        KD_arr[:] = output2[:]
        DK_arr[:] = output3[:]
        DD_arr[:] = output4[:]
    
    
    return output1, output2, output3, output4

##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################


def extend_parameter_for_dz(para1_arr, angle_num, module=np):
    
    nz = para1_arr.shape[-1];
    
    if module is torch:
        vel1 = para1_arr[..., 1:nz]
        vel2 = para1_arr[..., nz-1].contiguous().unsqueeze(-1)
        vel = torch.cat((vel1, vel2), axis=-1)

        vel1_3d = vel.unsqueeze(0).expand((angle_num, *vel.shape))
        vel2_3d = vel.unsqueeze(0).expand((angle_num, *vel.shape))

    else:
        
        vel1 = 1.0 * para1_arr[..., 1:nz]
        vel2 = 1.0 * para1_arr[..., nz-1][..., module.newaxis]
        vel  = module.concatenate((vel1, vel2), axis=-1)

        # Add a new dimension at the beginning and tile
        vel1_3d = module.expand_dims(para1_arr, axis=0)
        vel2_3d = module.expand_dims(vel, axis=0)

        # Tile along the first dimension
        vel1_3d = module.tile(vel1_3d, (angle_num, *( [1] * (vel1.ndim) )))
        vel2_3d = module.tile(vel2_3d, (angle_num, *( [1] * (vel2.ndim) )))

    return vel1_3d, vel2_3d

 

def cal_angle_ref_ND(in_list, angle_start=0, angle_num=90, dangle=1.0, model_para=0):
    '''
    This function aims to compute acoustic angle-dependent reflectivity coefficient through the following parameter
    (model_para=0) constant velocity, 
    (model_para=1) velocity-density, and 
    (model_para=2) velocity-impedance
    
    The input list:
    shape = [nx, nz],
    shape = [nx, ny, nz],
    shape = [batch, nx, ny, nz],
    
    return [angle_num, shape],
    
    model_para=0: constant density,
    model_para=1: vp-density,
    model_para=2: vp-impedance;
    '''

    vel_arr = in_list[0];
    
    module  = WR.get_module_type(vel_arr);
        
    try:
        para2_arr    = in_list[1];
    except:
        print( "There is no another parameter for cal_angle_ref_ND, so you should set the parameter model_para=0" )
    
    
    vel1_ND, vel2_ND = extend_parameter_for_dz(vel_arr, angle_num, module=module);
    vel1_ND_shape    = vel1_ND.shape;
    
    #########
    gama1            = module.arange(angle_start, angle_start + angle_num * dangle, dangle)/180.0 * module.pi;
    
    num_dims_to_add  = len(vel1_ND.shape) - len(gama1.shape);
    
    if module is torch:
        gama1     = gama1.to(vel_arr.device);
    
        for _ in range(num_dims_to_add):
            gama1 = gama1.unsqueeze(-1); ### last dimension
        
        gama1_ND  = gama1.expand(*vel1_ND_shape);
        
    else:
        for _ in range(num_dims_to_add):
            gama1 = module.expand_dims(gama1, axis=-1);
            
        gama1_ND = module.tile(gama1, (1,) * (len(vel1_ND_shape) - num_dims_to_add) + vel1_ND_shape[-num_dims_to_add:]);
    
    radian 	    =  vel2_ND/vel1_ND * module.sin( gama1_ND );
    
    gama2_ND    =  module.arcsin( module.clip(radian, -1.0, 1.0) ) ;
    

    if   model_para==0:
        A      = 1.0 * vel2_ND * module.cos( 1.0 * gama1_ND );
        B      = 1.0 * vel1_ND * module.cos( 1.0 * gama2_ND );

        output       = 1.0 * (A - B) / (A + B);
        
        mask         = module.isnan(output);
        output[mask] = 0; 
        
    elif model_para==1:
        den1_ND, den2_ND = extend_parameter_for_dz(para2_arr, angle_num, module=module);
        
        A      = 1.0 * den2_ND * vel2_ND * module.cos( 1.0 * gama1_ND );
        B      = 1.0 * den1_ND * vel1_ND * module.cos( 1.0 * gama2_ND );

        output       = 1.0 * (A - B) / (A + B);
        
        mask = module.isnan(output);
        output[mask] = 0; 
        
    else:
        I1_ND, I2_ND = extend_parameter_for_dz(para2_arr, angle_num, module=module);

        A      = 1.0 * I2_ND * module.cos( 1.0 * gama1_ND );
        B      = 1.0 * I1_ND * module.cos( 1.0 * gama2_ND );

        output = 1.0 * (A - B) / (A + B);
        
        mask         = module.isnan(output);
        output[mask] = 0; 
    
    
    return output

    # if   len( list(output.shape) ) ==2:
    #     return output.reshape( [1, 1] +  list(output.shape) )
    # elif len( list(output.shape) ) ==3:
    #     return output.reshape( [1] +  list(output.shape) )
    # else:
    #     return output






def apply_torch_2Doperator_AMP_IDLSM(op_list, x_list):
    '''
    apply_torch_2Doperator_acoustic_multi_parameter IDLSM
    '''
    size1, size2, nx, nz = x_list[0].shape ;
    
    output_list = [torch.zeros_like(tensor, requires_grad=False) for tensor in x_list]
    
    for ix in range(0, size1):
        for iz in range(0, size2):
            vel             = x_list[0][ix, iz, :, :].reshape(nx,nz)
            den             = x_list[1][ix, iz, :, :].reshape(nx,nz)
            mig_sys_part1   = op_list[0].apply(vel);
            mig_sys_part2   = op_list[1].apply(den);
            mig_sys_part3   = op_list[2].apply(vel);
            mig_sys_part4   = op_list[3].apply(den);
            
            output_list[0][ix, iz, :, :] =  mig_sys_part1 + mig_sys_part2;
            
            output_list[1][ix, iz, :, :] =  mig_sys_part3 + mig_sys_part4;
            
    return output_list


def apply_operator_multiparameter_ND(op_list, x_list, weight_matrix=None, use_torch=False):
    """
    Function to apply operations on 2D, 3D, or 4D tensors.
    
    Args:
    - op_list: list of operators to apply on the tensors
    - x_list: list of tensors, where the first tensor represents velocity and the second density
    - weight_matrix: optional matrix to weight the output
    - use_torch: if True, use the PyTorch-specific application method
    
    Returns:
    - output_list: list of resulting tensors after applying the operators
    """
    
    if len(x_list) != 2:
        raise RuntimeError("len(x_list) != 2, len(x_list)={}".format(len(x_list)))

    dims  = len(x_list[0].shape)
    shape = x_list[0].shape
    
    module = WR.get_module_type(x_list[0])
    
    output_list = [module.zeros_like(tensor) for tensor in x_list]

    # Determine whether to use PyTorch's `.apply()` method or a regular function call
    def apply_op(op, tensor):
        if module.__name__ == 'torch':  # If tensors are from PyTorch
            try:
                return op.apply(tensor)
            except:
                return op(tensor)
        else:  # Use standard function call for other frameworks
            return op(tensor)

    if dims == 2:
        vel = x_list[0]
        den = x_list[1]
        mig_sys_part1 = apply_op(op_list[0], vel)
        mig_sys_part2 = apply_op(op_list[1], den)
        mig_sys_part3 = apply_op(op_list[2], vel)
        mig_sys_part4 = apply_op(op_list[3], den)

        output_list[0] = (mig_sys_part1 + mig_sys_part2)
        output_list[1] = (mig_sys_part3 + mig_sys_part4)
        
        if weight_matrix is not None:
            output_list[0] *= weight_matrix
            output_list[1] *= weight_matrix

    elif dims == 3:
        size1, nx, nz = shape
        for ix in range(size1):
            vel = x_list[0][ix, :, :].reshape(nx, nz)
            den = x_list[1][ix, :, :].reshape(nx, nz)
            mig_sys_part1 = apply_op(op_list[0], vel)
            mig_sys_part2 = apply_op(op_list[1], den)
            mig_sys_part3 = apply_op(op_list[2], vel)
            mig_sys_part4 = apply_op(op_list[3], den)
            
            output_list[0][ix, :, :] = mig_sys_part1 + mig_sys_part2
            output_list[1][ix, :, :] = mig_sys_part3 + mig_sys_part4
            
            if weight_matrix is not None:
                output_list[0][ix, :, :] *= weight_matrix
                output_list[1][ix, :, :] *= weight_matrix

    else:
        size1, size2, nx, nz = shape
        for ix in range(size1):
            for iz in range(size2):
                vel = x_list[0][ix, iz, :, :].reshape(nx, nz)
                den = x_list[1][ix, iz, :, :].reshape(nx, nz)
                mig_sys_part1 = apply_op(op_list[0], vel)
                mig_sys_part2 = apply_op(op_list[1], den)
                mig_sys_part3 = apply_op(op_list[2], vel)
                mig_sys_part4 = apply_op(op_list[3], den)
                
                output_list[0][ix, iz, :, :] = mig_sys_part1 + mig_sys_part2
                output_list[1][ix, iz, :, :] = mig_sys_part3 + mig_sys_part4
                
                if weight_matrix is not None:
                    output_list[0][ix, iz, :, :] *= weight_matrix
                    output_list[1][ix, iz, :, :] *= weight_matrix
    
    return output_list

def apply_torch_operator_multiparameter_ND(op_list, x_list, weight_matrix=None):
    """
    Function to apply operations on 2D, 3D, or 4D tensors.
    
    Args:
    - op_list: list of operators to apply on the tensors
    - x_list: list of tensors, where the first tensor represents velocity and the second density
    
    Returns:
    - output_list: list of resulting tensors after applying the operators
    """

    if len(x_list) !=2:
        raise RuntimeError("len(x_list) !=2");

    dims  = len(x_list[0].shape)
    shape = x_list[0].shape
    
    module = WR.get_module_type(x_list[0]);
    
    output_list = [module.zeros_like(tensor) for tensor in x_list]
    if dims == 2:
        vel             = x_list[0]
        den             = x_list[1]
        mig_sys_part1   = op_list[0].apply(vel)
        mig_sys_part2   = op_list[1].apply(den)
        mig_sys_part3   = op_list[2].apply(vel)
        mig_sys_part4   = op_list[3].apply(den)

        output_list[0] = (mig_sys_part1 + mig_sys_part2)
        output_list[1] = (mig_sys_part3 + mig_sys_part4)
        if weight_matrix:
            output_list[0] = output_list[0] * weight_matrix
            output_list[1] = output_list[1] * weight_matrix

    elif dims == 3:
       size1, nx, nz = shape
       for ix in range(0, size1):
               vel             = x_list[0][ix, :, :].reshape(nx,nz)
               den             = x_list[1][ix, :, :].reshape(nx,nz)
               mig_sys_part1   = op_list[0].apply(vel);
               mig_sys_part2   = op_list[1].apply(den);
               mig_sys_part3   = op_list[2].apply(vel);
               mig_sys_part4   = op_list[3].apply(den);
               
               output_list[0][ix, :, :] =  mig_sys_part1 + mig_sys_part2;
               output_list[1][ix, :, :] =  mig_sys_part3 + mig_sys_part4;
               
               if weight_matrix:
                   output_list[0][ix, :, :] = output_list[0][ix, :, :] * weight_matrix
                   output_list[1][ix, :, :] = output_list[1][ix, :, :] * weight_matrix

    else:
       size1, size2, nx, nz = shape
       
       for ix in range(0, size1):
           for iz in range(0, size2):
               vel             = x_list[0][ix, iz, :, :].reshape(nx,nz)
               den             = x_list[1][ix, iz, :, :].reshape(nx,nz)
               mig_sys_part1   = op_list[0].apply(vel);
               mig_sys_part2   = op_list[1].apply(den);
               mig_sys_part3   = op_list[2].apply(vel);
               mig_sys_part4   = op_list[3].apply(den);
               
               output_list[0][ix, iz, :, :] =  mig_sys_part1 + mig_sys_part2;
               output_list[1][ix, iz, :, :] =  mig_sys_part3 + mig_sys_part4;
               
               if weight_matrix:
                   output_list[0][ix, iz, :, :] = output_list[0][ix, iz, :, :] * weight_matrix
                   output_list[1][ix, iz, :, :] = output_list[1][ix, iz, :, :] * weight_matrix
    
    return output_list






def save_oneslice_wavefield(save_array, p1):
    '''
    Save using cp.asnumpy
    '''
    if isinstance(save_array, cp.ndarray):
        save_array[...] = p1;
    else:
        save_array[...] = cp.asnumpy(p1);

def set_oneslice_wavefield(save_array, p1):
    '''
    Set using cp.asarray
    
    '''
    if isinstance(save_array, cp.ndarray):
        p1[:]   = save_array;
    else:
        p1[:]   = cp.asarray(save_array);

def add_oneslice_wavefield(save_array, p1):
    '''
    Set using cp.asarray
    
    '''
    if isinstance(save_array, cp.ndarray):
        p1[:] += save_array[:];
        
    else:
        p1[:]  += cp.asarray (save_array[:])

        
def wave_ray_correct_time(time_dict, max_ns=100):
    '''
    I want to use the max amplitude of source wavelet to correct the traveltime of time_dict,
    Since the min traveltime of each shot should be max_ns
    '''    
    for name, value in time_dict.items():
        
        time_min =   value.min().item();
        
        if time_min !=max_ns:
            file1=  "time_min={}, max_ns={}, time step has been corrected according to the value of source signal\n".format(  time_min,  max_ns );
            print(file1);
            time_dict[name] += (max_ns-time_min);
       
        
def wave_ray_write_read_dict(
    input_path: str,
    
    input_list_dict: List[Dict] = [],
    
    name_list: List[str] = ["amp", "time", "px", "py", "pz", "info_dict"],
    
    write_or_read: bool = False,
    
    load_to_cupy = False,
    
    ray_dims: int = 2,
    
    output_time: bool = True,
    
) -> Union[List[Dict], None]:
    """
    Reads or writes dictionaries for wave ray acquisition data.

    Parameters:
    - input_path (str): The path to the directory where files are read from or written to.
    
    - input_list_dict (List[Dict], optional): List of dictionaries to be written. Defaults to an empty list.
    
    - name_list (List[str], optional): List of file suffixes for constructing file names. Defaults to ["amp", "time", "px", "py", "pz", "min_info", "max_info"].
    
    - write_or_read (bool, optional): If False, read mode is used; if True, write mode is used. Defaults to False.
    
    - ray_dims (int, optional): Indicates the dimensionality of the ray data (2D or 3D). Defaults to 2.

    Returns:
    - Union[List[Dict], None]: 
        there are   amp, time, px, py, pz, max_info, min_info   seven dictionary
        
        If reading, returns a list of dictionaries; 
        if writing, returns None.
    """
    
    # List of suffixes for the file names
    # suffixes = ["amp", "time", "px", "py", "pz", "min_info", "max_info"]
    
    start_time = time.time()
    
    # Construct the file names using a loop
    file_names = [input_path + suffix + '.npz' for suffix in name_list]
    
    ##read
    if not write_or_read:
        # Initialize an empty list to store dictionaries
        dict_list = []
    
        # Read dictionaries from files
        for i, file_name in enumerate(file_names):
            # Read the file based on the ray_dims check for py_dict
            if i == 3 and ray_dims != 3:  # If it is the 'py' dictionary and ray_dims is not 3
                dict_list.append({})  # Append an empty dictionary for 'py dict'
            else:
                if load_to_cupy:
                    dict_arr  = WR.dict_read(file_name);
                    for key, value in dict_arr.items():
                        '''Check if the value is a NumPy array  Convert NumPy array to CuPy array'''
                        if isinstance(value, np.ndarray):
                            dict_arr[key] = cp.asarray(value, dtype=cp.float32)
                else:
                    dict_arr  = WR.dict_read(file_name);
                    dict_arr  = TF.dict_arr_to_float32_contiguous(dict_arr);
                
                ''' "min_info", "max_info" should be int 32 '''
                if file_name == file_names[-1]:
                    dict_arr  = TF.dict_arr_to_int32(dict_arr);
                
                    
                dict_list.append( dict_arr )  # Append the read dictionary
        
        return dict_list
    
    ##write
    else:
        
        for dictionary, file_name in zip(input_list_dict, file_names):
            WR.dict_write(file_name, dictionary)


    load_savez_time = time.time() - start_time
    if output_time:
        print("np.savez save or load time: {}s".format(load_savez_time) ) ;



def check_amp_time_pxpypz_dict(sx_arr_dict, sy_arr_dict, sz_arr_dict, gx_arr_dict, gy_arr_dict, gz_arr_dict, amp_dict, min_info_dict, max_info_dict, log_file="log_check_amp_time_pxpypz_dict.txt"):
    
    shot_num = len(sx_arr_dict);
    
    for ishot, (s_key, value) in enumerate(sx_arr_dict.items()):
        sx       = sx_arr_dict[s_key][0]
        sy       = sy_arr_dict[s_key][0]
        sz       = sz_arr_dict[s_key][0]
        
        gx_arr   = gx_arr_dict[s_key]
        gy_arr   = gy_arr_dict[s_key]
        gz_arr   = gz_arr_dict[s_key]
        rec_num  = len( gz_arr_dict[s_key] );
    
        
        file1 = "ishot={}\n".format(ishot)
        file2 = "shot_num={}, rec_num={}\n".format(shot_num, rec_num)
        file3 = "sx={} sy={} sz={}\n".format(sx, sy, sz);
        
        WR.write_txt(log_file, file1 + file2 + file3);
        
        s_amp_d     = amp_dict[s_key]
        s_min_info  = min_info_dict[s_key]
        s_max_info  = max_info_dict[s_key]
        
        for ig in range(0, rec_num):
            
            gx = gx_arr[ig]
            gy = gy_arr[ig]
            gz = gz_arr[ig]
    
            r_key   = (gx, gy, gz)
            
            r_amp_d     = amp_dict[r_key]
            r_min_info  = min_info_dict[r_key]
            r_max_info  = max_info_dict[r_key]
    
            
            if s_key not in amp_dict:
                raise ValueError("s_key={} is not in amp_dict".format(s_key))

            if r_key not in amp_dict:
                raise ValueError("r_key={} is not in amp_dict".format(r_key))
           
            
            model_min_info   = [min(s, r) for s, r in zip(s_min_info, r_min_info)]
            model_max_info   = [max(s, r) for s, r in zip(s_max_info, r_max_info)]

            
            bl = int(s_min_info[0]     - model_min_info[0]);
            br = int(model_max_info[0] - s_max_info[0]    );
            
            bb = int(s_min_info[1]     - model_min_info[1]);
            bf = int(model_max_info[1] - s_max_info[1]);
            
            if bl < 0:
                raise ValueError(
                    "Invalid padding width: bl < 0. Computed value: bl={}. "
                    "model_min_info[0]={}, s_min_info[0]={}".format(bl, model_min_info[0], s_min_info[0]) )
            
            if br < 0:
                raise ValueError("br<0")
            
            if bb < 0:
                raise ValueError("bb<0")
                
            if bf < 0:
                raise ValueError("bf<0")


def extend_or_crop_to_align_model_2D(
                                    p_vel_arr, 
                                    model_min_info, model_max_info, 
                                    
                                    s_amp_d, 
                                    s_min_info, s_max_info
                                    ):
    """
    Extend or crop s_amp_d to align with p_vel_arr based on their min and max information.

    Parameters:
    - p_vel_arr: The velocity array (used to determine the target size and alignment).
    - model_min_info: Tuple/list, minimum coordinate info for p_vel_arr (e.g., [x_min, z_min]).
    - model_max_info: Tuple/list, maximum coordinate info for p_vel_arr (e.g., [x_max, z_max]).
    - s_amp_d: The array to be extended or cropped.
    - s_min_info: Tuple/list, minimum coordinate info for s_amp_d.
    - s_max_info: Tuple/list, maximum coordinate info for s_amp_d.

    Returns:
    - The extended or cropped s_amp_d array, aligned with p_vel_arr.
    """
    
    '''
    if pad_left_x>0, we should pading
    if pad_right_x>0, we should pading
    '''
    
    pad_left_x  = s_min_info[0]     - model_min_info[0] 
    pad_right_x = model_max_info[0] - s_max_info[0]  

    if pad_left_x > 0:
        s_amp_d = cp.pad(s_amp_d, pad_width=((0, 0), (pad_left_x, 0)), mode='constant', constant_values=0)
    elif pad_left_x < 0:
        '''Crop from the left, it is a negative'''
        s_amp_d = s_amp_d[:, -pad_left_x:]


    if pad_right_x > 0:
        s_amp_d_x = cp.pad(s_amp_d, pad_width=((0, 0), (0, pad_right_x)), mode='constant', constant_values=0)
    elif pad_right_x < 0:  
        '''Crop from the right, it is a negative'''
        s_amp_d = s_amp_d[:, 0:s_amp_d.shape[1] + pad_right_x]

    return TF.array_cp_to_contiguous( s_amp_d )


def extend_or_crop_to_align_model_3D(
                                    p_vel_arr, 
                                    model_min_info, model_max_info, 
                                    
                                    s_amp_d, 
                                    s_min_info, s_max_info
                                    ):
    """
    Extend or crop s_amp_d (3D) to align with p_vel_arr based on their min and max information.

    Parameters:
    - p_vel_arr: The 3D velocity array (used to determine the target size and alignment).
    - model_min_info: Tuple/list, minimum coordinate info for p_vel_arr (e.g., [x_min, y_min, z_min]).
    - model_max_info: Tuple/list, maximum coordinate info for p_vel_arr (e.g., [x_max, y_max, z_max]).
    - s_amp_d: The 3D array to be extended or cropped.
    - s_min_info: Tuple/list, minimum coordinate info for s_amp_d.
    - s_max_info: Tuple/list, maximum coordinate info for s_amp_d.

    Returns:
    - The extended or cropped s_amp_d array, aligned with p_vel_arr.
    """
    '''
    if pad_left_x>0, we should pading
    if pad_right_x>0, we should pading
    '''
    pad_left_x  = s_min_info[0]     - model_min_info[0] 
    pad_right_x = model_max_info[0] - s_max_info[0]

    pad_left_y  = s_min_info[1]     - model_min_info[1] 
    pad_right_y = model_max_info[1] - s_max_info[1]

    '''Handle x-dimension padding or cropping'''
    if pad_left_x > 0:
        s_amp_d = cp.pad(s_amp_d, pad_width=((0, 0), (0, 0), (pad_left_x, 0)), mode='constant', constant_values=0)
    elif pad_left_x < 0:
        s_amp_d = s_amp_d[:, :, -pad_left_x:]

    if pad_right_x > 0:
        s_amp_d = cp.pad(s_amp_d, pad_width=((0, 0), (0, 0), (0, pad_right_x)), mode='constant', constant_values=0)
    elif pad_right_x < 0:
        s_amp_d = s_amp_d[:, :, 0:s_amp_d.shape[2] + pad_right_x]


    '''Handle y-dimension padding or cropping'''
    if pad_left_y > 0:
        s_amp_d = cp.pad(s_amp_d, pad_width=((0, 0), (pad_left_y, 0), (0, 0)), mode='constant', constant_values=0)
    elif pad_left_y < 0:
        s_amp_d = s_amp_d[:, -pad_left_y:, :]

    if pad_right_y > 0:
        s_amp_d = cp.pad(s_amp_d, pad_width=((0, 0), (0, pad_right_y), (0, 0)), mode='constant', constant_values=0)
    elif pad_right_y < 0:
        s_amp_d = s_amp_d[:, 0:s_amp_d.shape[1] + pad_right_y, :]


    return TF.array_cp_to_contiguous(s_amp_d)



def extend_amp_px_py_pz_to_vel_2D(p_vel_arr, model_min_info, model_max_info, s_amp_d, s_time_d, s_px_d, s_py_d, s_pz_d, s_min_info, s_max_info):
    
    if s_amp_d.shape != p_vel_arr.shape:
        
        bl = cp.int32(s_min_info[0] - model_min_info[0]);
        br = cp.int32(model_max_info[0] - s_max_info[0]);

        pad_func = lambda x: cp.pad(x,  pad_width=((0, 0), (bl, br)), mode='constant', constant_values=0)
        
        s_amp_d    =  pad_func(s_amp_d)
        s_time_d    = pad_func(s_time_d)
        s_px_d      = pad_func(s_px_d)
        s_pz_d      = pad_func(s_pz_d)
        
    return s_amp_d, s_time_d, s_px_d, s_py_d, s_pz_d


def extend_amp_px_py_pz_to_vel_3D(p_vel_arr, model_min_info, model_max_info, s_amp_d, s_time_d, s_px_d, s_py_d, s_pz_d, s_min_info, s_max_info):
    
    if s_amp_d.shape != p_vel_arr.shape:
        
        bl = cp.int32(s_min_info[0] - model_min_info[0]);
        br = cp.int32(model_max_info[0] - s_max_info[0]);
        
        bb = cp.int32(s_min_info[1] - model_min_info[1]);
        bf = cp.int32(model_max_info[1] - s_max_info[1]);

        pad_func1 = lambda x: cp.pad(x,  pad_width=((0, 0), (0, 0), (bl, br)), mode='constant', constant_values=0)
        pad_func2 = lambda x: cp.pad(x,  pad_width=((0, 0), (bb, bf), (0, 0)), mode='constant', constant_values=0)
        pad_func  = lambda x: pad_func2(pad_func1 (x) );
        
        s_amp_d    =  pad_func(s_amp_d)
        s_time_d    = pad_func(s_time_d)
        s_px_d      = pad_func(s_px_d)
        s_py_d      = pad_func(s_py_d)
        s_pz_d      = pad_func(s_pz_d)
        
    return s_amp_d, s_time_d, s_px_d, s_py_d, s_pz_d
    
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel
# --------------------kernel

def apply_1D_cupy_grad_3D(input_Nd, output1_Nd, output2_Nd, output3_Nd, ini_1D_arr1, out_1D_arr1, out_1D_arr2, out_1D_arr3, N_length, N_list, dx_op_lambda_back, grid, block):
    '''
    grad for applying 1D CUDA + C kernel operator.
    apply_1D_cupy_grad_3D(p2, wavefield_ND_arr1, wavefield_ND_arr2, wavefield_ND_arr3, ini_1D_arr1, out_1D_arr1, out_1D_arr2, out_1D_arr3, N_length, N_list, F_grad_op_lambda_forw, grid, block)
    '''
    
    ini_1D_arr1 = input_Nd.ravel(order='C')
    
    
    out_1D_arr1.fill(0)
    out_1D_arr2.fill(0)
    out_1D_arr3.fill(0)
    
    
    dx_op_lambda_back(ini_1D_arr1, out_1D_arr1, out_1D_arr2, out_1D_arr3)
    
    
    output1_Nd[:] = out_1D_arr1.reshape(N_list, order='C') 
    output2_Nd[:] = out_1D_arr2.reshape(N_list, order='C') 
    output3_Nd[:] = out_1D_arr3.reshape(N_list, order='C') 


def apply_1D_cupy_grad_2D(input_Nd, output1_Nd, output3_Nd, ini_1D_arr1, out_1D_arr1, out_1D_arr3, N_length, N_list, dx_op_lambda_back, grid, block):
    '''
    grad for applying 1D CUDA + C kernel operator.
    '''
    
    ini_1D_arr1 = input_Nd.ravel(order='C')
    
    
    out_1D_arr1.fill(0)
    out_1D_arr3.fill(0)
    
    
    dx_op_lambda_back(ini_1D_arr1, out_1D_arr1, out_1D_arr3)
    
    
    output1_Nd[:] = out_1D_arr1.reshape(N_list, order='C') 

    output3_Nd[:] = out_1D_arr3.reshape(N_list, order='C') 


def apply_1D_cupy_operator_return_ND_array(input_Nd, output_Nd, ini_1D_arr1, out_1D_arr1, N_length, N_list, dx_op_lambda_back, grid, block):
    '''
    Optimized function for applying 1D CUDA + C kernel operator.
    '''
    
    ini_1D_arr1 = input_Nd.ravel(order='F')
    
    out_1D_arr1.fill(0)
    
    dx_op_lambda_back(ini_1D_arr1, out_1D_arr1)
    
    output_Nd[:] = out_1D_arr1.reshape(N_list, order='F')



def apply_1D_cupy_operator_return_ND_array_old(input_Nd, output_Nd, ini_1D_arr, out_1D_arr, N_length, N_list, dx_op_lambda_back, grid, block):
    '''
    apply_1D_operator_for_PDE
    Usage:
        
        ##p _ x
        apply_1D_cupy_operator_return_ND_array(input_Nd, ini_1D_arr, out_1D_arr, wavefield_arr1, N_length, N_list, dx_op_lambda_back, grid, block);

    '''
    ini_1D_arr.reshape(N_length).fill(0);
    ini_1D_arr[:]       = 1.0 * input_Nd.reshape(N_length);
    
    # if len(N_list) ==2:
    #     reshape_2D[grid, block](ini_1D_arr, input_Nd, N_list[0], N_list[1], 1);
    # else:
    #     reshape_3D[grid, block](ini_1D_arr, input_Nd, N_list[0], N_list[1], N_list[2], 1);
        
    output_Nd.reshape(N_length).fill(0);
    
    ##apply operator
    dx_op_lambda_back(ini_1D_arr, output_Nd);

    output_Nd = output_Nd.reshape(N_list);
    
    # if len(N_list) ==2:
    #     reshape_2D[grid, block](out_1D_arr, output_Nd, N_list[0], N_list[1], 1);
    # else:
    #     reshape_3D[grid, block](out_1D_arr, output_Nd, N_list[0], N_list[1], N_list[2], 1);
    
    
    
    
def higher_order_time_vector_acoustic_func_2D(vx2, vx1, vz2, vz1, p2, p1, F_dx_op_lambda_forw, F_dz_op_lambda_forw, F_dx_op_lambda_back, F_dz_op_lambda_back, K_dx_op_lambda_forw, K_dz_op_lambda_forw, K_dx_op_lambda_back, K_dz_op_lambda_back, vel_vel, den_arr, attenuation_arr, pml_att, dt, N_length, N_list, SP_bool, cp, grid, block):    
    '''
    Usage:
        
        higher_order_time_vector_func_2D(vx2, vx1, vz2, vz1, p2, p1, F_dx_op_lambda_forw, F_dz_op_lambda_forw, F_dx_op_lambda_back, F_dz_op_lambda_back, K_dx_op_lambda_forw, K_dz_op_lambda_forw, K_dx_op_lambda_back, K_dz_op_lambda_back, vel_vel, den_arr, attenuation_arr, pml_att, dt/self.TIME_order, N_length, N_list, self.SP_bool, cp);
    '''

    
    wavefield_ND_arr1  = cp.zeros(N_list, dtype=cp.float32);
    wavefield_ND_arr3  = cp.zeros(N_list, dtype=cp.float32);
    
    ini_1D_arr       = cp.zeros(N_length, dtype=cp.float32);
    out_1D_arr       = cp.zeros(N_length, dtype=cp.float32);
    
    if not SP_bool:
        apply_1D_cupy_operator_return_ND_array(vx2, ini_1D_arr, out_1D_arr, wavefield_ND_arr1, N_length, N_list, F_dx_op_lambda_back, grid, block);
        apply_1D_cupy_operator_return_ND_array(vz2, ini_1D_arr, out_1D_arr, wavefield_ND_arr3, N_length, N_list, F_dz_op_lambda_back, grid, block);
        
    else:
        wavefield_ND_arr1[:] = K_dx_op_lambda_back(vx2); ##this is no difference 
        wavefield_ND_arr3[:] = K_dz_op_lambda_back(vz2); ##this is no difference 
        
    
    ### fwd_tp<<<>>>
    p2[:]  = p1[:]  + 1.0 * vel_vel[:] * den_arr[:] * (wavefield_ND_arr1[:] + wavefield_ND_arr3[:]) * dt;
    
    
    if not SP_bool:
        apply_1D_cupy_operator_return_ND_array(p2, ini_1D_arr, out_1D_arr, wavefield_ND_arr1, N_length, N_list, F_dx_op_lambda_forw, grid, block);
        apply_1D_cupy_operator_return_ND_array(p2, ini_1D_arr, out_1D_arr, wavefield_ND_arr3, N_length, N_list, F_dz_op_lambda_forw, grid, block);
    else:    
        wavefield_ND_arr1[:] = K_dx_op_lambda_forw(p2);
        wavefield_ND_arr3[:] = K_dz_op_lambda_forw(p2); 
    

    vx2[:] = vx1[:] +  1.0/den_arr[:] * (wavefield_ND_arr1[:]) * dt;
    vz2[:] = vz1[:] +  1.0/den_arr[:] * (wavefield_ND_arr3[:]) * dt;
        
    p1[:]  = p2[:];
    vx1[:] = vx2[:];
    vz1[:] = vz2[:];
    
    if pml_att!=0:
        p1[:]  =  p1[:] * attenuation_arr;
        p2[:]  =  p2[:] * attenuation_arr;
        vx1[:] = vx1[:] * attenuation_arr;
        vx2[:] = vx2[:] * attenuation_arr;
        
        
        vz1[:] = vz1[:] * attenuation_arr;
        vz2[:] = vz2[:] * attenuation_arr;
        
    ###just used for loop
        
    
    
    
def higher_order_time_vector_acoustic_func_3D(vx2, vx1, vy2, vy1, vz2, vz1, p2, p1, F_dx_op_lambda_forw, F_dy_op_lambda_forw, F_dz_op_lambda_forw, F_dx_op_lambda_back, F_dy_op_lambda_back, F_dz_op_lambda_back, K_dx_op_lambda_forw, K_dy_op_lambda_forw, K_dz_op_lambda_forw, K_dx_op_lambda_back, K_dy_op_lambda_back, K_dz_op_lambda_back, vel_vel, den_arr, attenuation_arr, pml_att, dt, N_length, N_list, SP_bool, cp, grid, block):    
    '''
    Usage:
    higher_order_time_vector_func_3D(vx2, vx1, vy2, vy1, vz2, vz1, p2, p1, F_dx_op_lambda_forw, F_dy_op_lambda_forw, F_dz_op_lambda_forw, F_dx_op_lambda_back, F_dy_op_lambda_back, F_dz_op_lambda_back, K_dx_op_lambda_forw, K_dy_op_lambda_forw, K_dz_op_lambda_forw, K_dx_op_lambda_back, K_dy_op_lambda_back, K_dz_op_lambda_back, vel_vel, den_arr, attenuation_arr, pml_att, dt/self.TIME_order, N_length, N_list, self.SP_bool, cp);
    '''

    
    wavefield_ND_arr1  = cp.zeros(N_list, dtype=cp.float32);
    wavefield_ND_arr2  = cp.zeros(N_list, dtype=cp.float32);
    wavefield_ND_arr3  = cp.zeros(N_list, dtype=cp.float32);
    
    ini_1D_arr       = cp.zeros(N_length, dtype=cp.float32);
    out_1D_arr       = cp.zeros(N_length, dtype=cp.float32);
    
    if not SP_bool:
        apply_1D_cupy_operator_return_ND_array(vx2, ini_1D_arr, out_1D_arr, wavefield_ND_arr1, N_length, N_list, F_dx_op_lambda_back, grid, block);
        apply_1D_cupy_operator_return_ND_array(vy2, ini_1D_arr, out_1D_arr, wavefield_ND_arr2, N_length, N_list, F_dy_op_lambda_back, grid, block);
        apply_1D_cupy_operator_return_ND_array(vz2, ini_1D_arr, out_1D_arr, wavefield_ND_arr3, N_length, N_list, F_dz_op_lambda_back, grid, block);
        
    else:
        wavefield_ND_arr1[:] = K_dx_op_lambda_back(vx2); ##this is no difference 
        wavefield_ND_arr2[:] = K_dy_op_lambda_back(vy2); ##this is no difference 
        wavefield_ND_arr3[:] = K_dz_op_lambda_back(vz2); ##this is no difference 
        
    
    ### fwd_tp<<<>>>
    p2 = update_p_from_vxvyvz_att(p1, vel_vel, den_arr, wavefield_ND_arr1, wavefield_ND_arr2, wavefield_ND_arr3, dt, attenuation_arr);
    
    
    if not SP_bool:
        apply_1D_cupy_operator_return_ND_array(p2, ini_1D_arr, wavefield_ND_arr1, N_length, N_list, F_dx_op_lambda_forw, grid, block);
        apply_1D_cupy_operator_return_ND_array(p2, ini_1D_arr, wavefield_ND_arr2, N_length, N_list, F_dy_op_lambda_forw, grid, block);
        apply_1D_cupy_operator_return_ND_array(p2, ini_1D_arr, wavefield_ND_arr3, N_length, N_list, F_dz_op_lambda_forw, grid, block);
    else:    
        wavefield_ND_arr1[:] = K_dx_op_lambda_forw(p2);
        wavefield_ND_arr2[:] = K_dy_op_lambda_forw(p2); 
        wavefield_ND_arr3[:] = K_dz_op_lambda_forw(p2); 
    

    vx2, vy2, vz2 = update_v_from_p_att_3D(vx1, vy1, vz1, reciprocal_den_arr, wavefield_ND_arr1, wavefield_ND_arr2, wavefield_ND_arr3, dt, attenuation_arr);
    
    
    p1[:], vx1[:], vy1[:], vz1[:]  =  p2[:], vx2[:], vy2[:], vz2[:];
      

  
def compute_stable_coeff_ray_mig_hessian(s_amp_d, 
                                         s_min_info, 
                                         s_max_info, 
                                         sx, sy, sz, 
                                         mig_stable_coe,
                                         mig_stable_position,
                                         c_type=0,
                                         
                                         log_file="compute_stable_coeff_log.txt"
                                         ):
    '''
    Compute stablized coefficient for ray migration/Hessian using the division imaging condition, 
    Prior to migration, I can compute these and save into dictionary.
    
    return stable_coe
    '''
    if c_type==0:
        
        pos_x = sx - s_min_info[0]
        pos_y = sy - s_min_info[1]
        
        if mig_stable_position>s_amp_d.shape[0]:
            pos_z = s_amp_d.shape[0] - 1 ;
        else:
            pos_z = mig_stable_position;


        WR.write_txt(log_file, f"pos_z={pos_z}, Type of pos_z: {type(pos_z)}, dtype: {getattr(pos_z, 'dtype', None)} for stable_coeff", print_bool=False)
        WR.write_txt(log_file, f"pos_y={pos_y}, Type of pos_y: {type(pos_y)}, dtype: {getattr(pos_y, 'dtype', None)} for stable_coeff", print_bool=False)
        WR.write_txt(log_file, f"pos_x={pos_x}, Type of pos_x: {type(pos_x)}, dtype: {getattr(pos_x, 'dtype', None)} for stable_coeff", print_bool=False)
        
        pos_z = TF.array_to_int32(pos_z)
        pos_y = TF.array_to_int32(pos_y)
        pos_x = TF.array_to_int32(pos_x)


        if len(s_amp_d.shape) ==2: 
            stable_coeff = mig_stable_coe * s_amp_d[pos_z, pos_x];
        
        else:
            stable_coeff = mig_stable_coe * s_amp_d[pos_z, pos_y, pos_x];
    
        
        WR.write_txt(log_file, f"stable_coeff={stable_coeff}")
    
    
    # Ensure stable_coeff is a scalar regardless of input type
    if hasattr(stable_coeff, "item"):
        return stable_coeff.item()  # For NumPy or CuPy arrays
    else:
        return float(stable_coeff)  # For other numerical types





def find_range_num_in_array(x_min, x_max, gridx):
    """
    计算 gridx 数组中有多少个值落在 [x_min, x_max] 范围内，
    并返回最接近 x_min 的第一个 gridx id number，同时保留在该范围内的值。
    
    参数：
    - x_min: 范围的最小值
    - x_max: 范围的最大值
    - gridx: x 方向的网格数组
    
    返回：
    - count: 落在范围 [x_min, x_max] 的值的数量
    - x_min_idx: 最接近 x_min 的第一个 gridx id number
    - gridx_in_range_vals: 落在范围内的 gridx 值
    """
    # Check if arr is a list or numpy array, otherwise raise an error
    if not isinstance(gridx, (list, np.ndarray, cp.ndarray)):
        raise TypeError("Input 'arr' must be a list or numpy array.")
        
    # 转换为 CuPy 数组
    gridx = cp.asarray(gridx)
    
    # 找到落在 [x_min, x_max] 范围内的值
    gridx_in_range = (gridx >= x_min) & (gridx <= x_max)
    
    # 计算有多少个值落在范围内
    count = cp.sum(gridx_in_range)
    
    # 保留在范围内的值
    gridx_in_range_vals = gridx[gridx_in_range]
    
    # in gridx_in_range_vals 找到最接近 x_min 的值 id number
    x_min_idx = cp.abs(gridx_in_range_vals - x_min).argmin()
    
    # 返回结果并将 CuPy 数组转换为标准 Python 类型
    return cp.int32(count.get()), cp.int32(x_min_idx.get()) ###, gridx_in_range_vals.get()


def find_nearest_idx_and_value_nd(arr, val):
    """
    Find the indices of the nearest value in an N-dimensional array `arr` to the given value `val`,
    and return both the indices and the corresponding value.
    """
    
    # Check if arr is a list or numpy array, otherwise raise an error
    if not isinstance(arr, (list, np.ndarray)):
        raise TypeError("Input 'arr' must be a list or numpy array.")
        
    # Ensure that `arr` is a numpy array
    arr = np.asarray(arr)
    
    # Flatten the array and find the index of the closest value
    flat_idx = np.abs(arr - val).argmin()  # This gives the index in the flattened array
    
    if arr.ndim == 1:
        # If it's a 1D array, return the flat index and the value directly
        nearest_value = arr[flat_idx]
        return flat_idx, nearest_value
    
    # Convert the flat index back to N-dimensional indices
    nd_idx = np.unravel_index(flat_idx, arr.shape)
    
    # Get the value at the found index
    nearest_value = arr[nd_idx]
    
    return nd_idx, nearest_value
    

###########all function for image-domain inversion, I must hessian_ as key word  for evoking
###########all function for image-domain inversion, I must hessian_ as key word  for evoking
###########all function for image-domain inversion, I must hessian_ as key word  for evoking


def hessian_extension_boundary_psf(ini_obj1, func_time=True, debug=False):
    '''
    This function aims to extend the hessian and angle_hessian
    '''
    
    wx1_half_id, _ = find_nearest_idx_and_value_nd(ini_obj1.gridx, ini_obj1.wx//2+3)
    wy1_half_id, _ = find_nearest_idx_and_value_nd(ini_obj1.gridy, ini_obj1.wy//2+3)
    wz1_half_id, _ = find_nearest_idx_and_value_nd(ini_obj1.gridz, ini_obj1.wz//2+3)
    
    wx2_half_id, _ = find_nearest_idx_and_value_nd(ini_obj1.gridx, ini_obj1.nx - ini_obj1.wx//2 - 3)
    wy2_half_id, _ = find_nearest_idx_and_value_nd(ini_obj1.gridy, ini_obj1.ny - ini_obj1.wy//2 - 3)
    wz2_half_id, _ = find_nearest_idx_and_value_nd(ini_obj1.gridz, ini_obj1.nz - ini_obj1.wz//2 - 3)
    
    def hessian_replace_func(hessian, i1, x1, j1, y1, k1, z1):
        
        if x1 < ini_obj1.wx//2:
            hessian[...,k1:k1+1, j1:j1+1, i1:i1+1, :, :, :] = hessian[...,k1:k1+1, j1:j1+1, wx1_half_id:wx1_half_id+1, :, :, :]
        if x1 > ini_obj1.nx - ini_obj1.wx//2:
            hessian[...,k1:k1+1, j1:j1+1, i1:i1+1, :, :, :] = hessian[...,k1:k1+1, j1:j1+1, wx2_half_id:wx2_half_id+1, :, :, :]
        
        
        
        if y1 < ini_obj1.wy//2:
            hessian[...,k1:k1+1, j1:j1+1, i1:i1+1, :, :, :] = hessian[...,k1:k1+1, wy1_half_id:wy1_half_id+1, i1:i1+1, :, :, :]
        if y1 > ini_obj1.ny - ini_obj1.wy//2:
            hessian[...,k1:k1+1, j1:j1+1, i1:i1+1, :, :, :] = hessian[...,k1:k1+1, wy2_half_id:wy2_half_id+1, i1:i1+1, :, :, :]
        
        
        
        if z1 < ini_obj1.wz//2:
            hessian[...,k1:k1+1, j1:j1+1, i1:i1+1, :, :, :] = hessian[...,wz1_half_id:wz1_half_id+1, j1:j1+1, i1:i1+1, :, :, :]
        if z1 > ini_obj1.nz - ini_obj1.wz//2:
            hessian[...,k1:k1+1, j1:j1+1, i1:i1+1, :, :, :] = hessian[...,wz2_half_id:wz2_half_id+1, j1:j1+1, i1:i1+1, :, :, :]
            
    
    
    for i1, x1 in enumerate(ini_obj1.gridx):
        for j1, y1 in enumerate(ini_obj1.gridy):
            for k1, z1 in enumerate(ini_obj1.gridz):
                
                if hasattr(ini_obj1, 'hessian1'):
                    hessian_replace_func(ini_obj1.hessian1, i1, x1, j1, y1, k1, z1);
                
                if hasattr(ini_obj1, 'hessian2'):
                    hessian_replace_func(ini_obj1.hessian2, i1, x1, j1, y1, k1, z1);
                    
                if hasattr(ini_obj1, 'hessian3'):
                    hessian_replace_func(ini_obj1.hessian3, i1, x1, j1, y1, k1, z1);
                    
                if hasattr(ini_obj1, 'hessian4'):
                    hessian_replace_func(ini_obj1.hessian4, i1, x1, j1, y1, k1, z1);
                    
                '''angle_hessian '''
                if hasattr(ini_obj1, 'angle_hessian1'):
                    hessian_replace_func(ini_obj1.angle_hessian1, i1, x1, j1, y1, k1, z1);
                    
                if hasattr(ini_obj1, 'angle_hessian2'):
                    hessian_replace_func(ini_obj1.angle_hessian2, i1, x1, j1, y1, k1, z1);
    
    

def hessian_transfer_data_nearest_grid(ini_obj1, out_obj2, func_time=True, debug=False):
    """
    Transfer hessian data from ini_obj1 to out_obj2 based on the nearest grid points in ini_obj1's grid.
    
    Note that we will modify the value of out_obj2.gridx, out_obj2.gridy, out_obj2.gridz based on the nearest_grid from ini_obj1.gridx, ini_obj1.gridy, ini_obj1.gridz
    """
    if func_time:
        t_start = time.perf_counter();
    
    if ini_obj1.wz < out_obj2.wz or ini_obj1.wy < out_obj2.wy or ini_obj1.wx < out_obj2.wx:
        raise ValueError(f"{ini_obj1.wz} < {out_obj2.wz} or {ini_obj1.wy} < {out_obj2.wy} or {ini_obj1.wx} < {out_obj2.wx}" );
    
    if ini_obj1.numz < out_obj2.numz or ini_obj1.numy < out_obj2.numy or ini_obj1.numx < out_obj2.numx:
        print(f"Warning: { ini_obj1.numz} < {out_obj2.numz} or { ini_obj1.numy} < {out_obj2.numy} or { ini_obj1.numx} < {out_obj2.numx}" );
        
    if ini_obj1.angle_num < out_obj2.angle_num or ini_obj1.angle_start > out_obj2.angle_start or ini_obj1.angle_interval != out_obj2.angle_interval:
        raise ValueError(f"ini_obj1.angle_num < out_obj2.angle_num or ini_obj1.angle_start > out_obj2.angle_start or ini_obj1.angle_interval != out_obj2.angle_interval" );


    if ini_obj1.use_cupy == True:
        print(f"Warning: ini_obj1.use_cupy={ini_obj1.use_cupy}" );
    if out_obj2.use_cupy == True:
        print(f"Warning: out_obj2.use_cupy={out_obj2.use_cupy}" );
    
    
    wz_beg = int( (ini_obj1.wz - out_obj2.wz)//2 );
    wy_beg = int( (ini_obj1.wy - out_obj2.wy)//2 );
    wx_beg = int( (ini_obj1.wx - out_obj2.wx)//2 );
    
    w_z_slice = slice(wz_beg, wz_beg+out_obj2.wz);
    w_y_slice = slice(wy_beg, wy_beg+out_obj2.wy);
    w_x_slice = slice(wx_beg, wx_beg+out_obj2.wx);
    
    if debug:
        print(f"w_z_slice={w_z_slice}");
        print(f"w_y_slice={w_y_slice}");
        print(f"w_x_slice={w_x_slice}");
        
    # Loop through out_obj2's grid and find the nearest points in ini_obj1's grid
    for i2, x2 in enumerate(out_obj2.gridx):
        
        # Find the closest index in ini_obj1.gridx
        i1, gridx_value = find_nearest_idx_and_value_nd(ini_obj1.gridx, x2)  
        out_obj2.gridx[i2]  = gridx_value

        if debug:
            print(f"i1={i1},gridx_value={gridx_value}");

        for j2, y2 in enumerate(out_obj2.gridy):
            
            # Find the closest index in ini_obj1.gridy
            j1, gridy_value = find_nearest_idx_and_value_nd(ini_obj1.gridy, y2)  
            out_obj2.gridy[j2]  = gridy_value
            
            if debug:
                print(f"j1={j1},gridy_value={gridy_value}");

            for k2, z2 in enumerate(out_obj2.gridz):
                
                # Find the closest index in ini_obj1.gridz
                k1, gridz_value = find_nearest_idx_and_value_nd(ini_obj1.gridz, z2)  
                out_obj2.gridz[k2]  = gridz_value
                
                if debug:
                    print(f"k1={k1},gridz_value={gridz_value}");

                # Now copy the hessian values from ini_obj1 to out_obj2 at the found indices
                if hasattr(ini_obj1, 'hessian1') and hasattr(out_obj2, 'hessian1'):
                    # print(f"{out_obj2.hessian1[k2:k2+1, j2:j2+1, i2:i2+1, :, :, :].shape}")
                    # print(f"{ini_obj1.hessian1[k1:k1+1, j1:j1+1, i1:i1+1, w_z_slice, w_y_slice, w_x_slice].shape}")
                    out_obj2.hessian1[k2:k2+1, j2:j2+1, i2:i2+1, :, :, :] = ini_obj1.hessian1[k1:k1+1, j1:j1+1, i1:i1+1, w_z_slice, w_y_slice, w_x_slice]
                else:
                    print("There is no Hessian1")
                
                    
                if hasattr(ini_obj1, 'hessian2') and hasattr(out_obj2, 'hessian2'):
                    out_obj2.hessian2[k2:k2+1, j2:j2+1, i2:i2+1, :, :, :] = ini_obj1.hessian2[k1:k1+1, j1:j1+1, i1:i1+1, w_z_slice, w_y_slice, w_x_slice]
                else:
                    print("There is no Hessian2")
                    
                    
                if hasattr(ini_obj1, 'hessian3') and hasattr(out_obj2, 'hessian3'):
                    out_obj2.hessian3[k2:k2+1, j2:j2+1, i2:i2+1, :, :, :] = ini_obj1.hessian3[k1:k1+1, j1:j1+1, i1:i1+1, w_z_slice, w_y_slice, w_x_slice]
                else:
                    print("There is no Hessian3")
                    
                    
                if hasattr(ini_obj1, 'hessian4') and hasattr(out_obj2, 'hessian4'):
                    out_obj2.hessian4[k2:k2+1, j2:j2+1, i2:i2+1, :, :, :] = ini_obj1.hessian4[k1:k1+1, j1:j1+1, i1:i1+1, w_z_slice, w_y_slice, w_x_slice]
                else:
                    print("There is no Hessian4")


                '''get angle_hessian1 and angle_hessian2'''
                if hasattr(ini_obj1, 'angle_hessian1') and hasattr(out_obj2, 'angle_hessian1'):
                    ini_obj1_index  = np.abs(      np.int32(  (ini_obj1.angle_start - out_obj2.angle_start)/ini_obj1.angle_interval  )       );
                    angle_num       = out_obj2.angle_num;
                    
                    out_obj2.angle_hessian1[0:angle_num, k2:k2+1, j2:j2+1, i2:i2+1, :, :, :] = ini_obj1.angle_hessian1[ini_obj1_index:ini_obj1_index+angle_num, k1:k1+1, j1:j1+1, i1:i1+1, w_z_slice, w_y_slice, w_x_slice]
                
                if hasattr(ini_obj1, 'angle_hessian2') and hasattr(out_obj2, 'angle_hessian2'):
                    ini_obj1_index  = np.abs(      np.int32(  (ini_obj1.angle_start - out_obj2.angle_start)/ini_obj1.angle_interval  )       );
                    angle_num       = out_obj2.angle_num;
                    
                    out_obj2.angle_hessian2[0:angle_num, k2:k2+1, j2:j2+1, i2:i2+1, :, :, :] = ini_obj1.angle_hessian2[ini_obj1_index:ini_obj1_index+angle_num, k1:k1+1, j1:j1+1, i1:i1+1, w_z_slice, w_y_slice, w_x_slice]

    if debug:
        print("hessian_transfer_data_nearest_grid is complete.")

    if func_time:
        t_end = time.perf_counter();
        if debug:
            print("Time for hessian_transfer_data_nearest_grid {} s\n".format( t_end-t_start) );
    


def hessian_6D_hessian_to_3D_PSF(hessian1, gridx, gridy, gridz):
    """
    Convert a 6D Hessian array to a 3D PSF array by arranging the grid points and expanding
    the small grids (wz, wy, wx) into a larger 3D array.
    
    Parameters:
    - hessian1: 6D numpy array with shape [numz, numy, numx, wz, wy, wx]
    - gridx, gridy, gridz: Grid points for x, y, and z directions respectively.
    
    Returns:
    - 3D PSF array with shape [numz*wz, numy*wy, numx*wx]
    """
    # Check the shape of the original 6D Hessian array
    hessian_shape = hessian1.shape
    if len(hessian_shape) != 6:
        raise ValueError("hessian1 must be a 6D array")

    # Extract the dimensions from the shape of the Hessian array
    numz, numy, numx, wz, wy, wx = hessian_shape

    # Reshape the Hessian into a 3D array of shape [numz*wz, numy*wy, numx*wx]
    psf_3d = hessian1.reshape(numz, numy, numx, wz, wy, wx).transpose(0, 3, 1, 4, 2, 5).reshape(numz * wz, numy * wy, numx * wx)
    
    print(f"Converted 6D Hessian to 3D PSF with shape: {psf_3d.shape}")

    return psf_3d


#################FD function
#################FD function
#################FD function
#################FD function
#################FD function
#################FD function
#################FD function
#################FD function
def FD_second_derivative_coeff(method=0, order=10, dtype=np.float32):
    '''
    method=0:   tayler expansion for coefficent, references: Determination of finite-difference weights using scaled binomial windows, Chunlei Chu 1 and Paul L. Stoffa 2
    
    method=1:   zhang and Yao (2009), geophysics    
    
    method=2:   Reducing error accumulation of optimized finite-difference scheme using the minimum norm, Zhongzheng Miao 1 and Jinhai Zhang 2, GEOPHYSICS, VOL. 85, NO. 5 (SEPTEMBER-OCTOBER 2020); P  T275–T291
    '''
    if order % 2 !=0:
        print("Error: The order is not even.")
        sys.exit(1)
    
    coeff_arr = np.zeros(order//2+1, dtype=np.float32);
    
    if method == 0:
        
        for n in range(1, order//2+1):
            coe1         = comb(order, order//2+n);
            coe2         = comb(order, order//2);
            W_n_N        = coe1/coe2;
            coeff_arr[n] = -2/n/n * np.cos(n*np.pi) * W_n_N;
        
        coeff_arr[0] = 0 - 2*np.sum(coeff_arr);
    
    if method == 1:
        if order== 2:
            coeff_arr[1] = +1
        
        if order== 4:
            coeff_arr[1] = +1.37106192
            coeff_arr[2] = -0.09322459         

        
        elif order==6:
            coeff_arr[1] = +1.57500756
            coeff_arr[2] = -0.18267338
            coeff_arr[3] = +0.01742643
        
        elif order==8:
            coeff_arr[1] = +1.70507669
            coeff_arr[2] = -0.25861812
            coeff_arr[3] = +0.04577745 
            coeff_arr[4] = -0.00523630
        
        elif order==10:
            coeff_arr[1] = +1.77642739
            coeff_arr[2] = -0.30779013
            coeff_arr[3] = +0.07115999   
            coeff_arr[4] = -0.01422784
            coeff_arr[5] = +0.00168305
        
        elif order==12:
            coeff_arr[1] = +1.83730507
            coeff_arr[2] = -0.35408741
            coeff_arr[3] = +0.09988277
            coeff_arr[4] = -0.02817135
            coeff_arr[5] = +0.00653900
            coeff_arr[6] = -0.00092547
            
        elif order==14:
            coeff_arr[1] = +1.87636137
            coeff_arr[2] = -0.38612121
            coeff_arr[3] = +0.12263042
            coeff_arr[4] = -0.04190565
            coeff_arr[5] = +0.01330243
            coeff_arr[6] = -0.00344731
            coeff_arr[7] = +0.00055985
            
        elif order==16:
            coeff_arr[1] = +1.89789462
            coeff_arr[2] = -0.40456799
            coeff_arr[3] = +0.13676734
            coeff_arr[4] = -0.05150324
            coeff_arr[5] = +0.01893502
            coeff_arr[6] = -0.00619345
            coeff_arr[7] = +0.00159455
            coeff_arr[8] = -0.00020980
        
        else:
            print("order must be less 16")
            sys.exit(1)
        
        coeff_arr[0] = 0 - 2*np.sum(coeff_arr)
        
    
    if method == 2:
        
        if order== 2:
            coeff_arr[1] = +1
            
        if order== 4:
            coeff_arr[1] = +1.36849466
            coeff_arr[2] = -0.09250358            

        
        elif order==6:
            coeff_arr[1] = +1.57002351
            coeff_arr[2] = -0.18029126
            coeff_arr[3] = +0.01689178
        
        elif order==8:
            coeff_arr[1] = +1.69500160
            coeff_arr[2] = -0.25237286
            coeff_arr[3] = +0.04318410   
            coeff_arr[4] = -0.00467736
        
        elif order==10:
            coeff_arr[1] = +1.77605293
            coeff_arr[2] = -0.30760563
            coeff_arr[3] = +0.07113517   
            coeff_arr[4] = -0.01428477
            coeff_arr[5] = +0.00173138
        
        elif order==12:
            coeff_arr[1] = +1.83028226
            coeff_arr[2] = -0.34862082
            coeff_arr[3] = +0.09635125
            coeff_arr[4] = -0.02636447
            coeff_arr[5] = +0.00586305
            coeff_arr[6] = -0.00078397
            
        elif order==14:
            coeff_arr[1] = +1.86784273
            coeff_arr[2] = -0.37905046
            coeff_arr[3] = +0.11750996
            coeff_arr[4] = -0.03872739
            coeff_arr[5] = +0.01167996
            coeff_arr[6] = -0.00281204
            coeff_arr[7] = +0.00041066
            
        elif order==16:
            coeff_arr[1] = +1.89465931
            coeff_arr[2] = -0.40181229
            coeff_arr[3] = +0.13471016
            coeff_arr[4] = -0.05014533
            coeff_arr[5] = +0.01819832
            coeff_arr[6] = -0.00588168
            coeff_arr[7] = +0.00151336
            coeff_arr[8] = -0.00023904
            
        else:
            print("order must be less 16")
            sys.exit(1)
            
        
        coeff_arr[0] = 0 - 2*np.sum(coeff_arr)
        
    return coeff_arr
            



 
def FD_first_derivative_coeff_stagger(method=0, order=10, dtype=np.float32):
    '''
    method=0:   tayler expansion for coefficent, references: Determination of finite-difference weights using scaled binomial windows, Chunlei Chu 1 and Paul L. Stoffa 2
    method=1:   
    method=2:   
    '''
    if order % 2 !=0:
        print("Error: The order is not even.")
        sys.exit(1)  # 1 是一个非零的退出状态码，表示异常退出
    
    coeff_arr = np.zeros(order//2+1, dtype=np.float32);
    
    if method == 0:
        if   order == 2:
            coeff_arr[1] = 1;
        elif order == 4:
            coeff_arr[1] = +1.0*9/8;
            coeff_arr[2] = -1.0 /24;
        else:
            for n in range(1, order//2 +1 ):
                coe1         = (order - 1) / np.power(2, 2*order-3);
                coe2         = comb(order-3, (order-2)//2);
                coe3         = comb(order-1, order//2-1+n);
                W_n_N        = coe1*coe2*coe3;
                s1           = -1.0 * np.sin((1/2 - n) * np.pi);
                s2           = np.power(0.5-n, 2);
                
                coeff_arr[n] = s1/s2*W_n_N;
        
    return coeff_arr
    
    



def FD_first_derivative_coeff_standard(method=0, order=10, dtype=np.float32):
    '''
    method=0:   tayler expansion for coefficent, references: Determination of finite-difference weights using scaled binomial windows, Chunlei Chu 1 and Paul L. Stoffa 2
    method=1:   
    method=2:   
    '''
    if order % 2 !=0:
        print("Error: The order is not even.")
        sys.exit(1)  # 1 是一个非零的退出状态码，表示异常退出
    
    coeff_arr = np.zeros(order//2+1, dtype=np.float32);
    
    if method == 0:
        
        for n in range(1, order//2 +1 ):
            coe1         = comb(order, order//2+n);
            coe2         = comb(order, order//2);
            W_n_N        = coe1/coe2;
            coeff_arr[n] = -1/n * np.cos(n*np.pi) * W_n_N;
        
    return coeff_arr


def compute_C_nk(n, k ):
    
    C_nk = np.power(-1, k)/k * comb(n-k-1, k-1) * np.power(2.0 , (n-2*k-1)  ) ; 
    return C_nk

# C_nk = compute_C_nk(5, 2 );
# print(C_nk)




def FD_compute_coefficients(n):
    coefficients = []
    for k in range(0, n + 1):  # 从 x^1 到 x^max_degree
        # 计算二项式系数 binom(n, k)
        binom_coeff = math.comb(n, k)
        # 计算 (-1/2)^k
        term_coeff = ((-1/2) ** k)
        # 计算最终的系数
        coeff = binom_coeff * term_coeff
        coefficients.append(coeff)
    return coefficients

# # 指定 n 和 max_degree 的值
# n = 5  
# # 计算 x^1, x^2, ..., x^max_degree 前面的系数
# coefficients = compute_coefficients(n)

# print(coefficients) #[1.0, -2.5, 2.5, -1.25, 0.3125, -0.03125]


def FD_pad_lists_to_same_length(list1, list2):
    # 创建原始列表的副本
    new_list1 = list1.copy()
    new_list2 = list2.copy()

    # 找到两个列表的最大长度
    max_length = max(len(new_list1), len(new_list2))
    
    # 使用0填充较短的列表
    new_list1.extend([0] * (max_length - len(new_list1)))
    new_list2.extend([0] * (max_length - len(new_list2)))
    
    return new_list1, new_list2



def FD_multiple_angle_coeff(method=0, order=10, dtype=np.float32):
    '''
    Time-stepping wave-equation solution for seismic modeling using a multiple-angle formula and the Taylor expansion, GEOPHYSICS, VOL. 84, NO. 4 (JULY-AUGUST 2019); P  T299–T311, 10 FIGS., 9 TABLES. 10.1190/GEO2018-0463.1  
    '''
    
    coeff_arr = np.zeros(order+1, dtype=np.float32);

        
    if method==0:
        
        if   order == 1:
            coeff_arr[1] = 1;
            
        elif order == 2:
            coeff_arr[1] = +4;
            coeff_arr[2] = +1;
        
        elif order == 3:
            coeff_arr[1] = +9;
            coeff_arr[2] = +6;
            coeff_arr[3] = +1;
            
        elif order == 4:
            coeff_arr[1] = +16;
            coeff_arr[2] = +20;
            coeff_arr[3] = +8;
            coeff_arr[4] = +1;
            
        elif order == 5:
            coeff_arr[1] = +25;
            coeff_arr[2] = +50;
            coeff_arr[3] = +35;
            coeff_arr[4] = +10;
            coeff_arr[5] = +1;
            
        elif order == 6:
            coeff_arr[1] = +36;
            coeff_arr[2] = +105;
            coeff_arr[3] = +112;
            coeff_arr[4] = +54;
            coeff_arr[5] = +12;
            coeff_arr[6] = +1;
            
        elif order == 7:
            coeff_arr[1] = +49;
            coeff_arr[2] = +196;
            coeff_arr[3] = +294;
            coeff_arr[4] = +210;
            coeff_arr[5] = +77;
            coeff_arr[6] = +14;
            coeff_arr[7] = +1;
            
        elif order == 8:
            coeff_arr[1] = +64;
            coeff_arr[2] = +336;
            coeff_arr[3] = +672;
            coeff_arr[4] = +660;
            coeff_arr[5] = +352;
            coeff_arr[6] = +104;
            coeff_arr[7] = +16;
            coeff_arr[8] = +1;
        
        # else:
        #     print("order must be less 8")
        #     sys.exit(1)
            
    if method==1:
        ''' equation 10 and inserting equation 28 into equation 20'''
        ''' equation 9 in 2019, geophysics  '''
        if   order == 1:
            coeff_arr[1] = 1;
            
        elif order == 2:
            coeff_arr[1] = +4;
            coeff_arr[2] = +1;
            
        elif order == 3:
            coeff_arr[1] = +9;
            coeff_arr[2] = +6;
            coeff_arr[3] = +1;
        
        elif order >=3:
            
            n    = order;
            
            Hnn  = [coeff * np.power(2, n-1) for coeff in FD_compute_coefficients(n)]

            zero_list = [0] * len(Hnn)
            # print(Hnn)
            
            for k in range(1, n//2):
                
                C_nk = compute_C_nk(n, k);
                
                H_n_n_2k = [ n * C_nk * coeff for coeff in FD_compute_coefficients(n  - 2*k)]
        
                list1, list2 = FD_pad_lists_to_same_length(Hnn, H_n_n_2k);
                
                zero_list = [a + b for a, b in zip(zero_list, list2)]
                
                # print("H_n_n_2k is ", H_n_n_2k)
                # print("list1 is ", list1)
                # print("list2 is ", list2)
                # print("zero_list is ", zero_list)
        
            result    = [a + b for a, b in zip(list1, zero_list)]
            
            
            
            coeff_arr = 2*np.asarray(result, dtype=np.float32)
            
            # print("coeff_arr is ", coeff_arr)
        
    return np.abs(coeff_arr)
    
    
# angle_coeff = multiple_angle_coeff(method=1, order=10, dtype=np.float32)
# print(angle_coeff)
def FD_Lie_produce_coeff(method=0, order=10, dtype=np.float32):
    '''
    Lie_produce_coeff
    '''
    
    coeff_arr = np.zeros(order+1, dtype=np.float32);


    for k in range(0, order+1):
        coeff_arr[k] = comb(order, k);
        
        
    return coeff_arr

# # Example usage
# coefficients = Lie_produce_coeff(order=1, dtype=np.float32)
# print(coefficients)


def analytic_green_func(f, dt, c, distance, dims=2, correct_bool=False, correct_sign=+1, forward=True):
    """
        f(w):     band-limit source signal
        c:        velocity
        distance:
        2D correct: sqrt(jw), I have correted this wavelet
    """
    
    if isinstance(f, cp.ndarray):
        f = f;
    else:
        f = cp.asarray(f)
    
    time_shift = distance /c;
    
    if dims==2:
        #####I assume that, I have corrected the source signal using sqrt(iw);
        w_arr  = 2 * cp.pi * cp.fft.fftfreq(len(f), d=dt);
        
        if correct_bool:
            f2     = apply_sqrt_iw_operator(f, 0, dt, correct_sign=correct_sign, forward=forward, module=cp);
            output = cp.exp( -1j*  w_arr * time_shift ) * cp.fft.fft(f2, axis=0);
        else:
            output = cp.exp( -1j* w_arr * time_shift ) * cp.fft.fft(f, axis=0);

        amplitude_term  = 1.0/4.0 * cp.sqrt(2*c/distance/cp.pi) ;
    
        output2 = amplitude_term * cp.real(cp.fft.ifft (output, axis=0) )
        
    if dims==3:
        
        w_arr  = 2 * cp.pi * cp.fft.fftfreq(len(f), d=dt);
        
        output = cp.exp( -1j* w_arr * time_shift ) * cp.fft.fft(f, axis=0);
    
        amplitude_term  = 1.0 / 4.0 / cp.pi/ distance;
    
        output2 = amplitude_term * cp.real(cp.fft.ifft (output, axis=0) )
        
    return output2




def dict_wave_tapering_func(
                       vp_arr,
                       
                       model_interval=[10, 10],
                       
                       dt_lt_list = [0.001, 1000],  
                       
                       model_info = [],
                       
                       source_position_list=[0, 0, 0],
                       
                       receiver_position_list=[0, 0, 0],

                       wavelet_ns=300,
                       
                       dims=2,
                       
                       tapering_dims=[3000, 300],
                       
                       module=cp,
                       ):
    
    dt, lt = dt_lt_list
    
    tapering_matrix = cp.ones(tuple(tapering_dims), dtype=cp.float32);
    
    '''Check if vp_arr is a scalar'''
    if not hasattr(vp_arr, 'shape'):
        constant_vel = vp_arr;
    else:
        if   dims==2:
            sz   =  source_position_list[1] - model_info[1][2]
            # sy =
            sx   =  source_position_list[0] - model_info[1][0]
            
            constant_vel = vp_arr[sz, sx ];
        else:
            sz   =  source_position_list[2] - model_info[1][2]
            sy   =  source_position_list[1] - model_info[1][1]
            sx   =  source_position_list[0] - model_info[1][0]
            
            constant_vel = vp_arr[sz, sy, sx ];
    
    
    dt_arr = cp.zeros(list(receiver_position_list[0].shape), dtype=cp.float32);
    
    for g, s, interval in zip(receiver_position_list, source_position_list, model_interval):
        
        dt_arr[:] = dt_arr[:] + (g-s) * (g-s) * interval* interval;

    dt_arr = module.sqrt(   dt_arr ) / constant_vel  /dt + wavelet_ns;
    
    
    for idx in range(0, tapering_matrix.shape[1]):
        
        tapering_matrix[0:int(dt_arr[idx]), idx  ] = 0.0;
        # print(dt_arr[idx]);

    return tapering_matrix


def dict_wave_tapering_func_test_func():
    vp_arr          = 3000*cp.ones((200,300), dtype=cp.float32) 
    
    model_interval  = [10, 10]
    dt_lt_list      = [0.001, 1000]
    source_position_list          = [0, 0]
    receiver_position_list          = [
                        cp.zeros( (300,1) ).astype(cp.int32), 
                        cp.linspace(0, 300, num=300, endpoint=False).astype(cp.int32), 
                      ]
    receiver_position_list[1]       = receiver_position_list[1].reshape(-1,1)
    wavelet_ns      = 0
    dims            = 2
    tapering_dims   = [3000, 300]
    
    tapering_matrix = dict_wave_tapering(
        vp_arr=vp_arr,
        model_interval=model_interval,
        dt_lt_list=dt_lt_list,
        source_position_list=source_position_list,
        receiver_position_list=receiver_position_list,
        wavelet_ns=wavelet_ns,
        dims=dims,
        tapering_dims=tapering_dims,
        module=cp
    )
    
    PF.imshow(tapering_matrix, output_name="dict_wave_tapering.png");


###############the following code is used for inversion
###############the following code is used for inversion
###############the following code is used for inversion
###############the following code is used for inversion
###############the following code is used for inversion
def wave_obj_l2(obs_shot_d, cal_shot_d, adjoint_source_bool=False):
    '''
    This function aims to compute the L2 objective function value of obs_shot_d and cal_shot_d

    if adjoint_source_bool:
        return obj_l2, residual
    else:
        return obj_l2
    Return objective function value and adjoint source???
    '''
    module = WR.get_module_type(obs_shot_d);
    
    residual = cal_shot_d - obs_shot_d;
    obj_l2   = 0.5 * module.sum(residual * residual);
    
    if adjoint_source_bool:
        return obj_l2, residual
    else:
        return obj_l2

def wave_obj_zero_cc(obs_shot_d, cal_shot_d, adjoint_source_bool=False):
    '''
    Cross-correlation misfit function (Equation 7).
    
    Parameters:
    obs_shot_d : array_like
        Observed shot data (U_obs).
    cal_shot_d : array_like
        Calculated shot data (ΔU).
    adjoint_source_bool : bool, optional
        If True, returns the residual for adjoint source calculation.
        
    Returns:
    obj_value : float
        Objective function value (cross-correlation misfit).
    residual : array_like, optional
        Residual for adjoint source if adjoint_source_bool=True.
        
    example:    
        # Generate random CuPy arrays for testing
        obs_shot_d = cp.random.randn(100, 200).astype(cp.float32)  # Observed shot data (U_obs)
        cal_shot_d = obs_shot_d*10; #cp.random.randn(100, 200).astype(cp.float32)  # Calculated shot data (ΔU)

        # Test the function with the generated data

        adjoint_source, adjoint_source_result = wave_obj_zero_cc(obs_shot_d, cal_shot_d, adjoint_source_bool=True)



        print("Adjoint source result:", adjoint_source_result);
        print("Objective function with adjoint source result:", adjoint_source);
        
        see Zhang Yu 2015, Geophysics
            Liu Youshan 2017, GJI
            Correlative least-squares reverse time migration in viscoelastic media
Wei Zhang a,b , Jinghuai Gao a,b, ⁎ , Feipeng Li a,b , Ying Shi c , Xuan Ke c
    '''
    # Determine the module type (likely numpy or cupy)
    module = WR.get_module_type(obs_shot_d)
    
    # Calculate ΔU^2 and U_obs^2
    cal2_t_sum        = module.sum(cal_shot_d**2, axis=0)
    obs2_t_sum        = module.sum(obs_shot_d**2, axis=0)
    
    # Cross-correlation numerator
    cal_obs_t_sum     = module.sum(cal_shot_d * obs_shot_d, axis=0);
    # Cross-correlation denomerator
    norm_factor     = module.sqrt(cal2_t_sum) * module.sqrt(obs2_t_sum);

    # Compute the zero-lag cross-correlation objective
    obj_value     = -1 * module.sum( cal_obs_t_sum / norm_factor  ).item()
    
    if adjoint_source_bool:
        adjoint_source   = norm_factor * (   cal_obs_t_sum / cal2_t_sum * cal_shot_d - obs_shot_d);
        
        return obj_value, adjoint_source
    else:
        return obj_value



def wave_lag_cc_ND(obs_shot_d, cal_shot_d, tau_length, only_one_tau=False):
    '''
    compute the non-zero lag cross-correlation objective function for [lt, nx , ny, nz, nk,....]
    I will shift the obs_shot_d
    '''
    module           = WR.get_module_type(obs_shot_d);
    
    if not only_one_tau:
        
        tau_half         = np.int32(tau_length//2);
        
        lag_cc_result    = module.zeros( [tau_length, ] + list( obs_shot_d.shape[1:] ), dtype=module.float32 );
        
        # Loop over tau from -tau_length to +tau_length
        for tau in range(-tau_half, tau_half + 1):
            # Shift the observed data by tau (handling boundaries)
            if tau >= 0:
                shifted_obs = module.roll(obs_shot_d, shift=tau, axis=0)
                shifted_obs[:tau, ...] = 0  # Zero-padding at the start
            else:
                shifted_obs = module.roll(obs_shot_d, shift=tau, axis=0)
                shifted_obs[tau:, ...] = 0  # Zero-padding at the end
        
            lag_cc_result[tau + tau_half, ...] = module.sum( shifted_obs * cal_shot_d, axis=0);
   
    else:
        
        lag_cc_result    = module.zeros( [1] + list( obs_shot_d.shape[-1:] ), dtype=module.float32 );
        
        tau_shape        = tau_length.shape
        
        for index in module.ndindex( tau_shape ):
            
            tau_value = tau_length[index]
            
            obs       = obs_shot_d[:, index];
            cal       = cal_shot_d[:, index];
    
            if tau_value >= 0:
                shifted_obs = module.roll(obs, shift=tau_value, axis=0)
                shifted_obs[:tau_value, ...] = 0  # Zero-padding at the start
            else:
                shifted_obs = module.roll(obs, shift=tau_value, axis=0)
                shifted_obs[tau_value:, ...] = 0  # Zero-padding at the end
        
            lag_cc_result[0, index] = module.sum( obs * cal, axis=0);

    return lag_cc_result


def wave_shift_ND(obs_shot_d, tau_length):
    '''
        
    '''
    module          = WR.get_module_type(obs_shot_d);
    
    shifted_obs_total     = module.zeros_like(obs_shot_d, dtype=module.float32 );
        
    tau_shape       = tau_length.shape
    
    for index in module.ndindex( tau_shape ):
        
        tau_value = tau_length[index]

        obs       = obs_shot_d[:, index];

        if tau_value >= 0:
            shifted_obs = module.roll(obs, shift=tau_value, axis=0);
            shifted_obs[ :tau_value] = 0  # Zero-padding at the start
        else:
            shifted_obs = module.roll(obs, shift=tau_value, axis=0);
            shifted_obs[ tau_value:] = 0  # Zero-padding at the end

        shifted_obs_total[:, index] = shifted_obs

    return shifted_obs_total



def wave_obj_wti(obs_shot_d, cal_shot_d, adjoint_source_bool=False, dt=0.001, lag_ratio=1, epsilon=0.001):
    '''
        I will default the max tau is lag_ratio*lt, lt=obs_shot_d.shape[0]
        See  
        Wave-equation traveltime inversion, Y. Luo and G. T. Schuster, 1991

        Correlation-based reflection full-waveform inversion, Benxin Chi , Liangguo Dong , and Yuzhu Liu, 2015.
    
    '''

    module           = WR.get_module_type(obs_shot_d);
    
    tau_length       = np.int32(   obs_shot_d.shape[0] * lag_ratio   );
    
    if tau_length % 2==0:
        tau_length  = tau_length + 1; ###must be odd
    
    tau_half        = np.int32( tau_length//2 );
    
    # Ensure data is in float32 format and contiguous (on GPU if necessary)
    # obs_shot_d = module.ascontiguousarray(module.asarray(obs_shot_d, dtype=module.float32))
    # cal_shot_d = module.ascontiguousarray(module.asarray(cal_shot_d, dtype=module.float32))
    
    
    lag_cc_result    = wave_lag_cc_ND(obs_shot_d, cal_shot_d, tau_length, only_one_tau=False);
       
        
    tau_max_arr      = module.argmax(lag_cc_result, axis=0);
    
    delta_tau_arr    = tau_max_arr - tau_half;  #####the dimension is dims[1:]

    obj_value        = module.sum( delta_tau_arr * dt * delta_tau_arr * dt);
    
    
    
    if not adjoint_source_bool:
        
        return obj_value
    else:
        '''
        equation A-4, Chi 2015
        equation 8A, Luoyi, 1991
        '''
        dt_obs    = apply_iw_operator(obs_shot_d, axis=0, dt=dt, module=module);
        dtdt_obs  = apply_iw_operator(dt_obs    , axis=0, dt=dt, module=module);


        lag_cc_dtdt_dobs_cal = wave_lag_cc_ND(dtdt_obs, cal_shot_d, delta_tau_arr, only_one_tau=True);
        
        shifted_dt_obs       = wave_shift_ND(dt_obs, delta_tau_arr);
        
        
        
        delta_tau_expanded   = module.expand_dims(delta_tau_arr, axis=0);
        
        
        ##I will use the stablized division
        
        demoninator           = lag_cc_dtdt_dobs_cal * lag_cc_dtdt_dobs_cal
        if epsilon!=0:
            stablized_coeff   = module.max(demoninator).item();
        else:
            stablized_coeff   = 0;
        
        adjoint_source    = delta_tau_expanded * shifted_dt_obs * lag_cc_dtdt_dobs_cal / (demoninator + epsilon*stablized_coeff)
        
        # print(delta_tau_expanded.shape, shifted_dt_obs.shape, lag_cc_dtdt_dobs_cal.shape);
        # PF.imshow(lag_cc_dtdt_dobs_cal, output_name="./test-data/lag_cc_dtdt_dobs_cal.eps");
        # PF.imshow(obs_shot_d, output_name="./test-data/obs_shot_d.eps");
        # PF.imshow(cal_shot_d, output_name="./test-data/cal_shot_d.eps");
        # PF.imshow(lag_cc_result, output_name="./test-data/lag_cc_result.eps");
        
        # PF.imshow(dt_obs, output_name="./test-data/dt_obs.eps");
        # PF.imshow(dtdt_obs, output_name="./test-data/dtdt_obs.eps");
        
        # PF.imshow(shifted_dt_obs, output_name="./test-data/shifted_dt_obs.eps");
        
        # PF.imshow(adjoint_source, output_name="./test-data/adjoint_source.eps");
        
        # PF.imshow(cal_shot_d - obs_shot_d, output_name="./test-data/adjoint_res_source.eps");

        return obj_value, adjoint_source

# obs_shot_d = np.zeros([200, 1500], dtype=np.float32);
# cal_shot_d = np.zeros([200, 1500], dtype=np.float32);

# WR.read_file('./test-data/obs-sx-200-sy-0-sz-40-200-1500.bin', obs_shot_d, shape_list=[200, 1500]);
# WR.read_file('./test-data/cal-sx-200-sy-0-sz-40-200-1500.bin', cal_shot_d, shape_list=[200, 1500]);

# obs_shot_d = cp.asarray( obs_shot_d.T )
# cal_shot_d = cp.asarray( cal_shot_d.T )
# dt=0.001
# lag_ratio=1
# epsilon=0.001
# stablized_coeff   = 0


def generate_toplize_matrix_2D(obs_shot_d, wave_kernel_dict, flip_bool=False):
    
    module          =  WR.get_module_type(obs_shot_d);
    
    if not obs_shot_d.flags['C_CONTIGUOUS']:
        obs_shot_d  =  module.ascontiguousarray( obs_shot_d );
        print('obs_shot_d is not contious');
        
    if obs_shot_d.dtype != module.float32:
        obs_shot_d  =  obs_shot_d.astype(module.float32)
        print('obs_shot_d is not module.float32');
    
    if flip_bool:
        obs_shot_d  = module.ascontiguousarray( module.flip(obs_shot_d, axis=0) );
    

    lt, nx          = obs_shot_d.shape;
    
    
    dims     = [lt, lt, nx]
    
    top_matrix      = module.zeros(dims, dtype=obs_shot_d.dtype);

    block    = (32, 16, 1)
    
    grid = cal_grid_reverse( dims=dims, block = block, forward=False)
    
    wave_kernel_dict['cuda_form_Toeplitz_matrix_2D']( grid, block, (obs_shot_d, top_matrix, cp.int32(nx), cp.int32(lt)), );

    return top_matrix







# obs_shot_d = np.zeros([200, 1500], dtype=np.float32);
# cal_shot_d = np.zeros([200, 1500], dtype=np.float32);

# WR.read_file('./test-data/obs-sx-200-sy-0-sz-40-200-1500.bin', obs_shot_d, shape_list=[200, 1500]);
# WR.read_file('./test-data/cal-sx-200-sy-0-sz-40-200-1500.bin', cal_shot_d, shape_list=[200, 1500]);

# obs_shot_d = cp.ascontiguousarray( cp.asarray( obs_shot_d.T ) )
# cal_shot_d = cp.ascontiguousarray( cp.asarray( cal_shot_d.T ) )



# top_matrix = generate_toplize_matrix_2D(obs_shot_d, wave_kernel_dict, flip_bool=False);

# PF.imshow(top_matrix[:,:,0], output_name="./test-data/top_matrix-0.eps");
# WR.write_file("./test-data/top_matrix-0.bin", top_matrix[:,:,0]);

# source_signal           =  ricker_func(nt=1500, lt=1500, dt=0.001, freq=5, module=cp);
# phase_rotate_source_signal  =  phase_rotate_nd_array(source_signal, angle_degrees=45, axis=0, module=cp);


# PF.plot_graph([source_signal, ], output_name="./test-data/source_signal.eps");
# PF.plot_graph([phase_rotate_source_signal, ], output_name="./test-data/phase_rotate_source_signal.eps");

# top_matrix1 = generate_toplize_matrix_2D(phase_rotate_source_signal, wave_kernel_dict, flip_bool=False);
# top_matrix2 = generate_toplize_matrix_2D(source_signal, wave_kernel_dict, flip_bool=True);



# WR.write_file("./test-data/source_signal.bin", source_signal);
# WR.write_file("./test-data/phase_rotate_source_signal.bin", phase_rotate_source_signal);

# PF.imshow(top_matrix1, output_name="./test-data/top_matrix1.eps", xlabel='Times(ms)', ylabel='Times(ms)', d1=1, d2=1);
# WR.write_file("./test-data/top_matrix1.bin", top_matrix1);

# PF.imshow(top_matrix2, output_name="./test-data/top_matrix2.eps", xlabel='Times(ms)', ylabel='Times(ms)', d1=1, d2=1);
# WR.write_file("./test-data/top_matrix2.bin", top_matrix2);


def wave_awi_wiener_filter(D, p, epsilon=0.001):
    """

    """
    module          = WR.get_module_type(D);
    # 
    DtD             = module.dot(D.T, D)

    I               = epsilon * ( module.max(DtD).item() ) * module.eye(DtD.shape[0]);

    inverse_matrix  = module.linalg.inv(DtD +  I)

    Dtp             = module.dot(D.T, p)
    
    w               = module.dot(inverse_matrix, Dtp)

    del DtD, I, inverse_matrix, Dtp
    import gc
    gc.collect()

    return w


def wave_awi_weight_matrix(lt, mode=0, sigma=1.0, module=cp, dt=0.001, normalized=True):
    """
    生成一个形状为 (lt, 1) 的权重矩阵，基于指定的模式。
    
    参数:
    lt    : 时间长度，决定矩阵大小 (lt, 1)
    mode  : 模式选择:
            0 -> 以 lt/2 为中点，两端的值为 |id - lt/2|
            1 -> 以 lt/2 为中点，值为 (id - lt/2)^2
            2 -> 高斯函数，以 lt/2 为中点，往两端移动时值增大
    sigma : 高斯函数的标准差，用于模式 2
    
    返回:
    weight_matrix : 权重矩阵，形状为 (lt, 1)
    """
    
    id_array   = module.arange(lt).reshape(-1, 1)
    mid_point  = lt / 2
    
    if mode == 1:
        # mode 0: |id - lt/2|
        weight_matrix = module.abs(id_array*dt - mid_point*dt)
    
    elif mode == 2:
        # mode1: (id - lt/2)^2
        weight_matrix = (id_array*dt - mid_point*dt) ** 2
    
    elif mode == 3:
        weight_matrix = module.exp(-((id_array*dt - mid_point*dt) ** 2) / (2 * sigma ** 2))
        weight_matrix = (1 - weight_matrix ) * mid_point

    else:
        raise ValueError("Invalid mode. Choose 1, 2, or 3.")
    
    if normalized:
        weight_matrix  = weight_matrix / module.max(module.abs(weight_matrix));
    
    return weight_matrix.astype(  module.float32 )


# def wave_obj_awi(obs_shot_d, cal_shot_d, weight_func, adjoint_source_bool=False, epsilon=0.01, wave_kernel_dict=wave_kernel_dict):
#     '''
#         Adaptive waveform inversion: Practice Lluís Guasch 1 , Michael Warner 2 , and Céline Ravaut 3
#         Adaptive waveform inversion: Theory, 2015
#     '''
#     # epsilon=0.001

#     module                          = WR.get_module_type(obs_shot_d);

#     w_matrix                        = module.zeros_like(obs_shot_d);
    
#     if adjoint_source_bool:
#         adjoint_source              = module.zeros_like(obs_shot_d);

#     obj_value                       = 0;
    
#     T                               = weight_func;
#     T_square                        = T*T
    
#     for ix in range( 0, w_matrix.shape[1] ):
        
#         d        = module.ascontiguousarray( obs_shot_d[:, ix:ix+1] );
        
#         D        = generate_toplize_matrix_2D( d  , wave_kernel_dict)[:, :, 0];
        
#         p        = cal_shot_d[:, ix];
        
#         # w        = wave_awi_wiener_filter(D[:, :, 0], p, epsilon=epsilon);
        
#         DtD             = module.dot(D.T, D)

#         I               = epsilon * ( module.max(DtD).item() ) * module.eye(DtD.shape[0]);

#         inverse_matrix  = module.linalg.inv(DtD +  I);

#         Dtp             = module.dot(D.T, p);
        
#         w               = module.dot(inverse_matrix, Dtp);
        
#         w_matrix[:, ix] = w
        
        
        
#         w_l2        =  module.sum( w * w );
#         Tw          =  T * w
#         Tw_l2       =  module.sum( Tw * Tw );
        
#         f_awi       =  0.5 * (Tw_l2 / w_l2).item() ;  ###here is 0.5*, this is important
        
#         print("f_awi", f_awi)
    
#         obj_value   +=  f_awi;
    
#         if adjoint_source_bool:

#             middle_tem      = ( T_square - 2*f_awi) / w_l2;
            
#             middle_tem      = module.diag( middle_tem.flatten() );
            
#             adjoint_source[:, ix] = module.dot( D, module.dot( inverse_matrix, module.dot( middle_tem, w ) )  );  
#             # adjoint_source[:, ix] = module.dot(  module.dot(  module.dot(D, inverse_matrix ),  middle_tem ),  w );
                    
#             # print("final_result.shape is {}",final_result.shape);
            
#     if not adjoint_source_bool:
#         return obj_value, w_matrix
#     else:
#         return obj_value, w_matrix, adjoint_source


# weight_matrix1 = wave_awi_weight_matrix(1500, mode=1, sigma=1.0, module=cp, normalized=False);
# weight_matrix2 = wave_awi_weight_matrix(1500, mode=2, sigma=1.0, module=cp, normalized=False);
# weight_matrix3 = wave_awi_weight_matrix(1500, mode=3, sigma=0.05*2500, dt=0.001, module=cp, normalized=False);

# PF.plot_graph([weight_matrix1, ], output_name="./test-data/weight_matrix1.eps");
# PF.plot_graph([weight_matrix2, ], output_name="./test-data/weight_matrix2.eps");
# PF.plot_graph([weight_matrix3, ], output_name="./test-data/weight_matrix3.eps");


# obs_shot_d = np.zeros([200, 1500], dtype=np.float32);
# cal_shot_d = np.zeros([200, 1500], dtype=np.float32);

# WR.read_file('./test-data/obs-sx-200-sy-0-sz-40-200-1500.bin', obs_shot_d, shape_list=[200, 1500]);
# WR.read_file('./test-data/cal-sx-200-sy-0-sz-40-200-1500.bin', cal_shot_d, shape_list=[200, 1500]);

# obs_shot_d = cp.ascontiguousarray( cp.asarray( obs_shot_d.T ) )
# cal_shot_d = cp.ascontiguousarray( cp.asarray( cal_shot_d.T ) )

# padded_cal_shot_d = cp.pad(cal_shot_d, ((200, 0), (0, 0)), mode='constant')
    
# cal_shot_d= padded_cal_shot_d[:1500, :]

# PF.imshow(obs_shot_d, output_name="./test-data/obs_shot_d.eps", ylabel="Time(s)", d2=0.001);
# PF.imshow(cal_shot_d, output_name="./test-data/cal_shot_d.eps", ylabel="Time(s)", d2=0.001);

# obj_value, w_matrix, adjoint_source  =wave_obj_awi(obs_shot_d, cal_shot_d, weight_func = weight_matrix2, adjoint_source_bool=True, epsilon=0.01, wave_kernel_dict=wave_kernel_dict);

# PF.imshow(w_matrix, output_name="./test-data/w_matrix .eps", ylabel="Time(s)", d2=0.01);

# PF.imshow(adjoint_source, output_name="./test-data/adjoint_source.eps", ylabel="Time(s)", d2=0.01);
# WR.write_file('./test-data/adjoint_source.bin', adjoint_source.T);


# obj_value, adjoint_source_wti  =wave_obj_wti(obs_shot_d, cal_shot_d, adjoint_source_bool=True, dt=0.001, lag_ratio=1, epsilon=0.001);

# PF.imshow(adjoint_source_wti, output_name="./test-data/adjoint_source_wti.eps", ylabel="Time(s)", d2=0.01);
# WR.write_file('./test-data/adjoint_source_wti.bin', adjoint_source_wti.T);