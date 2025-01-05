

import sys
sys.path.append("/home/zhangjiwei/pyfunc/lib")

from lib_sys import *


def sign_1D_linear_taper(left: int = 1, 
                      length: int = 1, 
                      right: int = 1,
                      reverse: int = 1,
                      module=np):
    """
    Generate a linear tapering array where the center portion is 1, and both left and right sides
    taper linearly from 1 to 0.
    
    :param left: Number of elements on the left side that taper from 1 to 0
    :param length: Number of elements in the center where values are 1
    :param right: Number of elements on the right side that taper from 1 to 0
    :return: Numpy array with tapering values
    """
    # Create the taper for the left side (from 1 to 0)
    if reverse:
        left_taper = module.linspace(0, 1, left) if left > 0 else module.array([])
    else:
        left_taper = module.linspace(1, 0, left) if left > 0 else module.array([])

    # Create the middle portion, which is all ones
    middle = module.ones(length)

    # Create the taper for the right side (from 1 to 0)
    if reverse:
        right_taper = module.linspace(0, 1, right)[::-1] if right > 0 else module.array([])
    else:
        right_taper = module.linspace(1, 0, right)[::-1] if right > 0 else module.array([])

    # Concatenate the left taper, middle portion, and right taper
    taper_array = module.concatenate([left_taper, middle, right_taper])

    return taper_array


def sign_ND_linear_tapering(shape_list, tapering_list, module=np):

    tapering_matrix    = module.ones(shape_list, dtype=module.float32);
    
    dims_len           = len(shape_list)
    
    for dim in range( dims_len ):  # Loop through each dimension
        
        # Generate a taper for the current dimension
        left_overlap  = tapering_list[dim] 
        center_length = shape_list[dim]  - 2*tapering_list[dim]
        right_overlap = tapering_list[dim] 
        
        # Generate the taper array for this dimension
        
        taper = sign_1D_linear_taper(left=left_overlap, length=center_length, right=right_overlap, module=module)
    
        
        for i in range(shape_list[dim]):
            index      = [slice(None)] * dims_len
            index[dim] = i  

            tapering_matrix[tuple(index)] *= taper[i]  ###mutiply with explicitly expand

    return tapering_matrix 


def sign_ND_expand(in_array, left_shape, right_shape, padding_mode='edge', padding_value=0, module=np):
    """
    Expands the input array `in_array` to accommodate the sampling process, ensuring there is enough 
    padding around the edges for sampling to work without boundary issues.
    """
    in_shape       = in_array.shape
    
    # Calculate the padding needed in each dimension to handle the sampling region
    padding = [ (left_shape[i], right_shape[i]) for i in range(len(in_shape))  ]
    
    # Expand the array by padding each dimension with the calculated padding
    if padding_mode=='constant':
        expanded_array = module.pad(in_array, pad_width=padding, mode='constant', constant_values=padding_value)
    else:
        expanded_array = module.pad(in_array, pad_width=padding, mode='edge')
    
    return expanded_array


def sign_ND_cut(in_array, left_shape, right_shape, module=np):
    """
        cut in_array
    
        return cut_array
    """
    in_shape = in_array.shape
    slices = []

    # Create slices for each dimension, removing the left and right padding
    for i in range(len(in_shape)):
        start_idx = left_shape[i]
        end_idx   = in_shape[i] - right_shape[i]
        slices.append(slice(start_idx, end_idx))
    
    # Apply the slices to the array to cut it back to its original size
    cut_array = in_array[tuple(slices)]
    
    return cut_array

    
def sign_ND_sampling(in_array, 
                             block_ND_arr,
                             grid_list=None, 
                             sampling_matrix=None, 
                             
                             allocate_arr=False,
                             forward=True,
                             
                             module=np):
    """
    Samples the input array `in_array` based on grid points in `grid_list` and applies the `sampling_matrix`,
    assuming the array has been pre-expanded to handle boundary conditions.

    Parameters:
    - in_array: N-dimensional input array (pre-expanded).
    - grid_list: List of lists containing grid points for each dimension.
    - sampling_matrix: Matrix defining the sampling region. It can be set as ones or tapering function
    - module: Optional module (like np or cupy) for handling arrays (default is numpy).

    Returns:
    - ou_arr: The output array after sampling.
    """
    
    grid_shape     = [len(grid) for grid in grid_list]
    
    sampling_shape = list(sampling_matrix.shape);
    
    out_shape      = grid_shape + sampling_shape;
    in_shape       = [ i*j for i, j in zip(grid_shape, sampling_shape) ]
    
    ##forward, I can allocate or not allocate
    if forward:
        if allocate_arr:
            sampling_ND_array = module.zeros(out_shape, dtype=module.float32);
        else:
            sampling_ND_array = block_ND_arr
    
    ## but the adjoint, we must allocat in_array prior to the computation, since I do not want to compute shape in the code
    else:
        sampling_ND_array = block_ND_arr;
        
    
    add_list=[]
    for i, N in enumerate(sampling_shape):
        if N % 2==0:
            add_list.append(0);
        else:
            add_list.append(1);
    
    
    # Multi-dimensional index loop over grid_list
    for idx in np.ndindex(*grid_shape):
        sub_indices = []
        for i, grid in enumerate(grid_list):
            grid_point = grid[idx[i]]
            start_idx = grid_point - sampling_shape[i] // 2
            end_idx   = grid_point + sampling_shape[i] // 2 + add_list[i]
            # Assume sampling_shape is odd, ensuring a symmetric window
     
            # Get the sub-array indices for slicing
            sub_indices.append(slice(start_idx, end_idx))
     
        # Extract the sub-array from in_array using slices
        if forward:
            
            sub_array = in_array[tuple(sub_indices)]
     
            # Apply the sampling matrix to the sub-array
            sampled_sub_array = sub_array * sampling_matrix
         
            # Store the sampled sub-array in the output array at the current grid index
            sampling_ND_array[idx] = sampled_sub_array
            
        else:
            sampled_sub_array   = sampling_ND_array[idx]
            sub_array           = sampled_sub_array * sampling_matrix
            in_array[tuple(sub_indices)] += sub_array


def sign_ND_grid(sampling_start: list=[0, 0, 0],
                 sampling_interval: list=[1, 1, 1],
                 sampling_num: list=[1, 1, 1],
                 module=np):
    '''
                start_position,
                sampling length,
                sampling number,
    '''
    grid = []
    for start, stride_size, num in zip(sampling_start, sampling_interval, sampling_num):
        
        stop     = start + (num-1) * stride_size
        grid_dim = module.linspace(start, stop, num).astype(module.int32);
        
        grid.append(grid_dim)
    return grid


def sign_ND_int_num( dims, sampling_start, sampling_interval, add_num=0):
    '''
        Calculate the number of samples that can fit in the dimension
    '''    
    num_list = []
    for start, sample_size, dim in zip(sampling_start, sampling_interval, dims):
        num = (dim - start + sample_size - 1) // sample_size + add_num

        num_list.append(num)
    
    return num_list


def sign_ND_grid_expand(grid, expand_len): 
    '''
    In each grid array, we add expand_len
    '''
    expanded_grid = []
    
    for grid_arr, length in zip(grid, expand_len):
        
        if isinstance(grid_arr, list):
            grid_expand = [g + length for g in grid_arr]
        else:
            grid_expand = grid_arr + length
        expanded_grid.append(grid_expand)
    
    return expanded_grid
      
                            
class sign_block_ND_class():
    def __init__(self, 

                 dims: list =[1, 2, 3], 
                 window: list =[1, 2, 3], 
                 tapering: list =[1, 2, 3], 

                 sampling_start: list =[1, 2, 3], 
                 sampling_interval: list =[1, 2, 3], 
                 
                 padding_mode='edge',
                 padding_value=0,
                 module=np,
                 dtype=np.float32,
                 log_file='log.txt'):
        """
        Initialize the manipulator with input array, window sizes and taper sizes.
        :param dims: N-dimensional  dims
        
        :param window: The size of the windows in each dimension
        :param tapering: The size of the tapering (overlap) in each dimension
        
        :param sampling_start:    sampling_start
        :param sampling_interval: sampling_interval
        
        example:
        
        dims = [401, 301]
        sign_block_ND_obj = sign_block_ND_class(dims=dims, 
                                                
                                                window=[60, 60], 
                                                tapering=[10, 10],
                                                
                                                sampling_start=[0, 0],
                                                sampling_interval=[50, 50],
                                                
                                                padding_mode='edge',
                                                padding_value=0,
                                                module=np);
        ###if we want to patch the input, we can set window  = tapering + sampling
        PF.imshow(sign_block_ND_obj.sampling_matrix_for, output_name="sampling_matrix_for.eps")
        PF.imshow(sign_block_ND_obj.sampling_matrix_adj, output_name="sampling_matrix_adj.eps")

        # for_array1      = np.random.rand(*dims).astype(dtype=np.float32)
        for_array1      =  np.ones(dims).astype(dtype=np.float32)

        ND_array1       =  sign_block_ND_obj.forward(for_array1)

        # ND_array2      = np.random.rand(*ND_array1.shape).astype(dtype=np.float32)
        ND_array2       =  np.ones(ND_array1.shape).astype(dtype=np.float32)

        for_array2      =  sign_block_ND_obj.adjoint(ND_array2)

        dot1   =  np.sum(  for_array1 * for_array2);

        dot2   =  np.sum(  ND_array1 * ND_array2);

        print( "dot1={}, dot2={}".format(dot1, dot2) )


        PF.imshow(for_array1, output_name="for_array1.eps")
        PF.imshow(for_array2, output_name="for_array2.eps")        
        """
        self.readme     = {}; ##I can record something, when I save it
        
        self.name       = self.__class__.__name__
        
        self.log_file   = log_file
        
        
        #############
        self.module             = module
        self.dtype              = dtype
        
        self.dims               = dims
        
        self.padding_mode       = padding_mode;
        self.padding_value      = padding_value;
        
        
        self.window             = window

        self.tapering           = tapering
        
        
        
        #########
        self.sampling_matrix_for     = self.module.ones(self.window, dtype=self.dtype);
        
        self.sampling_matrix_adj     = sign_ND_linear_tapering(self.window, self.tapering, self.module);
        
        
        self.sampling_interval  = sampling_interval
        
        self.sampling_start     = sampling_start
        
        self.sampling_num       = sign_ND_int_num(self.dims, self.sampling_start, self.sampling_interval, add_num=1); ###default, add_num=1,

        self.grid               = sign_ND_grid(self.sampling_start, self.sampling_interval, self.sampling_num, self.module);
        
        
        self.expand_left        = [length//2 for length in self.window] ##left 
        self.expand_right       = [length+length//2    for length in self.window] ##right
        
        self.grid_expand        = sign_ND_grid_expand(self.grid, self.expand_left);
        
        
        self.dims_expand        = [ i + j + k  for i, j, k in zip(dims, self.expand_left, self.expand_right) ]

        self.block_dims         = self.sampling_num + self.window 

        '''
        I do not recommend this way to allocate  block_ND_arr
        '''
        # self.block_ND_arr       = self.module.zeros(self.block_dims,  dtype=self.dtype);
        # self.expand_arr         =    self.module.zeros(self.dims_expand, dtype=self.dtype);

        
        descriptions = {
                        'module': "module",
                        
                        'dims': "We should set this parameter as list(in_array.shape), note that this array does not depend on the dimension",
                        
                        'dtype':  "data type",
                        
                        'sampling_start': "sampling start position",
                        
                        'sampling_interval': "sampling interval",
                        
                        'sampling_num': "we compute the num with adding one",
                        
                        'window': "sampling matirx", 
                        
                        'tapering': "tapering matrix", 
                        
                        'grid': "grid[0] denotes the grid array of first dimension, grid[1] denotes the grid array of second dimension, grid[2] denotes the grid array of second dimension ", 
                        
                        'grid_expand': "grid array of the expanded array for the expanded array", 

                        'sampling_matrix_for': "forward matrix for blocking the input array", 
                        
                        'sampling_matrix_adj': "adjoint matrix for reverse operation", 
                        }
        
        ##step1    ##recording calss ini value     WR.
        WR.class_dict_ini_log(log_file, self.__class__, w_type="w");  
        
        ##step2    # Populate the dict with variable names, descriptions, and values WR.
        self.dict                   = {}
        WR.class_dict_description(self, descriptions);

        ##step3    ##record the final log WR.
        WR.class_dict_log_file(log_file, self.dict, w_type="a")


    def forward(self, in_array):
        #self.sampling_matrix_for
        expand_arr      = sign_ND_expand(in_array, self.expand_left, self.expand_right, self.padding_mode, self.padding_value, self.module);
        
        block_ND_arr       = self.module.zeros(self.block_dims,  dtype=self.dtype);
        
        sign_ND_sampling(expand_arr, block_ND_arr=block_ND_arr, grid_list=self.grid_expand, sampling_matrix=self.sampling_matrix_for, allocate_arr=False, forward=True, module=self.module);

        return block_ND_arr


    def adjoint(self, block_ND_arr):
        #self.sampling_matrix_adj
        expand_arr = self.module.zeros(self.dims_expand, dtype=self.dtype);
        
        sign_ND_sampling(expand_arr, block_ND_arr=block_ND_arr, grid_list=self.grid_expand, sampling_matrix=self.sampling_matrix_adj, allocate_arr=False, forward=False, module=self.module);
        
        ou_array        = sign_ND_cut(expand_arr, self.expand_left, self.expand_right, self.module);

        return ou_array


def signal_snr(signal, noisy_signal):

    module = WR.get_module_type(signal)    

    noise = noisy_signal - signal
    

    signal_power = module.sum(signal ** 2)
    noise_power = module.sum(noise ** 2)
    

    snr = 10 * module.log10(signal_power / noise_power)
    
    return snr.item()
















################################ 
################################    
################################  
################################ 
################################    
################################    

def sp_wavenumber(nx=100, dx=10, ini_dims=(100, 1), extend_dims=(1, 100), module=cp, broadcast_to=True):
    '''
    Usage: compute an array of 1D/2D/3D: kx, ky and kz
        test:
            kx1 = sp_wavenumber(nx=100, dx=10, ini_dims=(100,1), module=cp);
            kx2 = cp.fft.fftfreq(100, d=10) * 2*  cp.pi; kx2=kx2.reshape(100, 1).astype(cp.float32);
            k_res = kx1 - kx2;
            print(k_res.max());print(k_res.min());
    
    
    ##########note that  one is ini_dims=(nnx, 1, 1), extend_dims=(1, nny, nnz), broadcast_to=False
                 another one is ini_dims=(nnx, 1, 1), extend_dims=(nnx, nny, nnz), broadcast_to=True
    
    methdod1:
        kx_arr                = sp_wavenumber(nnx, dx, ini_dims=(nnx, 1, 1), extend_dims=(1, nny, nnz), module=cp, broadcast_to=False); #Note that, generally, some packages provide the number and frequency
        ky_arr                = sp_wavenumber(nny, dy, ini_dims=(1, nny, 1), extend_dims=(nnx, 1, nnz), module=cp, broadcast_to=False)
        kz_arr                = sp_wavenumber(nnz, dz, ini_dims=(1, 1, nnz), extend_dims=(nnx, nny, 1), module=cp, broadcast_to=False)
        kx_ky_kz_2            = kx_arr*kx_arr + ky_arr*ky_arr + kz_arr*kz_arr;
        
    methdod2:
        kx_arr                = sp_wavenumber(nnx, dx, ini_dims=(nnx, 1, 1), extend_dims=(nnx, nny, nnz), module=cp, broadcast_to=True); #Note that, generally, some packages provide the number and frequency
        ky_arr                = sp_wavenumber(nny, dy, ini_dims=(1, nny, 1), extend_dims=(nnx, nny, nnz), module=cp, broadcast_to=True)
        kz_arr                = sp_wavenumber(nnz, dz, ini_dims=(1, 1, nnz), extend_dims=(nnx, nny, nnz), module=cp, broadcast_to=True)
        kx_ky_kz_2            = kx_arr*kx_arr + ky_arr*ky_arr + kz_arr*kz_arr;
    
    '''
    
    dkx = 2.0 * module.pi / ( nx * dx);
    
    kx  = module.zeros((nx), dtype=module.float32);

    for ix in range(0,nx):
        if ix < nx//2:
            kx[ix] = 1.0*ix*dkx;
        else:
            # kx[ix] = 1.0*(ix+1-nx)*dkx;	## kx = cp.fft.fftfreq(nx, d=dx) * 2*  cp.pi; ## this is a mistake from matlab
            kx[ix] = 1.0*(ix -nx)*dkx;	## kx = cp.fft.fftfreq(nx, d=dx) * 2*  cp.pi;
    
    kx = module.reshape(kx, ini_dims);            

    if len(extend_dims)!=0:
        if broadcast_to:
            kx = cp.broadcast_to(kx, extend_dims)
        else:    
            kx = cp.tile(kx, extend_dims)

    return kx.astype(module.float32)



def generate_kx_kz_2D(nx=100, dx=100, nz=100, dz=1):
    
    dkx=2.0*np.pi/(nx*dx);
    dkz=2.0*np.pi/(nz*dz);
    
    kx = np.zeros((nz,nx));
    kz = np.zeros((nz,nx));

    
    for ix in range(0,nx):
        for iz in range(0,nz):
            
            if ix < nx//2:
                kx[iz][ix] = 1.0*ix*dkx;
            else:
                # kx[iz][ix] = 1.0*(ix+1-nx)*dkx;	 ###this is an error
                kx[iz][ix] = 1.0*(ix-nx)*dkx;	 ###this is an error

            if iz<nz//2 :
                kz[iz][ix]=1.0*iz*dkz;
            else:
                kz[iz][ix]=1.0*(iz-nz)*dkz;

    k_2_k = kx*kx + kz*kz;

    return k_2_k, kx, kz 

def wavelet_ormsby(fl1,fl2,fh1,fh2, dt, nt, lt=0, amplitude=1, module=cp):
    
    if lt==0:
        lt=nt;
    # 创建时间轴
    
    t_r  = module.linspace(-nt / 2, nt / 2, nt) * dt
    
    pi   = module.pi
    
    tmp1 = pi*fh2*pi*fh2/(pi*fh2-pi*fh1)*module.sinc(fh2*t_r)*module.sinc(fh2*t_r);
    tmp2 = pi*fh1*pi*fh1/(pi*fh2-pi*fh1)*module.sinc(fh1*t_r)*module.sinc(fh1*t_r);
    
    tmp3 = pi*fl2*pi*fl2/(pi*fl2-pi*fl1)*module.sinc(fl2*t_r)*module.sinc(fl2*t_r);
    tmp4 = pi*fl1*pi*fl1/(pi*fl2-pi*fl1)*module.sinc(fl1*t_r)*module.sinc(fl1*t_r);

    wavelet =  tmp1 - tmp2 - (tmp3-tmp4);

    wavelet /= module.max(module.abs(wavelet))
    
    
    if lt != nt:
       pad_length = lt - nt
       if pad_length > 0:
           wavelet = module.pad(wavelet, (0, pad_length), 'constant')
    
    return (amplitude * wavelet).astype( module.float32 )

def wavelet_ormsby_test():
    module=np
    f1, f2, f3, f4 = 5, 10, 40, 45
    dt = 0.001 
    nt = 400 
    lt = 2000 
    amplitude = 1
    
    wavelet = wavelet_ormsby(f1, f2, f3, f4, dt, nt, lt, amplitude)
    print(wavelet)
    
    PF.plot_graph([wavelet[0:400]], output_name="wavelet_ormsby.eps");
    
    PF.plot_graph([module.abs(module.fft.fft(wavelet, axis=0))[0:300] ], output_name="wavelet_ormsby_freq.eps");

def wavelet_klauder(f1, f2, dt, nt, T=7, lt=0, amplitude=1, module=cp):
    
    if lt==0:
        lt=nt;
    
    t_r = module.linspace(-nt / 2, nt / 2, nt) * dt
    
    k  = (f2 - f1 ) /T;
             
    f0 = (f2 + f1 ) /2.0;
    
    pi = module.pi;
    
    pikt = pi*k*t_r;

    wavelet =  module.sin( pikt * (T-t_r) ) / pikt  * module.cos(2.0 * pi * f0 * t_r); 

    wavelet /= module.max(module.abs(wavelet))
    
    
    if lt != nt:
       pad_length = lt - nt
       if pad_length > 0:
           wavelet = module.pad(wavelet, (0, pad_length), 'constant')
    
    return (amplitude * wavelet).astype( module.float32 )

def wavelet_klauder_test():
    module = np
    f1, f2 = 10, 40
    dt = 0.001 
    nt = 400 
    lt = 2000 
    amplitude = 1
    
    wavelet = wavelet_klauder(f1, f2, dt, nt, T=7, lt=lt);
    print(wavelet)
    
    PF.plot_graph([wavelet[0:400]], output_name="wavelet_klauder.eps");
    
    PF.plot_graph([module.abs(module.fft.fft(wavelet, axis=0))[0:200] ], output_name="wavelet_klauder_freq.eps");

def wavelet_ricker(nt=200, lt=1000, dt=0.001, freq=20, module=cp):
    
    return ricker_func(nt, lt, dt, freq, module);
    
def ricker_func(nt=200, lt=1000, dt=0.001, freq=20, module=cp):
    '''
    Function to generate a Ricker wavelet signal.

    Parameters:
    ----------
    nt : int, default=400
        Number of time sampling points (the total duration of the waveform).
        
    lt : int, default=1000
        The length of the resulting Ricker wavelet.
        
    dt : float, default=0.001
        The time sampling interval in seconds. Typically represents the sampling rate of the wavelet.
        
    freq : float, default=20
        The central frequency of the wavelet in Hz, controlling the oscillation frequency of the wavelet.
        
    module : module, default=cp (CuPy)
        This parameter allows users to specify whether to use the NumPy or CuPy module for generating the wavelet.
        Defaults to `CuPy` for GPU-accelerated computation.
        The user can choose `np` (NumPy) or `cp` (CuPy) for generating the Ricker wavelet on the CPU or GPU.

    Returns:
    ----------
    ricker : array_like, shape (lt,)
        The generated Ricker wavelet as a `module.float32` array (either NumPy or CuPy array).
        The length of the wavelet is `lt`, ensuring that even if `nt` is smaller, the final array will still have a length of `lt`.
        The peak amplitude of the Ricker wavelet is located at the center of the signal.

    Description:
    ----------
    The Ricker wavelet is a common seismic waveform model used to simulate the time evolution of seismic waves. It is a zero-phase, 
    asymmetric waveform commonly used in seismology and signal processing to simulate the response of seismic wave propagation.
    This function allows users to control the oscillatory characteristics of the wavelet via time step and central frequency parameters.

    Formula:
    ----------
    The formula for the Ricker wavelet:
    ricker(t) = (1 - 2 * (π * freq * (t - t0))^2) * exp(-(π * freq * (t - t0))^2)
    where `t0` is the center of the wavelet, `freq` is the central frequency, and `t` is the time relative to the center.

    Example:
    ----------
    # Generate a Ricker wavelet using NumPy
    import numpy as np
    ricker_np = ricker_func(nt=500, lt=1000, dt=0.001, freq=25, module=np)

    # Generate a Ricker wavelet using CuPy
    import cupy as cp
    ricker_cp = ricker_func(nt=500, lt=1000, dt=0.001, freq=25, module=cp)
    '''

    ricker = module.zeros( (lt, 1), dtype=module.float32 );
    
    for it in range (0,nt):
        t_real      = (it-nt//2 + 1)*dt;
        tmp         = module.pi * freq *t_real;
        tmp         = tmp * tmp ;
        ricker[it] = (1.0-2.0*tmp)*module.exp(-1.0*tmp);
    
    return ricker.astype(module.float32)


def phase_rotate_nd_array(array, angle_degrees, axis=0, module=cp):
    """
    对 N 维数组沿指定轴进行相位旋转，正频部分应用 `module.exp(1j * angle_radians)`，负频部分应用 `module.exp(-1j * angle_radians)`。
    
    Parameters:
    array: 输入数组，进行相位旋转的对象。
    angle_degrees: 旋转的相位角，单位为度。
    axis: 沿着执行 FFT 和相位旋转的轴，默认为 0。
    module: 使用的模块 (默认是 CuPy，GPU 计算；也可以用 NumPy 替代用于 CPU 计算)。
    
    Returns:
    旋转后相位的数组，形状与输入数组相同。
    """
    angle_radians = angle_degrees / 180.0 * module.pi

    # Perform FFT along the specified axis
    fft_array = module.fft.fft(array, axis=axis)

    # 获取频率索引，正频为 >= 0，负频为 < 0
    freqs = module.fft.fftfreq(array.shape[axis])

    # 初始化相位偏移数组，正频率部分用正的相位旋转，负频率部分用负的相位旋转
    phase_shift = module.ones_like(fft_array, dtype=module.complex64);
    
    # 对正频率部分使用 `exp(1j * angle_radians)`
    pos_freqs_idx = freqs >= 0
    phase_shift[pos_freqs_idx] = module.exp(1j * angle_radians);

    # 对负频率部分使用 `exp(-1j * angle_radians)`
    phase_shift[~pos_freqs_idx] = module.exp(-1j * angle_radians);

    # 应用相位旋转
    rotated_fft_array = fft_array * phase_shift

    # Inverse FFT to go back to the time/space domain
    rotated_array = module.fft.ifft(rotated_fft_array, axis=axis)

    # 返回实部并确保数据类型为 float32
    return (rotated_array.real).astype(module.float32)


# def phase_rotate_nd_array(array, angle_degrees, axis=0, module=cp):
#     """
#     Rotate the phase of an N-dimensional array along a specified axis by a given angle.
    
#     Parameters:
#     array: The input array to be phase rotated.
#     angle_degrees: The phase rotation angle in degrees.
#     axis: The axis along which to apply the FFT and phase rotation (default is 0).
#     module: The module to use (default is CuPy for GPU, can be replaced by NumPy for CPU).
    
#     Returns:
#     A phase-rotated array with the same shape as the input array.
#     """
#     angle_radians = angle_degrees / 180.0 * module.pi

#     # Perform FFT along the specified axis
#     fft_array = module.fft.fft(array, axis=axis)

#     # Generate frequency indices based on the shape of the array along the axis
#     freqs = module.fft.fftfreq(array.shape[axis])

#     # Apply phase shift depending on the frequency components
#     phase_shift = module.exp(1j * angle_radians * freqs)

#     # Reshape phase shift to apply correctly along the specified axis
#     phase_shift = module.reshape(phase_shift, [-1 if i == axis else 1 for i in range(array.ndim)])

#     # Apply the phase shift
#     rotated_fft_array = fft_array * phase_shift

#     # Inverse FFT to go back to the time/space domain
#     rotated_array = module.fft.ifft(rotated_fft_array, axis=axis)

#     return (rotated_array.real).astype(module.float32)


def apply_sqrt_iw_operator(signal, axis, dt, correct_sign=+1, forward=True, limit_f=0, module=cp):
    """
    Applies the square root operator sqrt(iω) to the specified dimension of an N-dimensional signal.

    Parameters:
    signal (cupy.ndarray): Input N-dimensional signal.
    axis (int): The dimension along which to apply the operator.
    dt (float): Time sampling interval.
    correct_sign (int): Sign correction, default is +1.
    forward (bool): Whether to apply the forward operator, default is True.
    limit_f (float): Frequency limit, default is 0 (no limit). When non-zero, filtering is only applied to frequencies below the limit.
    module: The computation module, default is cupy (cp).

    Returns:
    cupy.ndarray: The signal after applying the operator.
    """
    module     = WR.get_module_type(signal)
    
    # Perform Fourier transform along the specified axis
    signal_fft = module.fft.fft(signal, axis=axis)
    
    # Get frequency components
    shape = signal.shape[axis]
    freqs = module.fft.fftfreq(shape, d=dt)
    
    # Calculate the sqrt(iω) operator
    if forward:
        iw_operator = module.sqrt(correct_sign * 1j * 2 * module.pi * freqs)
    else:
        # Avoid division by zero
        freqs[freqs == 0] = 1e-10
        iw_operator = 1.0 / module.sqrt(correct_sign * 1j * 2 * module.pi * freqs)
    
    # Limit the frequency range if limit_f is non-zero
    if limit_f > 0:
        iw_operator = module.where(module.abs(freqs) <= limit_f, iw_operator, 1.0)
    
    
    # Expand dimensions for broadcasting
    iw_operator = module.reshape(iw_operator, [(1 if i != axis else -1) for i in range(signal.ndim)])
    
    # Apply the sqrt(iw) operator
    signal_fft_corrected = signal_fft * iw_operator ##This broadcasting is implicit and is handled by the CuPy (or NumPy) backend during the multiplication.
    
    # Perform the inverse Fourier transform
    corrected_signal = module.fft.ifft(signal_fft_corrected, axis=axis)
    
    # Since the inverse Fourier transform may return complex numbers, return the real part (assuming input was real)
    return corrected_signal.real.astype(signal.dtype)



def apply_iw_operator(signal, axis, dt, correct_sign=+1, power_num=1, forward=True, limit_f=0, module=cp):
    """
    Applies the (iω)^power_num operator to the specified dimension of an N-dimensional signal,
    optionally limiting the operation to frequencies below limit_f.

    Parameters:
    signal (cupy.ndarray): Input N-dimensional signal.
    axis (int): The dimension along which to apply the operator.
    dt (float): Time sampling interval.
    correct_sign (int): Sign correction, default is +1.
    power_num (int): The power to which iω is raised. Default is 1.
    forward (bool): Whether to apply the forward operator, default is True.
    limit_f (float): Frequency limit. If non-zero, the operator is only applied to frequencies below the limit.
    module: The computation module, default is cupy (cp).

    Returns:
    cupy.ndarray: The signal after applying the operator.
    """
    module     = WR.get_module_type(signal)
    
    # Perform Fourier transform along the specified axis
    signal_fft = module.fft.fft(signal, axis=axis)
    
    # Get frequency components
    shape = signal.shape[axis]
    freqs = module.fft.fftfreq(shape, d=dt)
    
    # Compute the (iω)^power_num operator
    iw_operator = module.power(correct_sign * 1j * 2 * module.pi * freqs, power_num)
    
    # Limit the frequency range if limit_f is non-zero
    if limit_f > 0:
        iw_operator = module.where(module.abs(freqs) <= limit_f, iw_operator, 1.0)
    
    # Reshape for broadcasting
    iw_operator = module.reshape(iw_operator, [(1 if i != axis else -1) for i in range(signal.ndim)])
    
    # Apply the (iω)^power_num operator
    signal_fft_corrected = signal_fft * iw_operator ##This broadcasting is implicit and is handled by the CuPy (or NumPy) backend during the multiplication.
    
    # Perform the inverse Fourier transform
    corrected_signal = module.fft.ifft(signal_fft_corrected, axis=axis)
    
    # Return the real part of the signal (assuming the input was real)
    return corrected_signal.real.astype(signal.dtype)*dt



def compute_autocorrelation(signal, axis=0):
    """
    Computes the autocorrelation of the given signal along a specified axis.
    
    Parameters:
    signal (cupy.ndarray): The input signal, which can be multidimensional.
    axis (int): The axis along which to compute autocorrelation.
    
    Returns:
    autocorr: The autocorrelation of the signal along the specified axis.
    """
    module     = WR.get_module_type(signal)
    
    # Move the desired axis to the front (axis=0) for easier iteration
    signal = module.moveaxis(signal, axis, 0)
    
    autocorr_list = []
    
    # Iterate over each slice along the other dimensions
    for i in range(signal.shape[1]):
        # Compute the autocorrelation for each slice
        autocorr = module.correlate(signal[:, i], signal[:, i], mode='full')
        autocorr_list.append(autocorr)
    
    length = len(autocorr)
    
    return (module.array(autocorr_list)).reshape(length, -1)


def generate_1D_arrary_with_tapering(center_value, flat_width, left_length, right_length, decay_power=1, decay_type="cos", tapering_or_not=False, module=cp):
    '''
    Usage:
    att_x1 = generate_1D_arrary_with_tapering(center_value=0, flat_width=200, left_length=100, right_length=100, decay_power=1, decay_type="cos", tapering_or_not=False, module=cp);
    PF.plot_graph([att_x1, ], output_name="att_x1.eps")  ###for abc1

    att_x2 = generate_1D_arrary_with_tapering(center_value=1, flat_width=200, left_length=100, right_length=100, decay_power=1, decay_type="cos", tapering_or_not=True, module=cp);
    PF.plot_graph([att_x2, ], output_name="att_x2.eps")   ###for abc2
    
    '''
    length      = flat_width + left_length + right_length; ##
    
    
    middle_arr    = module.zeros(flat_width, dtype=module.float32);
    middle_arr[:] = center_value;
    
    indices1     = module.arange(left_length);
    indices2     = module.arange(right_length);
    
    if    decay_type=="cos":
        left_arr     = module.power(module.cos(indices1/left_length * module.pi/2), decay_power)
        left_arr     = module.flip(left_arr)

        right_arr    = module.power(module.cos(indices2/right_length * module.pi/2), decay_power)
    elif  decay_type=="linear":
        left_arr     = module.power(indices1/left_length, decay_power)
        right_arr    = module.power(indices2/right_length, decay_power)
        
        right_arr    = module.flip(right_arr)  
        
    if tapering_or_not:
        output_arr  =   module.concatenate((left_arr, middle_arr, right_arr), axis=0)
        
    else:
        output_arr  =   module.concatenate((module.flip(left_arr), middle_arr, module.flip(right_arr)), axis=0)
        
    return output_arr.astype(module.float32)  

    
def generate_1D_arrary_with_tapering_with_constant(center_value, flat_width, left_length, right_length, constant_left_length=0, constant_left_value=0, constant_right_length=0, constant_right_value=0, decay_power=1, decay_type="cos", tapering_or_not=False, module=cp):
    '''
    Generate a 1D array with a central flat region, optional tapering on both sides, 
    and optional constant regions on the left and right.
    
    Parameters:
    center_value: float
        Value for the central flat region.
    flat_width: int
        Number of elements in the central flat region.
    left_length: int
        Number of elements for the left tapering region.
    right_length: int
        Number of elements for the right tapering region.
    constant_left_length: int, optional
        Number of constant elements to add on the left.
    constant_left_value: float, optional
        Value of the constant elements added on the left.
    constant_right_length: int, optional
        Number of constant elements to add on the right.
    constant_right_value: float, optional
        Value of the constant elements added on the right.
    decay_power: float, optional
        Power to which the tapering function is raised.
    decay_type: str, optional
        Type of tapering: "cos" (cosine) or "linear".
    tapering_or_not: bool, optional
        If True, apply tapering directly; otherwise, flip the tapering values.
    module: cp or np, optional
        The numerical module to use, either CuPy (cp) or NumPy (np).

    Returns:
    A 1D array with the specified tapering, constant regions, and center value.
    '''
    # Calculate total length
    length = flat_width + left_length + right_length + constant_left_length + constant_right_length

    # Create the middle flat region
    middle_arr = module.zeros(flat_width, dtype=module.float32)
    middle_arr[:] = center_value

    # Create indices for tapering
    indices1 = module.arange(left_length)
    indices2 = module.arange(right_length)

    # Calculate tapering for left and right
    if decay_type == "cos":
        left_arr = module.power(module.cos(indices1 / left_length * module.pi / 2), decay_power)
        left_arr = module.flip(left_arr)
        right_arr = module.power(module.cos(indices2 / right_length * module.pi / 2), decay_power)
    elif decay_type == "linear":
        left_arr = module.power(indices1 / left_length, decay_power)
        right_arr = module.power(indices2 / right_length, decay_power)
        right_arr = module.flip(right_arr)
    
    # Apply tapering or flip
    if tapering_or_not:
        tapered_arr = module.concatenate((left_arr, middle_arr, right_arr), axis=0)
    else:
        tapered_arr = module.concatenate((module.flip(left_arr), middle_arr, module.flip(right_arr)), axis=0)
    
    # Add constant region on the left if specified
    if constant_left_length > 0:
        constant_left_arr = module.full(constant_left_length, constant_left_value, dtype=module.float32)
        tapered_arr = module.concatenate((constant_left_arr, tapered_arr), axis=0);
    
    # Add constant region on the right if specified
    if constant_right_length > 0:
        constant_right_arr = module.full(constant_right_length, constant_right_value, dtype=module.float32);
        tapered_arr = module.concatenate((tapered_arr, constant_right_arr), axis=0);
    
    return tapered_arr.astype(module.float32)
 
    
 



def sign_colored_noise(shape, sigma=1, max_value=1, filter_lengths=None, dtype=np.float32):
    """
    Generate an N-dimensional colored noise array with specified shape, sigma, and maximum value.

    Parameters:
    - shape: list or tuple, specifying the size for each dimension.
    - sigma: float, standard deviation for the white noise.
    - max_value: float, the maximum scale factor for the final noise (default is 1).
    - filter_lengths: list of integers or None, specifying filter lengths for each dimension.
                      If None, no filtering is applied in that dimension.

    Returns:
    - colored_noise: N-dimensional array with colored noise, scaled by max_value.
    
    # Example usage
    shape = (50, 50, 50)  # Dimensions of the noise
    sigma = 0.06          # Standard deviation of the white noise
    max_value = 1         # Maximum scaling factor for the noise
    filter_lengths = [5, 10, 15]  # Filter lengths for each axis

    colored_noise = sign_colored_noise(shape, sigma, max_value, filter_lengths)
    """
    
    # start_time = 
    
    if isinstance(filter_lengths, (int, float)):
        filter_lengths = [int(filter_lengths)] * len(shape)
    elif filter_lengths is None:
        filter_lengths = [None] * len(shape)
    
    # Generate initial white noise
    noise = np.random.normal(0, sigma, shape)
    
    # Apply filters along each dimension if filter length is provided
    for axis, (size, f_length) in enumerate(zip(shape, filter_lengths)):
        if f_length and size > 1:
            noise = filtfilt(np.ones(f_length) / f_length, 1, noise, axis=axis, method='gust')
            # print(f"Filter applied along axis {axis} with length {f_length}")
    
    colored_noise = max_value * noise
    
    return colored_noise.astype( dtype )


def sign_mask_generate(data_shape, proportions=[0, 0.8], module=np):
    """
    Generate a mask filter based on the given proportions,
    filling specified proportions with 0 .

    Parameters:
        data_shape (list or tuple): The shape of the data (e.g., [100, 300]).
        proportions (list): The proportions of each dimension to be filled with 0 (e.g., [0.5, 0.5]).
        module (module): The numerical module to use, default is numpy.

    Returns:
        module.ndarray: The generated mask filter.
    """
    if len(data_shape) != len(proportions):
        raise ValueError("The lengths of data_shape and proportions must match.")
    
    # Initialize a mask filled with 1
    mask = module.ones(data_shape, dtype=module.float32)

    for dim, proportion in enumerate(proportions):
        
        if proportion < 0 or proportion > 1:
            raise ValueError(f"Proportion for dimension {dim} is invalid: {proportion}. Must be in [0, 1].")
        
        size = data_shape[dim]
        zero_size = int(size * proportion)
        # Randomly select indices to fill with 0
        zero_indices = module.random.choice(size, zero_size, replace=False)

        # Create a slice object to select all indices in other dimensions
        slices = [slice(None)] * len(data_shape)
        for idx in zero_indices:
            slices[dim] = idx
            mask[tuple(slices)] = 0  # Set selected indices to 0

    return mask

def sign_mask_generate_test():
    # Example usage
    data_shape = [10, 10]
    proportions = [1, 0.8]
    mask = sign_mask_generate(data_shape, proportions, module=np)
    print(np.sum(mask))




