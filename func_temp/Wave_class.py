


# import sys
# sys.path.append("/home/zhangjiwei/pyfunc/lib")

from lib_sys import *


# from wave_libary1 import *
# from wave_common import *


class Wave_Equation_Solver():
    '''
    no mater how the dimension of input, we can consider them as [nz, ny, nx] and [dz, dy, dx]
    no mater how the dimension of input, we can consider them as [nz, ny, nx] and [dz, dy, dx]
    no mater how the dimension of input, we can consider them as [nz, ny, nx] and [dz, dy, dx]
    
    dx: axis=2
    dy: axis=1
    dz: axis=0
    '''
    def __init__(self, 
                 TIME_order: int = 1, 
                 
                 FD_order: int = 10, 
                 
                 FD_coemethod: int =0,
                 
                 SP_bool: bool = True,
                 
                 log_file: str = "Wave_Equation_Solver_log.txt",
                ):
        '''
        see the following information
        
        TIME_order: FD order for second time derivative or first-order derivative 
                    Second-order: <7 works
                    First-order:  infinite order works
        
        FD_order: FD order for the spatial derivatives,
        
        FD_coemethod: 0, 1, or 2; works for the second-order coeff method,
        
        SP_bool:  FFT (True) or FD (False) to compute the fisrt or second order derivatives in the spatial directions,
        
        
        vector_weq_bool: 
                       True: Variable density acoustic wave equation (velocity-stress), 
                       note that there is a integral operation in the source term.
                       False: Constant density acoustic wave equation (second order)
                       
        
        log_file: str = "log.txt"
        '''
        self.readme         = {}; ##I can record something
        
        self.name           = self.__class__.__name__
        
        self.log_file       = log_file
        
        self.TIME_order     = np.int32(TIME_order);       ### TIME_order, <=7 works,
        
        self.FD_order       = np.int32(FD_order);         ### FD order 
        
        self.FD_coemethod   = np.int32(FD_coemethod); ### only works for the second order coeff method
        
        self.SP_bool        = np.int32(SP_bool);          ## FFT or FD to compute the fisrt D or second D
        
        self.radius         = np.int32(FD_order//2);       ### half of FD order 
        
        self.coe_1_d        = cp.asarray(      FD_first_derivative_coeff_stagger(method=0, order=self.FD_order, dtype=np.float32), dtype=np.float32       ); #  first deritative coeff  for stagger grid, default method=0
        
        self.coe_2_d        = cp.asarray(      FD_second_derivative_coeff(method=self.FD_coemethod, order=self.FD_order, dtype=np.float32)       , dtype=np.float32 ); #  second deritative coeff
        
        self.coe_time_2_d   = cp.asarray(   FD_multiple_angle_coeff(method=0, order=self.TIME_order, dtype=np.float32), dtype=np.float32     ); #  high  order time order coeff, default method=0
        
        self.coe_time_1_d   = cp.asarray(   FD_Lie_produce_coeff(method=0, order=self.TIME_order, dtype=np.float32), dtype=np.float32    ); #  high  order time order coeff, default method=0
        
        
        self.FD2d_dict         = FD2d_dict_func( log_file );
        self.FD3d_dict         = FD3d_dict_func( log_file );
        self.wave_kernel_dict  = wave_kernel_dict_func( log_file );

        self.ray_kernel2d_dict = ray_kernel2d_dict_func( log_file );
        self.ray_kernel3d_dict = ray_kernel3d_dict_func( log_file );
    
        descriptions = {
                        'TIME_order': "time order of FD for PDE",
                        
                        'FD_order': "spatial finite difference (FD) order (2N, for instance FD_order=10, radius=5) for first order and second deritative coefficent",
                        
                        'FD_coemethod': "The way to compute the second deritative coeff, see zhang 2009, geophysics", 
                        
                        'SP_bool': 'If it is true, I will compute the spatial deritatives for PDE using the cupy.fft function, it is highly computational costs than the spatial FD method',
                        
                        'coe_1_d': "First order deritative coefficients, see reference: Determination of finite-difference weights using scaled binomial windows",
                        
                        'coe_2_d': "Second order deritative coefficients, see reference:  Determination of finite-difference weights using scaled binomial windows",
                        
                        'coe_time_2_d': "high order time FD coefficients, see reference: Time-stepping wave-equation solution for seismic modeling using a multiple-angle formula and the Taylor expansion",
                        
                        'FD2d_dict':  "Raw Kernel for 1D/2D FD operations",
                        
                        'FD3d_dict':  "Raw Kernel for    3D FD operations",
                        
                        'wave_kernel_dict':  "Some operations for wave equation migration, RTM, one-way, and ray-based migration",
                        
                        'ray_kernel2d_dict':  "2D ray-based modeling/migration/hessian kernel function",
                        
                        'ray_kernel3d_dict':  "3D ray-based modeling/migration/hessian kernel function",
                        }
        

        
        ##step1    ##recording calss ini value    
        WR.class_dict_ini_log(log_file, self.__class__, w_type="w");  
        
        ##step2    # Populate the dict with variable names, descriptions, and values
        self.dict                   = {}
        WR.class_dict_description(self, descriptions);

        ##step3    ##record the final log
        WR.class_dict_log_file(log_file, self.dict, w_type="a")


    

class Wave_Mod_Para():
    """
    This class allocate somes parameter array on cupy for further modeling, migration, and inversion,
    
    Note that our array should be [nz, ny, nx]
    but when I will transfer the parameter into a function, I will always use [dx, dy, dz] and dims=[nx, ny, nz] and [start_x, start_y, start_z]

    All parameters are set as grid point (dx, dy, dz)
    """
    def __init__(self, 
                 
                 dims: list=[1, 1, 1],
                 
                 dims_interval: list=[1, 1, 1],
                 
                 vp: int = 0, 
                 vs: int = 0, 
                 den: int = 0, 
                 
                 s_vp: int = 0, 
                 s_vs: int = 0,
                 s_den: int = 0, 
                 
                 rel_vp: int = 0, 
                 rel_vs: int = 0, 
                 rel_den: int = 0, 
                 
                 ref: int = 0,
                 
                 use_cupy = True,

                 log_file="Wave_Mod_Para_log.txt"
                ):
        
        self.readme     = {}; ##I can record something

        self.name       = self.__class__.__name__
        
        self.log_file   = log_file
        
        self.use_cupy   = use_cupy
        
        
        
        self.cpasnumpy_or = lambda x: x if self.use_cupy else cp.asnumpy(x);

        # Initialize dimensions and intervals
        self.dims = list( map(np.int32, dims) )
        self.dims_interval = list( map(np.float32, dims_interval) )
        
        if len(self.dims) == 3:
            self.nx, self.ny, self.nz = np.int32(self.dims)
            self.dx, self.dy, self.dz = np.float32(self.dims_interval)
        elif len(self.dims) == 2:
            self.nx = np.int32(self.dims[0])
            self.ny = np.int32(1)
            self.nz = np.int32(self.dims[1])
            self.dx = np.float32(self.dims_interval[0])
            self.dy = np.float32(1)
            self.dz = np.float32(self.dims_interval[1])
        else:
            raise ValueError("{}, dims is must be 2 or 3".format(self.name))
        
        
        # convert_to_cpcontiguousarray = lambda x: cp.ascontiguousarray( cp.asarray( x ).astype(cp.float32) );
        convert_to_cpcontiguousarray = lambda x: cp.ascontiguousarray(cp.asarray(x).astype(cp.float32)) if isinstance(x, (list, tuple, cp.ndarray, np.ndarray)) else x
        
        # Convert input arrays to contiguous CuPy arrays
        self.vp         = convert_to_cpcontiguousarray( vp );
        self.vs         = convert_to_cpcontiguousarray( vs );
        self.den        = convert_to_cpcontiguousarray( den );
        self.s_vp       = convert_to_cpcontiguousarray( s_vp );
        self.s_vs       = convert_to_cpcontiguousarray( s_vs );
        self.s_den      = convert_to_cpcontiguousarray( s_den );
        self.rel_vp     = convert_to_cpcontiguousarray( rel_vp );
        self.rel_vs     = convert_to_cpcontiguousarray( rel_vs );
        self.rel_den    = convert_to_cpcontiguousarray( rel_den );
        self.ref        = convert_to_cpcontiguousarray( ref );
        
        
        self.rel_ip     =  self.rel_vp  + self.rel_den;
        
        self.rel_k      =  2.0 * self.rel_vp  + self.rel_den;
        
        # rel_vel_arr  = (vel_arr - s_vel_arr )/s_vel_arr
        # rel_den_arr  = (den_arr - s_den_arr )/s_den_arr
        # rel_Ip_arr   = rel_vel_arr     + rel_den_arr
        # rel_K_arr    = 2.0*rel_vel_arr + rel_den_arr
        
        
        descriptions = {
                        'dims': "When I will initalized the Class, I can set it as vel.T.shape for simplification\n"
                                "Note that our array should be [nz, ny, nx]\n"
                                "but when I will transfer the parameter into a function, I will always use [dx, dy, dz], dims=[nx, ny, nz], [start_x, start_y, start_z], [sx, sy, sz], [gx, gy, gz], [min_x, min_y, min_z], [max_x, max_y, max_z]\n"

                                "All parameters are set as grid point (dx, dy, dz)", 
            
            
                        'vp': "True P-wave velocity model",
                        
                        'vs': "True S-wave velocity model",
                        
                        'den': "True denisty model",
                        

                        }
        
        
        ##step1    ##recording calss ini value    
        WR.class_dict_ini_log(log_file, self.__class__, w_type="w");  
        
        ##step2    # Populate the dict with variable names, descriptions, and values
        self.dict                   = {}
        WR.class_dict_description(self, descriptions);

        ##step3    ##record the final log
        WR.class_dict_log_file(log_file, self.dict, w_type="a")








class Wave_Mig_Para():
    """
    This class holds parameters related to migration, ADCIGs extraction,
    and angle-dependent Kirchhoff modeling for seismic applications.
    
    Note that our array should be [nz, ny, nx]
    but when I will transfer the parameter into a function, I will always use [dx, dy, dz] and dims=[nx, ny, nz] and [start_x, start_y, start_z]

    All parameters are set as grid point (dx, dy, dz)
    """
    def __init__(self, 
                 
                 dims: list=[1, 1, 1],
                 
                 dims_interval: list=[1, 1, 1],
                 
                 smooth_simga_list: list = [1, 1, 1],
                 
                 angle_start: float = -45.0, 
                 angle_interval: float = 1.0, 
                 angle_num: int = 90, 
                 angle_sigma: float = 8.0, 
                 angle_weight_len: int = 0, 
                 angle_tapering_power: float = 3.0,
                 
                 angle_mig_bool: int = 0,
                 angle_tapering_bool: int =1, 
                 
                 
                 mig_stable_coe: float = 10.0,
                 mig_stable_position: int = 100,
                 
                 imaging_conidition_type: int = 0,
                 division_mark: int =0,
                 
                 
                 mig_it_start: int =0,
                 
                 use_cupy = True,
                 
                 log_file="Wave_Mig_Para_log.txt"
                ):

        self.readme     = {}; ##I can record something

        self.name       = self.__class__.__name__
        
        self.log_file   = log_file
        
        self.use_cupy   = use_cupy
        
        

        self.cpasnumpy_or = lambda x: x if self.use_cupy else cp.asnumpy(x);

        # Initialize dimensions and intervals
        self.dims = list( map(np.int32, dims) )
        self.dims_interval = list( map(np.float32, dims_interval) )
        
        if len(self.dims) == 3:
            self.nx, self.ny, self.nz = np.int32(self.dims)
            self.dx, self.dy, self.dz = np.float32(self.dims_interval)
        elif len(self.dims) == 2:
            self.nx = np.int32(self.dims[0])
            self.ny = np.int32(1)
            self.nz = np.int32(self.dims[1])
            self.dx = np.float32(self.dims_interval[0])
            self.dy = np.float32(1)
            self.dz = np.float32(self.dims_interval[1])
        else:
            raise ValueError("{}, dims is must be 2 or 3".format(self.name))

        


        self.smooth_simga_list      = smooth_simga_list;
        
        
        
        self.angle_start            = np.float32(angle_start);
        self.angle_interval         = np.float32(angle_interval);
        self.angle_num              = np.int32(angle_num);
        self.angle_sigma            = np.float32(angle_sigma);
        self.angle_weight_len       = np.int32(angle_weight_len);
        self.angle_tapering_power   = np.float32(angle_tapering_power);
        self.angle_end              = np.float32( self.angle_start + self.angle_num * self.angle_interval  )
        
        
        self.angle_mig_bool         = np.int32(angle_mig_bool );
        self.angle_tapering_bool    = np.int32(angle_tapering_bool );
        
        
        self.mig_stable_coe         = np.float32(mig_stable_coe);
        self.mig_stable_position    = np.int32(mig_stable_position);
        
    
        self.imaging_conidition_type = np.int32(imaging_conidition_type );
        self.division_mark           = np.int32(division_mark );
        
    
        ####
        self.mig_it_start           = np.int32(mig_it_start );
    
    
    
        ##allocate some arrays
        ##allocate some arrays, it is convinient to save these arrays
        ## note that    self.dims[::-1]
        if self.use_cupy:
            self.mig1                   = cp.zeros(self.dims[::-1], dtype=cp.float32);
            self.mig2                   = cp.zeros(self.dims[::-1], dtype=cp.float32);
        else:
            self.mig1                   = np.zeros(self.dims[::-1], dtype=np.float32);
            self.mig2                   = np.zeros(self.dims[::-1], dtype=np.float32);
        
        
        
        if self.angle_mig_bool:
            if self.use_cupy:
                self.angle_mig1         = cp.zeros([self.angle_num,] + self.dims[::-1], dtype=cp.float32);
                self.angle_mig2         = cp.zeros([self.angle_num,] + self.dims[::-1], dtype=cp.float32);
            else:
                self.angle_mig1         = np.zeros([self.angle_num,] + self.dims[::-1], dtype=np.float32);
                self.angle_mig2         = np.zeros([self.angle_num,] + self.dims[::-1], dtype=np.float32);
        
        #########some des in dictionary
        file1 = ("\n(1) Angle-dependent Kirchhoff modeling (limited angle)\n"
        "(2) Extracting ADCIGs\n"
        "(3) Migration precondition\n"
        "(4) Hessian")
        
        descriptions = {
                'dims': "When I will initalized the Class, I can set it as vel.T.shape for simplification\n"
                        "Note that our array should be [nz, ny, nx]\n"
                        "but when I will transfer the parameter into a function, I will always use [dx, dy, dz], dims=[nx, ny, nz], [start_x, start_y, start_z], [sx, sy, sz], [gx, gy, gz], [min_x, min_y, min_z], [max_x, max_y, max_z]\n"

                        "All parameters are set as grid point (dx, dy, dz)", 
                        
                        
            
                'smooth_simga_list': "list of float32\n"
                                     "Sigma parameters of Gaussian smooth filter for smoothing the velocity, density, and so on",
                                
                'angle_start': "float32\n"
                               "start position of angle {}".format(file1),
                
                'angle_interval': "float32\n"
                                  "angle_interval  {}".format(file1),
                
                'angle_num': "int32\n"
                             "sampling number of angle_interval  {}".format(file1),
                
                'angle_sigma': "float32\n"
                               "denominator of weight factor for ADCIGs, mig = mig + mig_value * exp  -(int_angle-float_angle)*(int_angle-float_angle)/angle_sigma  {}".format(file1),
                
                'angle_weight_len': "int32\n"
                                    "weight times/length for weight for ADCIGs   {}".format(file1),
                
                'angle_tapering_power': "float32\n"
                                        "at the limited angle range, I will apply an weight factor on the ADCIGs and migrated image, this parameter is used for cos(\cita) ^ angle_tapering_power  {}".format(file1),
                
                
                'angle_end': "float32\n"
                                  "self.angle_start + self.angle_num * self.angle_interval;",
                
                'angle_mig_bool': "int32\n"
                                  "If it is non-zero, we will compute the ADCIGs and angle-dependent Hessian",
                
                
                'angle_tapering_bool': "int32\n"
                                       "If it is non-zero, we will tapering the migrated image, Hessian operator based on the angle range provided by angle_para_list;"
                                       "if it is zero, all angle parameters are not used for preconditions for ADCIGs and hessian",
                
                
                'mig_stable_coe':  "int32\n"
                                   "the parameter of division imaging condition: grid point in the depth direction, the final stable_coe is computed from mig_stable_coe * source_amplitude (sx, mig_stable_position) for Kirchhoff migration, RTM, one-way migration",
                
                
                'mig_stable_position': "float32\n"
                                       "the parameter of division imaging condition: weight factor for division imaging condition, the final stable_coe is computed from mig_stable_coe * source_amplitude (sx, mig_stable_position) for Kirchhoff migration, RTM, one-way migration",
                
                
                'imaging_conidition_type': "int32\n"
                                           "imaging_conidition_type, if it is 0, I will apply the cross-correlation imaging condition, 1, deconvolution,   2 scattering*",
                
                
                'division_mark': "int32\n"
                                 "if it is 0, I will apply the multiply imaging condition, otherwise I will apply the division imaging condition",
                                 
                'mig_it_start': "int32\n"
                                 "mig_it_start==0, we can not fullly reconstruct the wavefield at the source position, I intend to exclude these wavefields, when I will apply the imaging condition, so I set the it==mig_it_start",                 
                                 
                'mig1': "np.array\n"
                         "The modulus/K migrated  image,  vp=2*mig1, den=mig1+mig2",
 
                'mig2': "np.array\n"
                         "The image of denisty,  vp=2*mig1, den=mig1+mig2",
                                 
                'angle_mig1': "np.array\n"
                         "angle-domain migration 1 for kirchhoff and born",
 
                'angle_mig2': "np.array\n"
                         "angle-domain migration 2 for kirchhoff and born",                 
                }
        
        
        
        
        ##step1    ##recording calss ini value    
        WR.class_dict_ini_log(log_file, self.__class__, w_type="w");  
        
        ##step2    # Populate the dict with variable names, descriptions, and values
        self.dict                   = {}
        WR.class_dict_description(self, descriptions);

        ##step3    ##record the final log
        WR.class_dict_log_file(log_file, self.dict, w_type="a")
            
 
        

class Wave_A_FWI_Para():
    """
    
    """
    def __init__(self, 
                 
                 dims: list=[1, 1, 1],
                 
                 dims_interval: list=[1, 1, 1],
                 
                 obj_func_type: str = "l2",
                 
                 inv_para_list: np.array = [1, 1],
                 
                 update_para_list=[True, True, ],
                 
                 max_iter_list=[200, 200, 200, 200],
                 
                 multi_scale_freq_list=[0, 10, 20, 30],
                 
                 global_step_length_try_list=[0.01, 0.02],
                 
                 global_step_random_shot_ratio = 0.5,
                 
                 grad_mask_up    = [0, 0],
                 grad_mask_down  = [0, 0],
                 grad_mask_left  = [0, 0],
                 grad_mask_right = [0, 0],
                 grad_mask_back  = [0, 0],
                 grad_mask_front = [0, 0],
                 
                 out_info_iter=10,
                 
                 use_cupy = True,
                 
                 log_file="Wave_FWI_Para_log.txt"
                ):

        self.readme     = {}; ##I can record something

        self.name       = self.__class__.__name__
        
        self.log_file   = log_file
        
        self.use_cupy   = use_cupy
        
        
        
        self.cpasnumpy_or = lambda x: x if self.use_cupy else cp.asnumpy(x);

        # Initialize dimensions and intervals
        self.dims = list( map(np.int32, dims) )
        self.dims_interval = list( map(np.float32, dims_interval) )
        
        if len(self.dims) == 3:
            self.nx, self.ny, self.nz = np.int32(self.dims)
            self.dx, self.dy, self.dz = np.float32(self.dims_interval)
        elif len(self.dims) == 2:
            self.nx = np.int32(self.dims[0])
            self.ny = np.int32(1)
            self.nz = np.int32(self.dims[1])
            self.dx = np.float32(self.dims_interval[0])
            self.dy = np.float32(1)
            self.dz = np.float32(self.dims_interval[1])
        else:
            raise ValueError("{}, dims is must be 2 or 3".format(self.name))

        
        self.obj_func_type = obj_func_type;
        
        self.inv_para_list = inv_para_list
        self.inv_para_num  = len(inv_para_list);

        self.update_para_list = update_para_list;

        self.max_iter_list             = max_iter_list
            
        self.multi_scale_freq_list     = multi_scale_freq_list
        self.multi_scale_num           = len(self.multi_scale_freq_list) -1;
    
        if len(self.multi_scale_freq_list)-1 != len(self.max_iter_list):
            raise ValueError(  "len(self.multi_scale_freq_list)-1 != len(self.max_iter_list)" )        

        
        self.global_step_length_try_list = global_step_length_try_list;
        
        self.global_step_random_shot_ratio = global_step_random_shot_ratio;
        
        
        
        self.full_obj_list_arr   = [ np.zeros(iter_max, dtype=np.float32) for iter_max in max_iter_list ];
        
        self.random_obj_list_arr = [ np.zeros(iter_max, dtype=np.float32) for iter_max in max_iter_list ];
        
        self.update_length_list = [];
        
        
        self.out_info_iter = out_info_iter;


        #############self.grad_mask
        nz_radius = self.nz - grad_mask_up[1] - grad_mask_down[1]
        bu        = grad_mask_up[1]   - grad_mask_up[0];
        bd        = grad_mask_down[1] - grad_mask_down[0];
        bu_length = grad_mask_up[0];
        bd_length = grad_mask_down[0];
        att_z = generate_1D_arrary_with_tapering_with_constant(1, nz_radius, bu, bd, bu_length, 0, bd_length, 0,    decay_power=2, decay_type='cos', tapering_or_not=True, module=cp, ).reshape(self.nz, 1);
        
        
        nx_radius = self.nx - grad_mask_left[1] - grad_mask_right[1]
        bl        = grad_mask_left[1]   - grad_mask_left[0];
        br        = grad_mask_right[1] - grad_mask_right[0];
        bl_length = grad_mask_left[0];
        br_length = grad_mask_right[0];
        att_x = generate_1D_arrary_with_tapering_with_constant(1, nx_radius, bl, br, bl_length, 0, br_length, 0,    decay_power=2, decay_type='cos', tapering_or_not=True, module=cp, ).reshape(1, self.nx);
        
        self.grad_mask = cp.tile(att_x, (self.nz, 1)) * cp.tile(att_z, (1, self.nx)); 
    
        self.grad_mask = cp.ascontiguousarray( cp.asarray(self.grad_mask, dtype=cp.float32) );
    
        
        PF.imshow(self.grad_mask, output_name="grad_mask.eps", cmap='seismic');
        WR.write_file("grad_mask.bin", self.grad_mask.T);
        
        #########some des in dictionary
        descriptions = {
                'dims': "When I will initalized the Class, I can set it as vel.T.shape for simplification\n"
                        "Note that our array should be [nz, ny, nx]\n"
                        "but when I will transfer the parameter into a function, I will always use [dx, dy, dz], dims=[nx, ny, nz], [start_x, start_y, start_z], [sx, sy, sz], [gx, gy, gz], [min_x, min_y, min_z], [max_x, max_y, max_z]\n"

                        "All parameters are set as grid point (dx, dy, dz)", 
               
                }
        
        
        
        
        ##step1    ##recording calss ini value    
        WR.class_dict_ini_log(log_file, self.__class__, w_type="w");  
        
        ##step2    # Populate the dict with variable names, descriptions, and values
        self.dict                   = {}
        WR.class_dict_description(self, descriptions);

        ##step3    ##record the final log
        WR.class_dict_log_file(log_file, self.dict, w_type="a")
            

class Wave_tapering_calss():
    """
    
    """
    def __init__(self, 
                 
                 dims: list=[1, 1, 1],
                 
                 dims_interval: list=[1, 1, 1],

                 
                 grad_mask_up    = [0, 0],
                 grad_mask_down  = [0, 0],
                 grad_mask_left  = [0, 0],
                 grad_mask_right = [0, 0],
                 grad_mask_back  = [0, 0],
                 grad_mask_front = [0, 0],

                 use_cupy = True,
                 
                 log_file="Wave_FWI_Para_log.txt"
                ):

        self.readme     = {}; ##I can record something

        self.name       = self.__class__.__name__
        
        self.log_file   = log_file
        
        self.use_cupy   = use_cupy
        
        
        
        self.cpasnumpy_or = lambda x: x if self.use_cupy else cp.asnumpy(x);

        # Initialize dimensions and intervals
        self.dims = list( map(np.int32, dims) )
        self.dims_interval = list( map(np.float32, dims_interval) )
        
        if len(self.dims) == 3:
            self.nx, self.ny, self.nz = np.int32(self.dims)
            self.dx, self.dy, self.dz = np.float32(self.dims_interval)
        elif len(self.dims) == 2:
            self.nx = np.int32(self.dims[0])
            self.ny = np.int32(1)
            self.nz = np.int32(self.dims[1])
            self.dx = np.float32(self.dims_interval[0])
            self.dy = np.float32(1)
            self.dz = np.float32(self.dims_interval[1])
        else:
            raise ValueError("{}, dims is must be 2 or 3".format(self.name))


        #############self.grad_mask
        nz_radius = self.nz - grad_mask_up[1] - grad_mask_down[1]
        bu        = grad_mask_up[1]   - grad_mask_up[0];
        bd        = grad_mask_down[1] - grad_mask_down[0];
        bu_length = grad_mask_up[0];
        bd_length = grad_mask_down[0];
        att_z = generate_1D_arrary_with_tapering_with_constant(1, nz_radius, bu, bd, bu_length, 0, bd_length, 0,    decay_power=2, decay_type='cos', tapering_or_not=True, module=cp, ).reshape(self.nz, 1);
        
        
        nx_radius = self.nx - grad_mask_left[1] - grad_mask_right[1]
        bl        = grad_mask_left[1]   - grad_mask_left[0];
        br        = grad_mask_right[1] - grad_mask_right[0];
        bl_length = grad_mask_left[0];
        br_length = grad_mask_right[0];
        att_x = generate_1D_arrary_with_tapering_with_constant(1, nx_radius, bl, br, bl_length, 0, br_length, 0,    decay_power=2, decay_type='cos', tapering_or_not=True, module=cp, ).reshape(1, self.nx);
        
        self.grad_mask = cp.tile(att_x, (self.nz, 1)) * cp.tile(att_z, (1, self.nx)); 
    
        self.grad_mask = cp.ascontiguousarray( cp.asarray(self.grad_mask, dtype=cp.float32) );
    
        
        PF.imshow(self.grad_mask, output_name="grad_mask.jpg", cmap='seismic');
        WR.write_file("grad_mask.bin", self.grad_mask.T);
        
        #########some des in dictionary
        descriptions = {
                'dims': "When I will initalized the Class, I can set it as vel.T.shape for simplification\n"
                        "Note that our array should be [nz, ny, nx]\n"
                        "but when I will transfer the parameter into a function, I will always use [dx, dy, dz], dims=[nx, ny, nz], [start_x, start_y, start_z], [sx, sy, sz], [gx, gy, gz], [min_x, min_y, min_z], [max_x, max_y, max_z]\n"

                        "All parameters are set as grid point (dx, dy, dz)", 
               
                }
        
        
        
        
        ##step1    ##recording calss ini value    
        WR.class_dict_ini_log(log_file, self.__class__, w_type="w");  
        
        ##step2    # Populate the dict with variable names, descriptions, and values
        self.dict                   = {}
        WR.class_dict_description(self, descriptions);

        ##step3    ##record the final log
        WR.class_dict_log_file(log_file, self.dict, w_type="a")
        
        
      
class Wave_Hes_Para():
    """
    This class holds hessian parameters related to migration, ADCIGs extraction,
    and angle-dependent Kirchhoff modeling for seismic applications.
    
    
    Note that our array should be [nz, ny, nx]
    but when I will transfer the parameter into a function, I will always use [dx, dy, dz] and dims=[nx, ny, nz] and [start_x, start_y, start_z]

    All parameters are set as grid point (dx, dy, dz)
    """
    def __init__(self, 
                 
                 dims: list=[1, 1, 1],
                 
                 dims_interval: list=[1, 1, 1],
                 
                 start_x: int = 0, 
                 start_y: int = 0, 
                 start_z: int = 0, 
                 
                 wx: int = 0, 
                 wy: int = 1, 
                 wz: int = 0,
                 
                 
                 samplingx: int = 0, 
                 samplingy: int = 1, 
                 samplingz: int = 0, 
                 
                 angle_num: int = 0, 
                 angle_start: int = 0,
                 angle_interval: int = 0,
                 angle_tapering_bool: int =1,
                 
                 angle_hes_bool: int = 0,
                 hessian_num: int = 4,
                 
                 z_dir_speedup_ratio: int = 1,

                 compute_or_inv: int = 0, 
                 
                 use_cupy = True,

                 log_file="Wave_Hes_Para_log.txt"
                ):

        self.readme     = {}; ##I can record something

        self.name       = self.__class__.__name__
        
        self.log_file   = log_file
        
        self.use_cupy   = use_cupy
        
        

        self.cpasnumpy_or = lambda x: x if self.use_cupy else cp.asnumpy(x);
        

        # Initialize dimensions and intervals
        self.dims = list( map(np.int32, dims) )
        self.dims_interval = list( map(np.float32, dims_interval) )
        
        if len(self.dims) == 3:
            self.nx, self.ny, self.nz = np.int32(self.dims)
            self.dx, self.dy, self.dz = np.float32(self.dims_interval)
        elif len(self.dims) == 2:
            self.nx = np.int32(self.dims[0])
            self.ny = np.int32(1)
            self.nz = np.int32(self.dims[1])
            self.dx = np.float32(self.dims_interval[0])
            self.dy = np.float32(1)
            self.dz = np.float32(self.dims_interval[1])
        else:
            raise ValueError("{}, dims is must be 2 or 3".format(self.name))
        

        self.start_x            = np.int32(start_x);
        self.start_y            = np.int32(start_y);
        self.start_z            = np.int32(start_z);

        if compute_or_inv==0:
            if self.start_x <0 or self.start_y <0 or self.start_z <0:
                raise ValueError("{}: start_x, start_y, and start_z must be greater or equal to zero respectively.".format(self.name) )

        


        self.wx            = np.int32(wx);
        self.wy            = np.int32(wy);
        self.wz            = np.int32(wz);
        
        if self.wx % 2 == 0 or self.wy % 2 == 0 or self.wz % 2 == 0:
            raise ValueError("{}: wx, wy, and wz must be odd integers.".format(self.name) )
        
        
        
        self.samplingx            = np.int32(samplingx);
        self.samplingy            = np.int32(samplingy);
        self.samplingz            = np.int32(samplingz);

        if self.samplingx == 0 or self.samplingy == 0 or self.samplingz == 0:
            raise ValueError("{}: samplingx, samplingy, and samplingz must not be 0.".format(self.name) )
            
        if compute_or_inv==0:    
            if self.start_x > self.nx or self.start_y > self.ny or self.start_z > self.nz:
                raise ValueError("{}: start_x, start_y, and start_z must not be greater than nx, ny, and nz respectively.".format(self.name) )
                
            self.numx           = np.int32(   (self.nx - self.start_x - 1)   //self.samplingx + 1);
            
            self.numy           = np.int32(   (self.ny - self.start_y - 1)   //self.samplingy + 1);
            
            self.numz           = np.int32(   (self.nz - self.start_z - 1)   //self.samplingz + 1);
            
        else:
            self.numx           = np.int32(   (self.nx - 1)   //self.samplingx + 1);
            
            self.numy           = np.int32(   (self.ny - 1)   //self.samplingy + 1);
            
            self.numz           = np.int32(   (self.nz - 1)   //self.samplingz + 1);
        
        
        
        
        
        if self.numx == 0 or self.numy == 0 or self.numz == 0:
            print("{}: self.numx={}, self.numy={}, self.numz={}".format(self.numx, self.numy, self.numz).format(self.name)  );
            raise ValueError("{}: numx, numy, and numz must not be 0.".format(self.name) )




        self.z_dir_speedup_ratio           = np.int32(z_dir_speedup_ratio);
        
        if self.z_dir_speedup_ratio > self.numz:
            print("Warning: {} z_dir_speedup_ratio {} > self.numz {}, so I set them equal".format(self.name, self.z_dir_speedup_ratio, self.numz) );
            self.z_dir_speedup_ratio = self.numz;

        
        self.hessian_shape       = [self.numz, self.numy, self.numx, self.wz, self.wy, self.wx]
        self.hessian_mem         = np.prod(self.hessian_shape)/1024.0/1024.0;
        self.angle_hessian_shape = [angle_num, self.numz, self.numy, self.numx, self.wz, self.wy, self.wx]
        self.angle_hessian_mem   = np.prod(self.angle_hessian_shape)/1024.0/1024.0;
        
        print(f"{self.name}: self.hessian_shape ={self.hessian_shape} along z y x, wz, wy, wx");
        print(f"self.hessian_mem={self.hessian_mem} M");
        print(f"self.angle_hessian_mem={self.angle_hessian_mem} M");
        
        
        
        
        self.hessian_num        = hessian_num
        ###multi-parameter Hessian operator
        if self.use_cupy:
            if self.hessian_num>=1:
                self.hessian1        = cp.zeros(self.hessian_shape, dtype=cp.float32);
                
            if self.hessian_num>=2:
                self.hessian2        = cp.zeros(self.hessian_shape, dtype=cp.float32);
                
            if self.hessian_num>=3:
                self.hessian3        = cp.zeros(self.hessian_shape, dtype=cp.float32);
                
            if self.hessian_num>=4:
                self.hessian4        = cp.zeros(self.hessian_shape, dtype=cp.float32);
        else:
            if self.hessian_num>=1:
                self.hessian1        = np.zeros(self.hessian_shape, dtype=np.float32);
                
            if self.hessian_num>=2:
                self.hessian2        = np.zeros(self.hessian_shape, dtype=np.float32);
                
            if self.hessian_num>=3:
                self.hessian3        = np.zeros(self.hessian_shape, dtype=np.float32);
                
            if self.hessian_num>=4:
                self.hessian4        = np.zeros(self.hessian_shape, dtype=np.float32);


        self.angle_start            = np.float32(angle_start);
        self.angle_interval         = np.float32(angle_interval);
        self.angle_num              = np.int32(angle_num);
        
        self.angle_end              = np.float32(self.angle_start + self.angle_num * self.angle_interval);
        
        
        
        
        self.angle_tapering_bool    = np.int32( angle_tapering_bool )
        self.angle_hes_bool         = np.int32( angle_hes_bool );
        
        
        ###it is less revision to save object
        if self.use_cupy:
            if self.angle_hes_bool >=1:
                self.angle_hessian1   = cp.zeros([self.angle_num] + self.hessian_shape, dtype=cp.float32);
            
            if self.angle_hes_bool >=2:
                self.angle_hessian2   = cp.zeros([self.angle_num] + self.hessian_shape, dtype=cp.float32);
        else:
            if self.angle_hes_bool >=1:
                self.angle_hessian1   = np.zeros([self.angle_num] + self.hessian_shape, dtype=np.float32);
            
            if self.angle_hes_bool >=2:
                self.angle_hessian2   = np.zeros([self.angle_num] + self.hessian_shape, dtype=np.float32);
        
    
    
        # generate grid point for sampling position
        self.gridx = np.arange(self.start_x, self.start_x + self.numx * self.samplingx, self.samplingx).astype(np.int32)
        
        self.gridy = np.arange(self.start_y, self.start_y + self.numy * self.samplingy, self.samplingy).astype(np.int32)
        
        self.gridz = np.arange(self.start_z, self.start_z + self.numz * self.samplingz, self.samplingz).astype(np.int32)
        
        self.grid_angle = np.arange(self.angle_start, self.angle_start + self.angle_num  * self.angle_interval, self.angle_interval).astype(np.float32)
    
    
        if compute_or_inv==0:
            # Check if any values in gridx, gridy, or gridz exceed or equal nx, ny, or nz respectively
            if np.any(self.gridx >= self.nx):
                raise ValueError("{}: Some values in gridx are greater than or equal to nx.".format(self.name) )
            if np.any(self.gridy >= self.ny):
                raise ValueError("{}: Some values in gridy are greater than or equal to ny.".format(self.name) )
            if np.any(self.gridz >= self.nz):
                raise ValueError("{}: Some values in gridz are greater than or equal to nz.".format(self.name) ) 
    



        descriptions = {
                        'compute_or_inv': "If compute_or_inv=0, I will use the information to compute number of grid by np.int32(   (self.nx - self.start_x - 1)   //self.samplingx + 1);\n"
                        "If compute_or_inv=1, I will use the information to compute number of grid by self.numx           = np.int32(   (self.nx - 1)   //self.samplingx + 1);",
            
                        'dims': "When I will initalized the Class, I can set it as vel.T.shape for simplification\n"
                                "Note that our array should be [nz, ny, nx]\n"
                                "but when I will transfer the parameter into a function, I will always use [dx, dy, dz], dims=[nx, ny, nz], [start_x, start_y, start_z], [sx, sy, sz], [gx, gy, gz], [min_x, min_y, min_z], [max_x, max_y, max_z]\n"

                                "All parameters are set as grid point (dx, dy, dz)\n" 
                                "I can set it as different parameter, when I will implement the image-domain inversion", 
            
                        'start_x': "  start position for computing the Hessian operator in the X direction",
                        
                        'wx': "  size of Hessian (6D) or imaging resolution function (3D) in the X direction",
                        
                        'samplingx': "  sampling interval of Hessian (6D) or imaging resolution function (3D) in the X direction",
                        
                        'numx': "  sampling number of Hessian (6D) or imaging resolution function (3D) in the X direction",
                        
                        
                        
                        
                        'start_y': "  start position for computing the Hessian operator in the Y direction",
                        
                        'wy': "  size of Hessian (6D) or imaging resolution function (3D) in the Y direction",
                        
                        'samplingy': "  sampling interval of Hessian (6D) or imaging resolution function (3D) in the Y direction",
                        
                        'numy': "  sampling number of Hessian (6D) or imaging resolution function (3D) in the Y direction",
                        
                        
                        
                        'start_z': "  start position for computing the Hessian operator in the Z direction",
                        
                        'wz': "  size of Hessian (6D) or imaging resolution function (3D) in the Z direction",
                        
                        'samplingz': "  sampling interval of Hessian (6D) or imaging resolution function (3D) in the Z direction",
                        
                        'numz': "  sampling number of Hessian (6D) or imaging resolution function (3D) in the Z direction",
                        
                        
                        
                        'z_dir_speedup_ratio': "int32\n"
                                                "  there is a loop for z direction, when I want to compute the Hessian matrix.\n"
                                                "  It means that I want to compute [z_dir_speedup_ratio, self.numy, self.numx, self.wz, self.wy, self.wx] for the imaging resolution function (Hessian operator) on device for one depth loop",
                        
                        
                        'hessian_num': "int32\n"
                                        "  if ==4, I will allocate four hessian on numpy/cpu, self.hessian1, self.hessian2, self.hessian3, self.hessian4",
                        
                        
                        'angle_hes_bool': "int32\n"
                                          "  If it is non-zero, we will compute the angle-dependent Hessian, \n"
                                          "  if == 1, allocate angle_hessian1 \n"
                                          "  if == 2 angle_hessian1 and angle_hessian2,  default allocate on numpy/cpu, self.angle_hessian1, self.angle_hessian2",
                         
                        'angle_tapering_bool': "int32\n"
                                               "  This parameter is useless, when I will compute the Hessian, since angle_mapping and angle_tapering only depends on the migraion class",
                         
                        
                        'angle_num': "int32\n"
                                     "  It should be set as the same number of angle_num of migrated image, when I will compute the Hessian matrix\n"
                                     "  However, when I will implement the image-domain inversion, I can set it as another value, but it is smaller than the angle_num of computed Hessian matrix",
                           
                        
                        'angle_interval': "float32\n"
                                       "  This parameter is useless, when I will compute the Hessian, since angle_mapping and angle_tapering only depends on the migraion class \n"
                                       "  Note that this float parameter is used for implementing the image-domain inversion rather than the computation of Hessian matrix.\n"
                                       "  When I will computed the angle-dependent Hessian, it is set as the same parameter of migration class\n"
                                       "  This parameter is useless, when I will compute the Hessian, since it only depends on the migraion class",
                           
                                     
                        'angle_start': "float32\n"
                                       "  This parameter is useless, when I will compute the Hessian, since angle_mapping and angle_tapering only depends on the migraion class \n"
                                       "  Note that this float parameter is used for implementing the image-domain inversion rather than the computation of Hessian matrix.\n"
                                       "  When I will computed the angle-dependent Hessian, it is set as the same parameter of migration class\n"
                                       "  This is a angle_start parameter angle-dependent image-domain inversion,  when the Hessian matrix has been computed\n"
                                       "  We need to compute the start id number through (Hessian.start-migration.start) / angle_interval, the angle_num number is  angle_num",


                        'hessian1': "  H_KK  the dimension of hessian1 is [numz, numy, numx, wz, wy, wx] on numpy.float32,  scale_k_k     = 1.0f *scale    * 1.0f;   scale_k_rou   = 1.0f *scale    * scale_anlge_y; scale_rou_k   = 1.0f *scale    * scale_anlge_x; scale_rou_rou = 1.0f *scale    * scale_anlge_x * scale_anlge_y;", 
                        
                        
                        'hessian2': "  H_K rou  the dimension of hessian1 is [numz, numy, numx, wz, wy, wx] on numpy.float32,  scale_k_k     = 1.0f *scale    * 1.0f;   scale_k_rou   = 1.0f *scale    * scale_anlge_y; scale_rou_k   = 1.0f *scale    * scale_anlge_x; scale_rou_rou = 1.0f *scale    * scale_anlge_x * scale_anlge_y;", 
                        
                        
                        'hessian3': "  H_rou K  the dimension of hessian1 is [numz, numy, numx, wz, wy, wx] on numpy.float32,  scale_k_k     = 1.0f *scale    * 1.0f;   scale_k_rou   = 1.0f *scale    * scale_anlge_y; scale_rou_k   = 1.0f *scale    * scale_anlge_x; scale_rou_rou = 1.0f *scale    * scale_anlge_x * scale_anlge_y;", 
                        
                        
                        'hessian4': "  H_rou rou  the dimension of hessian1 is [numz, numy, numx, wz, wy, wx] on numpy.float32,  scale_k_k     = 1.0f *scale    * 1.0f;   scale_k_rou   = 1.0f *scale    * scale_anlge_y; scale_rou_k   = 1.0f *scale    * scale_anlge_x; scale_rou_rou = 1.0f *scale    * scale_anlge_x * scale_anlge_y;", 
                        
                        
                        'gridx':    "  int32 array of sampling position for X direction", 
                        
                        'gridy':    "  int32 array of sampling position for Y direction", 
                        
                        'gridz':    "  int32 array of sampling position for Z direction", 
                        
                        'angle_hessian1': "  the dimension of hessian1 is [angle_num, numz, numy, numx, wz, wy, wx] on numpy.float32,   mainly used for Kirchhoff/Born migration,    hessian1 is mapped into angle domain", 
                        
                        'angle_hessian2': "  the dimension of hessian2 is [angle_num, numz, numy, numx, wz, wy, wx] on numpy.float32,   mainly used for Kirchhoff/Born migration,    hessian1 is mapped into angle domain", 
                        }
        
        
        
        ##step1    ##recording calss ini value    
        WR.class_dict_ini_log(log_file, self.__class__, w_type="w");  
        
        ##step2    # Populate the dict with variable names, descriptions, and values
        self.dict                   = {}
        WR.class_dict_description(self, descriptions);

        #step3    ##record the final log
        WR.class_dict_log_file(log_file, self.dict, w_type="a")





class Wave_Sou_Para():
    """
    Prior to modeling, migration, Hessian,
    I compute the various wavelet.
    
    Note that our array should be [nz, ny, nx]
    but when I will transfer the parameter into a function, I will always use [dx, dy, dz] and dims=[nx, ny, nz] and [start_x, start_y, start_z]

    All parameters are set as grid point (dx, dy, dz)
    """
    def __init__(self, 
                 source_signal = cp.zeros((5, 2), dtype=cp.float32),
                 
                 ns: int = 200,
                 
                 dt: float = 0.001,
                 
                 mod_dims: int = 2,
                 
                 add_mark: int = 1,
                 
                 source_type: int = 1,
                 
                 eps = "wavelet.eps",
                 
                 use_cupy = True,
                 
                 log_file="Wave_Sou_Para_log.txt",
                ):
        
        self.readme     = {}; ##I can record something
        
        self.name       = self.__class__.__name__
        
        self.log_file   = log_file
        
        self.use_cupy   = use_cupy
        
        
        
        self.cpasnumpy_or = lambda x: x if self.use_cupy else cp.asnumpy(x);

        self.f              = cp.asarray(source_signal).astype(cp.float32);
        
        self.lt             = cp.int32( self.f.shape[0] );
        
        self.num            = cp.int32( self.f.shape[1] );
        
        self.ns             = cp.int32(ns); ##we can compute it by the self
        
        self.dt             = cp.float32(dt);
        
        self.mod_dims       = cp.int32(mod_dims);
        
        self.add_mark       = cp.int32(add_mark);
        
        self.source_type    = cp.int32(source_type);
        
        self.max_ns         = cp.int32(cp.argmax(source_signal[:, 0]).get())
        
        self.f_l1           = cp.float32( cp.sum(cp.abs (source_signal[:, 0]) ).get() );
        
        self.f_l2           = cp.float32( cp.sum( (source_signal[:, 0] * source_signal[:, 0]) ).get() );
        
        
        
        
        ###the filter for the observed recordings, when I will apply the deconvolution imaging condition
        '''
        according to the dot product test, we can derive all filter using different imgaing condition
        '''
        
        # modeling
        if self.mod_dims==2:
            '''two way'''
            self.f2                 =  apply_sqrt_iw_operator(self.f, 0, dt, correct_sign=+1, module=cp);##sqrt(iw)
            
            
            '''born'''
            self.f_b_mig_cc         =  apply_iw_operator(     apply_sqrt_iw_operator(self.f, 0, dt, correct_sign=+1, module=cp), 0, dt, correct_sign=+1, power_num=1, module=cp  ); #sqrt(iw) * iw
            self.f_b_hes_cc_full    =  compute_autocorrelation(self.f_b_mig_cc);
            self.f_b_hes_cc         =  self.f_b_hes_cc_full[self.lt-1:, :];
            
            
            self.b_mig_filter_op    = lambda x, limit_f :  apply_iw_operator(   apply_sqrt_iw_operator(x, 0, dt, correct_sign=-1, limit_f=limit_f, module=cp)    , 0, dt, correct_sign=-1, power_num=1, limit_f=limit_f, module=cp);  ###-iw * sqrt(-iw)
            self.f_b_hes_de         =  apply_iw_operator(     apply_sqrt_iw_operator(self.f_b_mig_cc, 0, dt, correct_sign=-1, module=cp), 0, dt, correct_sign=-1, power_num=1, module=cp  );#sqrt(-iw) * -iw   *  sqrt(iw) * iw
            
            
            '''kirchhoff'''
            self.f_k_mig_cc         =  apply_sqrt_iw_operator(self.f, 0, dt, correct_sign=+1, module=cp);##sqrt(iw)
            self.f_k_hes_cc_full    =  compute_autocorrelation(self.f_k_mig_cc);
            self.f_k_hes_cc         =  self.f_k_hes_cc_full[self.lt-1:, :];
            
            
            self.k_mig_filter_op    = lambda x, limit_f :  apply_sqrt_iw_operator(x, 0, dt, correct_sign=-1, limit_f=limit_f, module=cp);
            self.f_k_hes_de         =  apply_sqrt_iw_operator(self.f_k_mig_cc, 0, dt, correct_sign=-1, module=cp);    #sqrt(-iw)
            
        else:
            '''two way'''
            self.f2 = self.f; # there is no correct, modeling
            
            
            '''born'''
            self.f_b_mig_cc         =  apply_iw_operator(     apply_iw_operator(self.f, 0, dt, correct_sign=+1, module=cp), 0, dt, correct_sign=+1, power_num=1, module=cp  ); #sqrt(iw) * iw
            self.f_b_hes_cc_full    =  compute_autocorrelation(self.f_b_mig_cc);
            self.f_b_hes_cc         =  self.f_b_hes_cc_full[self.lt-1:, :];
            
            
            self.b_mig_filter_op    = lambda x, limit_f :  apply_iw_operator(   apply_iw_operator(x, 0, dt, correct_sign=-1, limit_f=limit_f, module=cp)    , 0, dt, correct_sign=-1, power_num=1, limit_f=limit_f, module=cp);  ###-iw * sqrt(-iw)
            self.f_b_hes_de         =  apply_iw_operator(     apply_iw_operator(self.f_b_mig_cc, 0, dt, correct_sign=-1, module=cp), 0, dt, correct_sign=-1, power_num=1, module=cp  );#sqrt(-iw) * -iw   *  sqrt(iw) * iw
            
            
            '''kirchhoff'''
            self.f_k_mig_cc         =  apply_iw_operator(self.f, 0, dt, correct_sign=+1, module=cp);    #sqrt(iw)
            self.f_k_hes_cc_full    =  compute_autocorrelation(self.f_k_mig_cc);
            self.f_k_hes_cc         =  self.f_k_hes_cc_full[self.lt-1:, :];
            
            
            self.k_mig_filter_op    = lambda x, limit_f :  apply_iw_operator(x, 0, dt, correct_sign=-1, limit_f=limit_f, module=cp);
            self.f_k_hes_de         =  apply_iw_operator(self.f_k_mig_cc, 0, dt, correct_sign=-1, module=cp);    #sqrt(-iw)
        
        
        
        self.f1     = ( cp.cumsum( self.f2  ) * dt     );
        
        
        

        
        ## it is better to set these variables as contiguousarray, so I can apply 1D kernel to process various methods without specifing the  N dimension of the input data.
        if self.num == 1:
            lam= lambda arr: cp.ascontiguousarray(  arr.reshape(self.lt, 1).astype(cp.float32)  );
        else:
            lam= lambda arr: cp.ascontiguousarray(  arr.astype(cp.float32)  )
            
        
        
        
        self.f2             = lam( self.f2 )
        self.f1             = lam( self.f1 )
            
        
        
        
        self.f_b_mig_cc     = lam( self.f_b_mig_cc )
        self.f_k_mig_cc     = lam( self.f_k_mig_cc )
        
        
        self.f_b_mod        = self.f_b_mig_cc
        self.f_k_mod        = self.f_k_mig_cc
            
            
            
        self.f_b_hes_cc     = lam( self.f_b_hes_cc )
        self.f_k_hes_cc     = lam( self.f_k_hes_cc )
            
            
        self.f_b_hes_de     = lam( self.f_b_hes_de )
        self.f_k_hes_de     = lam( self.f_k_hes_de )
        

        

        descriptions = {
                        'dims': "When I will initalized the Class, I can set it as vel.shape for simplification\n"
                                "Note that [nx, ny, nz]\n",
            
                        'lt': "the total length",
                        
                        'num': 'the source numbers for one modeling',
                        
                        'ns': "wavelet length of source signal",
                        
                        'max_ns':"length of max value in the source signal",
            
                        'add_mark': "add_mark=0,  P = P + source, else: P = P= source",
                        
                        'source_type': "used for elastic wave equation",
        
        
                        'mod_dims': "value=2 or 3: 2D or 3D using different filters for source signal and observed data \n"
                                    "when I will apply the deconvolution imaging condition, I must apply the filter on the observed data",            
        
        
        
                        'f': "Initial source wavelet , it is better to set these variables as contiguousarray, so I can apply 1D kernel to process various methods without specifing the  N dimension of the input data.",
                        
                        'f2': "2D case: Initial source wavelet f with sqrt(iw);    3D case: Initial source wavelet, generally I will use it as the source signal of the second order two-way equation (TW)",
                        
                        'f1': "The time-integral of f2 for the first-order wave equation, since there is an intergral operator in the source term, I must apply this integral operator, generally I will use it as the source signal of velocity-stress two-way equation (TW)",
                   
                        
                   
                        'b_mig_filter_op': "When I will apply the ray-based Born migration, I need to apply the filter on the observed data",
                        
                        'k_mig_filter_op': "When I will apply the ray-based Kirchhoff migration, I need to apply the filter on the observed data",
                        
                        'f_b_mig_cc': "The source signal of the cross-correlation imaging condition,  We do not need to apply the filter on the recordings, when I apply the cross-correlation imaging condition, this is the source signal of ray-Born modeling",

                        'f_k_mig_cc': "The source signal of the cross-correlation imaging condition, We do not need to apply the filter on the recordings, when I apply the cross-correlation imaging condition, this is the source signal of ray-Kirchhoff modeling",
                   
                        
                   
                        'f_b_hes_cc': "The source signal for CC-Hessian, Born migration",
                        
                        'f_k_hes_cc': "The source signal for CC-Hessian, Kirchhoff migration",
                        
                   
                        'f_b_hes_de': "The source signal for DE-Hessian, Born migration",
                        
                        'f_k_hes_de': "The source signal for DE-Hessian, Kirchhoff migration",
                        
                        }
        
        
        ##step1    ##recording calss ini value    
        WR.class_dict_ini_log(log_file, self.__class__, w_type="w");  
        
        ##step2    # Populate the dict with variable names, descriptions, and values
        self.dict                   = {}
        WR.class_dict_description(self, descriptions);

        ##step3    ##record the final log
        WR.class_dict_log_file(log_file, self.dict, w_type="a")




     
        
class Wave_Acq_Para():
    '''
    
    Note that our array should be [nz, ny, nx]
    but when I will transfer the parameter into a function, I will always use [dx, dy, dz] and dims=[nx, ny, nz] and [start_x, start_y, start_z]

    All parameters are set as grid point (dx, dy, dz)
    
    '''
    def __init__(self, 
                 sx_start: int=0,  sx_interval: int=0, sx_number: int=1, 
                 
                 sy_start: int=0,  sy_interval: int=0, sy_number: int=1, 
                 
                 sz_start: int=0,  sz_interval: int=0, sz_number: int=1, 
                 
                 gx_start: int=0,  gx_interval: int=0, gx_number: int=1, 
                 
                 gy_start: int=0,  gy_interval: int=0, gy_number: int=1, 
                 
                 gz_start: int=0,  gz_interval: int=0, gz_number: int=1, 
                 
                 gx_offset: int=0, gy_offset: int=0,
                 
                 x_aperture: int=0, y_aperture: int=0,
                 
                 full_model_bool: bool=True,
                 
                 use_cupy = True, 
                 
                 log_file: str="Wave_Acq_Para_log.txt",
                 ):

        self.readme     = {}; ##I can record something

        self.name       = self.__class__.__name__
        
        self.log_file   = log_file
        
        self.use_cupy   = use_cupy
        
        
            
        self.cpasnumpy_or = lambda x: x if self.use_cupy else cp.asnumpy(x);

        self.sx_start = sx_start
        self.sx_interval = sx_interval
        self.sx_number = sx_number
        
        self.sy_start = sy_start
        self.sy_interval = sy_interval
        self.sy_number = sy_number
        
        self.sz_start = sz_start
        self.sz_interval = sz_interval
        self.sz_number = sz_number
        
        # 接收器参数
        self.gx_start = gx_start
        self.gx_interval = gx_interval
        self.gx_number = gx_number
        
        self.gy_start = gy_start
        self.gy_interval = gy_interval
        self.gy_number = gy_number
        
        self.gz_start = gz_start
        self.gz_interval = gz_interval
        self.gz_number = gz_number
        
        # 偏移量
        self.gx_offset = gx_offset
        self.gy_offset = gy_offset
        
        
        ##migration aperture
        self.x_aperture = x_aperture
        self.y_aperture = y_aperture
        
        ##full model
        self.full_model_bool = full_model_bool
        
        

        #####################
        self.names = ['sx_start', 'sx_interval', 'sx_number', 
               'sy_start', 'sy_interval', 'sy_number', 
               'sz_start', 'sz_interval', 'sz_number', 
               'gx_start', 'gx_interval', 'gx_number', 
               'gy_start', 'gy_interval', 'gy_number', 
               'gz_start', 'gz_interval', 'gz_number', 
               'gx_offset', 'gy_offset',
               'x_aperture', 'y_aperture',
               'full_model_bool', 
               ]
        
        
        descriptions = {
                        'gx_offset': "When the receivers are moved with the source position",
                        
                        'gy_offset': "When the receivers are moved with the source position",
                        
                        
                        'x_aperture': "x_aperture and y_aperture aim to extend range of velocity model for modeling, migration, and inversion,  it is same meaning with the migration aperture in the Kichhoff migration", 
                        
                        'x_aperture': "y_aperture and y_aperture aim to extend range of velocity model for modeling, migration, and inversion,  it is same meaning with the migration aperture in the Kichhoff migration",
                        
                        
                        'full_model_bool': "if it is true,  when I will implement the modeling, migration, and inversion, I will use the max range of velocity for modeling, migration, and inversion", 
                        
                        }
        
        
        ##step1    ##recording calss ini value    
        WR.class_dict_ini_log(log_file, self.__class__, w_type="w");  
        
        ##step2    # Populate the dict with variable names, descriptions, and values
        self.dict                   = {}
        WR.class_dict_description(self, descriptions);

        ##step3    ##record the final log
        WR.class_dict_log_file(log_file, self.dict, w_type="a")
    
    
    def display(self):
        print(f"Source X: start={self.sx_start}, interval={self.sx_interval}, number={self.sx_number}")
        print(f"Source Y: start={self.sy_start}, interval={self.sy_interval}, number={self.sy_number}")
        print(f"Source Z: start={self.sz_start}, interval={self.sz_interval}, number={self.sz_number}")
        print(f"Receiver X: start={self.gx_start}, interval={self.gx_interval}, number={self.gx_number}")
        print(f"Receiver Y: start={self.gy_start}, interval={self.gy_interval}, number={self.gy_number}")
        print(f"Receiver Z: start={self.gz_start}, interval={self.gz_interval}, number={self.gz_number}")
        print(f"Offset X: {self.gx_offset}, Offset Y: {self.gy_offset}")
        print(f"aperture X: {self.x_aperture}, Offset Y: {self.y_aperture}")
        print(f"full_model_bool: {self.full_model_bool}")
        


