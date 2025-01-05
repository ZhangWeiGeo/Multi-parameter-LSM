






# import sys
# sys.path.append("/home/zhangjiwei/pyfunc/Library")

# from Library import *


# sys.path.append("/home/zhangjiwei/pyfunc/acq_info/vp-true-400-200/")

# from path_info    import *

## other parameters are same with the migration operator
hes_inv_obj = Wave_Hes_Para(
    
                dims=hes_obj.dims,
                dims_interval=hes_obj.dims_interval,
                
                start_x = hes_obj.start_x, 
                start_y = hes_obj.start_y, 
                start_z = hes_obj.start_z, 
                
                wx = hes_obj.wx, 
                wy = hes_obj.wy, 
                wz = hes_obj.wz,
                

                samplingx = hes_obj.samplingx*1, 
                samplingy = hes_obj.samplingy, 
                samplingz = hes_obj.samplingz*1, 
                
                angle_start    = hes_obj.angle_start, 
                angle_interval = hes_obj.angle_interval, 
                angle_num      = hes_obj.angle_num, 
               
                angle_hes_bool = hes_obj.angle_hes_bool,
                hessian_num = hes_obj.hessian_num,
                
                z_dir_speedup_ratio = hes_obj.z_dir_speedup_ratio,
                
                compute_or_inv =1,
                
                use_cupy = False,
                
                log_file=path_dict['log'] +"Hes_inv_Para2_log.txt"
                )




