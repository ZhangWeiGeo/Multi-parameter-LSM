
import sys
import os

parameters_arr = sys.argv[:];    # parameters_arr = sys.argv[1:];
if len(parameters_arr) ==1:
    
    soft_tv_scale         = 0.01
    beta_tv               = 0.01
    

    soft_l1_scale         = 0
    beta_l1               = 0
    
    smooth_l2_scale       = 0.0001
    
    '''if or not use analitical step-length'''
    multi_step_length     = True
    
    
    '''ADAM or CG: projected GD and reparameterization'''
    max_iter              =  500
    admm_inner_iter_num   =  8
    
    train_snr             =  30
    
else:
    soft_tv_scale         = float(parameters_arr[1])
    beta_tv               = float(parameters_arr[2])
    

    soft_l1_scale         = float(parameters_arr[3])
    beta_l1               = float(parameters_arr[4])
    
    
    smooth_l2_scale       = float(parameters_arr[5])
    
    '''if or not use analitical step-length'''
    multi_step_length     = eval(parameters_arr[6])
    
    
    '''ADAM or CG: projected GD and reparameterization'''
    max_iter              =  int(parameters_arr[7])
    admm_inner_iter_num   =  int(parameters_arr[8])
    
    train_snr             =  int(parameters_arr[9])
    #runpy3 $script_number $time_minutes $memory_value "$run_path$PYTHON_FILE" "$soft_tv_scale" "$beta_tv" "$soft_l1_scale" "$beta_l1" "$multi_step_length" "$max_iter" "$admm_inner_iter_num" "$train_snr"

code_debug = True;
local_vars = locals();

sys_info_dict         = {};
sys_info_dict['code'] = f"code:{parameters_arr[0]}";

for key, value in list(local_vars.items()): 
    if isinstance(value, (int, float, bool, list, str)):
        if key not in ["sys_info_dict", "parameters_arr"]:  
            sys_info_dict[key] = f"{value} (type: {type(value).__name__})"

for key, value in sys_info_dict.items():
    print(f"{key}: {value}\n");