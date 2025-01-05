
import sys
import os

sys.path.append("./func_temp/");
sys.path.append("./func/");
sys.path.append("./func_ID/");

from Wave_class import *
from wave_common import *


from lib_sys import *

device, cp_device  = WR.get_device_ini(0)
WR.get_seed()

# from lib_pylops import *


import torch

from ID_cuda       import NonStationaryConvolve2D
from ID_cuda       import NonStationaryConvolve3D
from torchoperator import TorchOperator as my_TorchOperator








from main_para    import *


'''get some path information'''
main_path = os.path.dirname( os.path.abspath(__file__) ) + "/"

ini_path    = main_path + "./ini/";    
ini_path=WR.get_unique_dir(ini_path);
os.makedirs(ini_path, exist_ok=True);

inv_path    = main_path + "./inv/";
inv_path=WR.get_unique_dir(inv_path);
os.makedirs(inv_path, exist_ok=True);

inv_log_file = inv_path + "log.txt";  log_file = inv_log_file

WR.write_txt(inv_log_file, f"ini_path={ini_path}");
WR.write_txt(inv_log_file, f"inv_path={inv_path}");
WR.dict_write_as_txt(inv_log_file, sys_info_dict, w_type='a+');

path_dict={}
path_dict['ray-data'] = "./data/"
mod_obj  = WR.class_read(path_dict['ray-data'] + "mod_para_obj.pkl");
mig_obj  = WR.class_read(path_dict['ray-data'] + "ray_b_mig_de.pkl");
hes_inv_obj  = WR.class_read(path_dict['ray-data'] + "ray_b_hes_de.pkl");



'''step1: I will get the part of migrated image'''
inv_mig_index = (slice(hes_inv_obj.start_z, hes_inv_obj.start_z+hes_inv_obj.nz), slice(hes_inv_obj.start_x, hes_inv_obj.start_x+hes_inv_obj.nx));

mig_arr1        =  mig_obj.mig1[inv_mig_index];      ##K migrated image
mig_arr2        =  mig_obj.mig2[inv_mig_index];     ##den migrated image
rel_vp_arr      =  mod_obj.rel_vp[inv_mig_index];    ##
rel_den_arr     =  mod_obj.rel_den[inv_mig_index];   ##




mig_arr1     = TF.array_to_float32_contiguous(mig_arr1.T     );
mig_arr2     = TF.array_to_float32_contiguous(mig_arr2.T     );
rel_vp_arr   = TF.array_to_float32_contiguous(rel_vp_arr.T   );
rel_den_arr  = TF.array_to_float32_contiguous(rel_den_arr.T  );

psfs1  =  np.squeeze(hes_inv_obj.hessian1, axis=(1, 4)).transpose(1, 0, 3, 2);
psfs2  =  np.squeeze(hes_inv_obj.hessian2, axis=(1, 4)).transpose(1, 0, 3, 2);
psfs3  =  np.squeeze(hes_inv_obj.hessian3, axis=(1, 4)).transpose(1, 0, 3, 2);
psfs4  =  np.squeeze(hes_inv_obj.hessian4, axis=(1, 4)).transpose(1, 0, 3, 2);




'''model_paramerization'''
model_paramerization = 1;

if model_paramerization!=0:
    file1 = "model_paramerization is {}, (0:K-denisty, 1: velocity-density, 2:impedance-velocity, 3:**)".format(model_paramerization);
    mig_arr1[:], mig_arr2[:] = acoustic_model_paramerization_ref(mig_arr1, mig_arr2, covering_variable=False, model_paramerization_mark=model_paramerization);
    
    psfs1[:], psfs2[:], psfs3[:], psfs4[:] = acoustic_model_paramerization_hessian( psfs1,  psfs2, psfs3, psfs4, covering_variable=False, model_paramerization_mark=model_paramerization);
    
    # rel_para1, rel_para2 = acoustic_model_paramerization_ref(rel_vp_arr, rel_den_arr, covering_variable=False, model_paramerization_mark=model_paramerization);
else:
    rel_para1, rel_para2 = rel_vp_arr, rel_den_arr;

rel_para1, rel_para2 = rel_vp_arr, rel_den_arr;




mig_arr1_torch      = TF.array_to_torch(mig_arr1, device);
mig_arr2_torch      = TF.array_to_torch(mig_arr2, device);
rel_para1_torch     = TF.array_to_torch(rel_para1, device);
rel_para2_torch     = TF.array_to_torch(rel_para2, device);

inv_dims            = list( mig_arr1_torch.shape  );
binname             = WR.bin_name(mig_arr1_torch);


'''normalized blurring functions'''
psf_value_normalized = True;
forward_scale = 1.0;
if psf_value_normalized:
    forward_scale   = max(np.max(np.abs(psfs1)), np.max(np.abs(psfs2)), np.max(np.abs(psfs3)), np.max(np.abs(psfs4)));  
    psfs1            = psfs1/forward_scale ;
    psfs2            = psfs2/forward_scale ;
    psfs3            = psfs3/forward_scale ;
    psfs4            = psfs4/forward_scale ;


psfs1 = TF.array_to_float32_contiguous(psfs1);
psfs2 = TF.array_to_float32_contiguous(psfs2);
psfs3 = TF.array_to_float32_contiguous(psfs3);
psfs4 = TF.array_to_float32_contiguous(psfs4);

file1 = WR.array_info(psfs1, "psf1");
file2 = WR.array_info(psfs2, "psf2");
file3 = WR.array_info(psfs3, "psf3");
file4 = WR.array_info(psfs4, "psf4");
WR.write_txt(inv_log_file, file1 + file2 + file3 + file4);
    


psfx  = hes_inv_obj.gridx;  psfx = psfx - psfx[0];
psfz  = hes_inv_obj.gridz;  psfz = psfz - psfz[0];





cp_op1 = NonStationaryConvolve2D(dims=inv_dims, hs=cp.asarray(psfs1).astype(cp.float32), ihx=psfx, ihz=psfz, dtype=cp.float32, engine="cuda");
cp_op2 = NonStationaryConvolve2D(dims=inv_dims, hs=cp.asarray(psfs2).astype(cp.float32), ihx=psfx, ihz=psfz, dtype=cp.float32, engine="cuda");
cp_op3 = NonStationaryConvolve2D(dims=inv_dims, hs=cp.asarray(psfs3).astype(cp.float32), ihx=psfx, ihz=psfz, dtype=cp.float32, engine="cuda");
cp_op4 = NonStationaryConvolve2D(dims=inv_dims, hs=cp.asarray(psfs4).astype(cp.float32), ihx=psfx, ihz=psfz, dtype=cp.float32, engine="cuda");

forward_cupy_op_list        = [ cp_op1._forward, cp_op2._forward, cp_op3._forward, cp_op4._forward];
adjoint_cupy_op_list        = [ cp_op1._adjoint, cp_op3._adjoint, cp_op2._adjoint, cp_op4._adjoint ];


WR.write_txt(inv_log_file, f"dims={inv_dims}");
WR.write_txt(inv_log_file, f"ihx={psfx}");
# WR.write_txt(inv_log_file, f"ihy={grid_psfz}");
WR.write_txt(inv_log_file, f"ihz={psfz}");



'''define forward and adjoint operator in TorchOperator of pylops'''
pylops_torch_for_op1               = lambda x: TorchOperator(cp_op1, device=device).apply(x);
pylops_torch_for_op2               = lambda x: TorchOperator(cp_op2, device=device).apply(x);
pylops_torch_for_op3               = lambda x: TorchOperator(cp_op3, device=device).apply(x);
pylops_torch_for_op4               = lambda x: TorchOperator(cp_op4, device=device).apply(x);
forward_torch_for_op_list   = [pylops_torch_for_op1, pylops_torch_for_op2, pylops_torch_for_op3, pylops_torch_for_op4]


pylops_torch_adj_op1               = lambda x: TorchOperator(cp_op1.T, device=device).apply(x);
pylops_torch_adj_op2               = lambda x: TorchOperator(cp_op2.T, device=device).apply(x);
pylops_torch_adj_op3               = lambda x: TorchOperator(cp_op3.T, device=device).apply(x);
pylops_torch_adj_op4               = lambda x: TorchOperator(cp_op4.T, device=device).apply(x);
adjoint_torch_for_op_list   = [pylops_torch_adj_op1, pylops_torch_adj_op3, pylops_torch_adj_op2, pylops_torch_adj_op4]



'''define forward operator list using my_TorchOperator'''
torch_for_op1               = lambda x: my_TorchOperator(cp_op1._forward, cp_op1._adjoint, device=cp_device, devicetorch=device).apply(x);
torch_for_op2               = lambda x: my_TorchOperator(cp_op2._forward, cp_op2._adjoint, device=cp_device, devicetorch=device).apply(x);
torch_for_op3               = lambda x: my_TorchOperator(cp_op3._forward, cp_op3._adjoint, device=cp_device, devicetorch=device).apply(x);
torch_for_op4               = lambda x: my_TorchOperator(cp_op4._forward, cp_op4._adjoint, device=cp_device, devicetorch=device).apply(x);
forward_torch_for_op_list   = [torch_for_op1, torch_for_op2, torch_for_op3, torch_for_op4];


'''define adjoint operator list using my_TorchOperator'''
torch_adj_op1           = lambda x: my_TorchOperator(cp_op1._adjoint, cp_op1._forward, device=cp_device, devicetorch=device).apply(x);
torch_adj_op2           = lambda x: my_TorchOperator(cp_op2._adjoint, cp_op2._forward, device=cp_device, devicetorch=device).apply(x);
torch_adj_op3           = lambda x: my_TorchOperator(cp_op3._adjoint, cp_op3._forward, device=cp_device, devicetorch=device).apply(x);
torch_adj_op4           = lambda x: my_TorchOperator(cp_op4._adjoint, cp_op4._forward, device=cp_device, devicetorch=device).apply(x);

'''if you want to compute the gradient, you need tranpose this one'''
adjoint_torch_for_op_list   = [torch_adj_op1, torch_adj_op3, torch_adj_op2, torch_adj_op4];

## adjoint_torch_for_op_list       = [torch_adj_op1, torch_adj_op2, torch_adj_op3, torch_adj_op4] ##if you want to compute the gradient, you need tranpose this one





'''The firs step is to check the error of forward operator'''


mig_sys_part1 = forward_scale * torch_for_op1( rel_para1_torch  )
mig_sys_part2 = forward_scale * torch_for_op2( rel_para2_torch  )
mig_sys_part3 = forward_scale * torch_for_op3( rel_para1_torch  )
mig_sys_part4 = forward_scale * torch_for_op4( rel_para2_torch  )



[mig_sys_1, mig_sys_2] = apply_operator_multiparameter_ND(forward_torch_for_op_list, [rel_para1_torch, rel_para2_torch]);


mig_sys_1 = mig_sys_part1 + mig_sys_part2 # TF.array_to_numpy()
mig_sys_2 = mig_sys_part3 + mig_sys_part4 # TF.array_to_numpy()






'''plot the error of forward operator'''

plot_min=0.6 * mig_arr1_torch.min().item();
plot_max=0.6 * mig_arr1_torch.max().item();

name='mig1-' + binname;
eps_name   =  ini_path   + name + ".jpg";
data_name  =  ini_path   + name + ".bin";
PF.imshow(mig_arr1_torch.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);


name='mig2-' + binname;
eps_name   =  ini_path   + name + ".jpg";
data_name  =  ini_path   + name + ".bin";
PF.imshow(mig_arr2_torch.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);

name='mig1-res-' + binname
eps_name   =  ini_path   + name + ".jpg";
data_name  =  ini_path   + name + ".bin";
PF.imshow(mig_arr1_torch.T-mig_sys_1.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);


name='mig2-res-' + binname
eps_name   =  ini_path   + name + ".jpg";
data_name  =  ini_path   + name + ".bin";
PF.imshow(mig_arr2_torch.T-mig_sys_2.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);


# name='mig1-res-nor-' + binname
# eps_name   =  ini_path   + name + ".jpg"
# data_name  =  ini_path   + name + ".bin"
# PF.imshow(mig_arr1_torch/np.max(mig_arr1_torch)-mig_sys_1/np.max(mig_sys_1), eps_name, data_name, vmin=-1.0, vmax=+1.0);

# name='mig2-res-nor-' + binname
# eps_name   =  ini_path   + name + ".jpg"
# data_name  =  ini_path   + name + ".bin"
# PF.imshow(mig_arr2_torch/np.max(mig_arr2_torch)-mig_sys_2/np.max(mig_sys_2), eps_name, data_name, vmin=-1.0, vmax=+1.0);


'''plot systhnetic migrated image'''
plot_min=0.6 * mig_sys_1.min().item();
plot_max=0.6 * mig_sys_1.max().item();

name='mig1-sys-' + binname
eps_name   =  ini_path   + name + ".jpg";
data_name  =  ini_path   + name + ".bin";
PF.imshow(mig_sys_1.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);


name='mig2-sys-' + binname
eps_name   =  ini_path   + name + ".jpg";
data_name  =  ini_path   + name + ".bin";
PF.imshow(mig_sys_2.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);


name='mig-sys-part1-' + binname
eps_name   =  ini_path   + name + ".jpg";
data_name  =  ini_path   + name + ".bin";
PF.imshow(mig_sys_part1.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);

name='mig-sys-part2-' + binname
eps_name   =  ini_path   + name + ".jpg";
data_name  =  ini_path   + name + ".bin";
PF.imshow(mig_sys_part2.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);

name='mig-sys-part3-' + binname
eps_name   =  ini_path   + name + ".jpg";
data_name  =  ini_path   + name + ".bin";
PF.imshow(mig_sys_part3.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);

name='mig-sys-part4-' + binname
eps_name   =  ini_path   + name + ".jpg";
data_name  =  ini_path   + name + ".bin";
PF.imshow(mig_sys_part4.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);



multiparameter_forward= lambda x: apply_operator_multiparameter_ND(forward_torch_for_op_list, x);
multiparameter_adjoint= lambda x: apply_operator_multiparameter_ND(adjoint_torch_for_op_list, x);




'''dot product test'''
start_time = time.time();

loop_num=1;
for i in range(0, loop_num):
    
    x1  = [ torch.randn_like(mig_arr1_torch), torch.randn_like(mig_arr2_torch) ];

    WR.dot_test(multiparameter_forward, multiparameter_adjoint, x1, tolerance=1e-6, output_bool=True)



def solver_plot_func(para_arr, output_name, plot_min=0, plot_max=0):
    
    if plot_min==0 and plot_max==0:
        plot_min = 0.1 * para_arr.min().item();
        plot_max = 0.1 * para_arr.max().item();
        
    PF.imshow(para_arr.T, d1=mod_obj.dx*0.001, d2=mod_obj.dz*0.001, 
                           vmin=plot_min, vmax=plot_max, 
                           output_name=output_name);

'''code'''
data_term_forward = multiparameter_forward
data_term_adjoint = multiparameter_adjoint

ini_inv_para_list         = [ 
                    torch.zeros( list(rel_para1_torch.shape) ).to(device), 
                   
                    torch.zeros( list(rel_para2_torch.shape) ).to(device) 
                ]

obs_shot_list = [mig_arr1_torch, mig_arr2_torch]
obs_l2        = WR.list_l2_norm_square(obs_shot_list)

loss_list   =[]
niter       = 500
for iter_loop in range(0, niter):
    start_time = time.time();

    mig_sys_list  = data_term_forward(ini_inv_para_list);
    
    mig_res_list  = WR.list_arr_math(mig_sys_list, obs_shot_list, "sub");
    
    loss_list.append(     WR.list_l2_norm_square(mig_res_list)    )
    
    mig_grad_list       = data_term_adjoint(  mig_res_list )
    
    try_sys_list        = data_term_forward(mig_grad_list)
    
    
    A              = WR.list_arr_math(try_sys_list, mig_res_list, "mul")
    B              = WR.list_arr_math(try_sys_list, try_sys_list, "mul")
    
    step_length    = WR.list_arr_sum(A) /WR.list_arr_sum(B)
    
    
    for idx in range(0, len(ini_inv_para_list) ):
        ini_inv_para_list[idx][...] -= step_length * mig_grad_list[idx][...]

    end_time       = time.time();
    
    ## Compute the total execution time
    execution_time = end_time - start_time
    print( "iter={}, res={}, step_length={}".format(iter_loop, loss_list[iter_loop]/obs_l2, step_length) )
    print( "Execution time numba:{} seconds\n".format(execution_time) )
    



'''plot output inverted image '''
# inv1, inv2     = solver_dict['inv_para_list'];

inv1, inv2     = ini_inv_para_list

plot_min = 0.6 * inv1.min().item();
plot_max = 0.6 * inv1.max().item();

name       = 'inv1-' + binname
eps_name   =  inv_path   + name + ".jpg"
data_name  =  inv_path   + name + ".bin"
PF.imshow(inv1.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);

plot_min = 0.6 * inv2.min().item();
plot_max = 0.6 * inv2.max().item();

name       = 'inv2-' + binname
eps_name   =  inv_path   + name + ".jpg"
data_name  =  inv_path   + name + ".bin"
PF.imshow(inv2.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);


'''plot migrated image'''
mig1, mig2     = mig_arr1_torch, mig_arr2_torch;

plot_min = 0.6 * mig1.min().item();
plot_max = 0.6 * mig1.max().item();

name       = 'mig1-' + binname
eps_name   =  inv_path   + name + ".jpg";
data_name  =  inv_path   + name + ".bin";
PF.imshow(mig1.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);

name       = 'mig2-' + binname
eps_name   =  inv_path   + name + ".jpg"
data_name  =  inv_path   + name + ".bin"
PF.imshow(mig2.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);



'''output residual in the image-domain'''
# res_list = solver_dict['res_list'];
res_list   = mig_res_list
mig1_res, mig2_res =    res_list[0], res_list[1] 

plot_min = 0.6 * mig1.min().item();
plot_max = 0.6 * mig1.max().item();

name       = 'inv-mig1-res-' + binname
eps_name   =  inv_path   + name + ".jpg"
data_name  =  inv_path   + name + ".bin"
PF.imshow(mig1_res.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);

name       = 'inv-mig2-res-' + binname
eps_name   =  inv_path   + name + ".jpg"
data_name  =  inv_path   + name + ".bin"
PF.imshow(mig2_res.T, output_name=eps_name, vmin=plot_min, vmax=plot_max);



'''plot True reflectivity'''
ref1, ref2     = rel_para1, rel_para2;

plot_min = 0.6 * ref1.min().item();
plot_max = 0.6 * ref1.max().item();

name       = 'rel-vp-' + binname
eps_name   =  inv_path   + name + ".jpg"
data_name  =  inv_path   + name + ".bin"
PF.imshow(ref1.T, output_name=eps_name, vmin=plot_min, vmax=plot_max, cmap='seismic', colorbar_label='Relative velocity perturbation');

name       = 'rel-den-' + binname
eps_name   =  inv_path   + name + ".jpg"
data_name  =  inv_path   + name + ".bin"
PF.imshow( ref2.T, output_name=eps_name, vmin=plot_min, vmax=plot_max, cmap='seismic', colorbar_label='Relative density perturbation');