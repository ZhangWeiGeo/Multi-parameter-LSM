


import write_read_func as WR
import torch_func as TF


import re
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.ticker import FuncFormatter

from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.backends.backend_agg as agg

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MaxNLocator
from PIL import Image



import os
# import sys
# import math


import cupy as cp
import numpy as np
import torch

def write_txt(filename, output_some, w_type='a+', print_bool=False):
    data=open(filename, w_type); ##'w+' 'a+'
    print(output_some, file=data)
    data.close()
    if print_bool:
        print(output_some);

def plot_polynomial(alpha_points, f_values, polynomial, output_name):
    # Generate values for plotting the fitted polynomial
    alpha_range = np.linspace(min(alpha_points), max(alpha_points), 100)
    f_range = polynomial(alpha_range)
    
    # Plot the original points and the fitted polynomial
    plt.figure(figsize=(8, 6))
    plt.plot(alpha_range, f_range, label='Fitted polynomial', color='blue')
    plt.scatter(alpha_points, f_values, color='red', label='Original data points', zorder=5)
    
    # Calculate the optimal step length from the polynomial coefficients
    a, b = polynomial.c[0], polynomial.c[1]  # Coefficients a and b from the quadratic term
    optimal_alpha = -b / (2 * a)
    optimal_f_value = polynomial(optimal_alpha)  # Calculate the function value at alpha_star
    
    # Plot the optimal step length
    plt.axvline(x=optimal_alpha, color='green', linestyle='--', label=f'Optimal step length: {optimal_alpha:.2f}')
    plt.scatter(optimal_alpha, optimal_f_value, color='green', zorder=6)  # Mark the optimal point on the curve
    plt.title('Parabolic Fitting for Step Length')
    plt.xlabel('Step Length')
    plt.ylabel('Objective Function Value')
    plt.legend()
    plt.grid(True)
    
    # Save the figure as both EPS and JPG
    plt.savefig(output_name + '.eps', format='eps', dpi=300)
    plt.savefig(output_name + '.png', format='jpg', dpi=300)
    # plt.show()
    

def gif_from_jpgs_lists(jpg_lists, gif_filename="wavefield.gif", duration=500):
    '''
    PF.gif_from_jpgs_lists( [source_jpg_list[1:], receiver_jpg_list[1:], RTM_jpg_list[1:]], gif_filename=output_path + "s_r_rtm.gif" );
    '''
    images = []
    
    # 假设所有的列表长度相同
    num_frames = len(jpg_lists[0])  # 假定三个列表长度相同
    
    for i in range(num_frames):
        img_row = []
        for jpg_list in jpg_lists:
            img_path = jpg_list[i]
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img_row.append(img)
            else:
                print(f"Warning: {img_path} not found. Skipping this file.")
        
        if len(img_row) == len(jpg_lists):  # 确保每一组都有三张图像
            # 水平拼接图片
            widths, heights = zip(*(img.size for img in img_row))
            total_width = sum(widths)
            max_height = max(heights)
            
            new_img = Image.new('RGB', (total_width, max_height))
            
            x_offset = 0
            for img in img_row:
                new_img.paste(img, (x_offset, 0))
                x_offset += img.width
            
            images.append(new_img)
    
    if images:
        images[0].save(gif_filename, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"GIF saved as {gif_filename}")
    else:
        print("No valid images found to create GIF.")

def gif_from_jpgs(filenames_list, gif_filename="wavefield.gif", duration=500):
    '''
    PF.gif_from_jpgs( receiver_jpg_list[1:], gif_filename=output_path + "receiv_p.gif" );
PF.gif_from_jpgs( RTM_jpg_list[1:], gif_filename=output_path + "RTM.gif" );


    '''
    images = []
    
    for img in filenames_list:
        if os.path.exists(img):
            images.append(Image.open(img))
        else:
            print(f"Warning: {img} not found. Skipping this file.")
    
    if images:
        images[0].save(gif_filename, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"GIF saved as {gif_filename}")
    else:
        print("No valid images found to create GIF.")
    
    

def past_image_and_colorbar(file1, file2, file3, ratio):
    
    image1 = Image.open(file1)
    image2 = Image.open(file2)
    
    w1, h1 = image1.size
    w2, h2 = image2.size
    
    length    = np.int(w1*ratio)
    total_lenth = w1 + length
    
    print("w1:{},h1:{}".format(w1, h1));print("w2:{},h2:{}".format(w2, h2))
    
    combine = Image.new('RGB', (total_lenth, max(h1,h2)), 'white')

    combine.paste( image2, (length,0) )

    # combine.paste( image1, (0,0) )
    
    combine.save(file3)
    




def plot_graph(input_array2,plot_number=1, 
               
               dz=1, x1beg=0, x1end=0, d1num=0, 
               
               x2beg=0, x2end=0, d2num=0, 
               
               label1="Times (ms)", label2="Relative amplitude", 
               
               xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", 
               
               figsize=(6, 3.5), 
               
               axis_width=1, axis_length=1, 
               
               linewidth=1, linestyle=("-", "-", "-", "-"), grid_linewidth=0, line_color=("k", "r", "b", "g"), 
               
               FontSize=(11, 11), fontsize=(11, 11), 
               
               legend_size=(8, 8), legend_position="best", legend_name=("","","","", ), 
               
               eps_dpi=300, output_name="tmp.eps", 
               
               title="", plot_end=0, plot2_end=0, 
               
               reverse_1=False, reverse_2=False, pltshow=False, 
               
               xscale_log=False, yscale_log=False, y_powerlimit=(-2, 2), 
               
               
               output_info=False ,
               log_file="plot_graph.txt"
               ):
    """
    input_array: input_arry, list of numpy array, list[0]=np.array
    plot_number: how many number of array is plotted in a figure. We can plot different length (but the same original position and the different end position)
    dz:
    x1beg:
    d1num:
    x2beg:
    d2num:
    
    legend_position: supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    
    pltshow: two options: True: plotshow and save figure, False: only save
    example1: P.plot_graph(in_arr, plot_number=plot_number, dz=dz, x1beg=f1, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, label1=label1, label2=label2, figsize=figsize1, axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=10, FontSize=10, legend_size=12, legend_position=legend_position, legend_name=legend_name, eps_dpi=300, output_name=ou_file);
    
    P.plot_graph(in_arr, plot_number=plot_number, dz=dz, x1beg=f1, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, label1=label1, label2=label2, figsize=figsize1, axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=10, FontSize=10, legend_size=12, legend_position=legend_position, legend_name=legend_name, eps_dpi=300, output_name=ou_file, plot_end=plot_end);
    """
    
    input_array = TF.list_to_numpy(input_array2);
    
    x_list=[]
    for i in range(0, plot_number):
        shape = input_array[i].shape
        nz    = shape[0]

        if     x1beg != 0 and x1end==0:
            x_array=np.linspace(x1beg, x1beg+nz*dz, nz)
        elif   x1beg != 0 and x1end!=0:
            x_array=np.linspace(x1beg, x1end, nz)
        elif   x1beg == 0 and x1end!=0:
            x_array=np.linspace(x1beg, x1end, nz)
        else:
            x_array=np.linspace(0, nz*dz, nz)
    
        x_list.append(x_array);
    
	# print_numpy_array_info(x_array, "x_array of plot_graph")
    if output_info:
        print("max of x_array is", np.max(x_array) ); print("min of x_array is", np.min(x_array) );
    
    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=(10,5))
    
    # Remove the margins of the subplot
    ax.margins(0)
    # Remove all the whitespace around the figure
    fig.set_constrained_layout(True)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.1, hspace=0.1) 
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95, wspace=0.1, hspace=0.1)

    
    if plot_number >= 1:
        plt.plot(x_list[0], input_array[0],  line_color[0]+linestyle[0] , linewidth=linewidth)
    if plot_number >= 2:
        plt.plot(x_list[1], input_array[1],  line_color[1]+linestyle[1] , linewidth=linewidth)
    if plot_number >= 3:
        plt.plot(x_list[2], input_array[2],  line_color[2]+linestyle[2] , linewidth=linewidth)
    if plot_number >= 4:
        plt.plot(x_list[3], input_array[3],  line_color[3]+linestyle[3] , linewidth=linewidth)
    if plot_number >= 5:
        plt.plot(x_list[4], input_array[4],  line_color[4]+linestyle[4] , linewidth=linewidth)


    plt.xlabel(label1, fontsize=FontSize[0], color="k")
    plt.ylabel(label2, fontsize=FontSize[1], color="k")
    
    
    if d1num!=0:
        plt.xlim(x1beg, x1end)
        num1 = int((x1end - x1beg) / d1num )
        xtick = np.linspace(x1beg,  x1beg + num1*d1num, num1+1)
        plt.xticks(xtick, fontsize=fontsize[0])
        
    if x2beg!=0 and x2end!=0 and d2num!=0:
        plt.ylim(x2beg, x2end)
        num2 = int((x2end - x2beg) / d2num ) 
        ytick = np.linspace(x2beg, x2beg + num2*d2num, num2+1)
        plt.yticks(ytick, fontsize=fontsize[1])
    
    if plot_end!=0:
        plt.xlim(x1beg, plot_end)
    if plot2_end!=0:
        plt.ylim(x2beg, plot2_end)
    

    y_tick_format = ticker.ScalarFormatter(useMathText=True)
    y_tick_format.set_powerlimits( y_powerlimit );
    y_tick_format.set_scientific(True)
    y_tick_format.set_useOffset(False)
    y_tick_format.format = '%.1E'
    ax.yaxis.set_major_formatter(y_tick_format)
    
    
    if xscale_log:
        plt.xscale('log') 
    if yscale_log:
        plt.yscale('log')  
    
    
    
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        plt.xticks(xtick_positions, xtick_lables)
        if output_info:
            print("xtick_positions is ", xtick_positions)
            print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        plt.yticks(ytick_positions, ytick_lables)
        if output_info:
            print("ytick_positions is ", ytick_positions)
            print("ytick_lables is ", ytick_lables)    
    
    
    
    
    
    # get current axis
    ax = plt.gca()
    # get current axis
    ax = plt.gca()
    xtick = ax.get_xticks()
    plt.xticks(fontsize=fontsize[0])
    ytick = ax.get_yticks()
    plt.yticks(fontsize=fontsize[1]) 
    
    # x y
    if axis_length!=0 and axis_width!=0:
        ax.tick_params(axis="both", direction="out", length=axis_length, width=axis_width, colors='k')
        ax.xaxis.set_ticks_position("bottom") #"top","bottom",
        ax.yaxis.set_ticks_position("left")  #"left","right"
        # ax.spines['right'].set_position(("data", 0.01))  #y轴在x轴的0.01

    #  #"top","bottom","left","right"
    if axis_width==0:
        ax.spines["top"].set_linewidth(axis_width)
        ax.spines["bottom"].set_linewidth(axis_width)
        ax.spines["left"].set_linewidth(axis_width)
        ax.spines["right"].set_linewidth(axis_width)
    
    if reverse_1:
        # Reverse the x-axis
        ax.invert_xaxis()
    if reverse_2:
        # Reverse the x-axis
        ax.invert_yaxis()    
    
    if title:
        plt.title(title);
        
    # grid parameters
    if grid_linewidth!=0:
        plt.grid(True, which='major', axis='both', color='k', linestyle='-',  linewidth=grid_linewidth)
        ax.xaxis.grid(color='k', linestyle='--', linewidth=grid_linewidth)
        ax.yaxis.grid(color='k', linestyle='--', linewidth=grid_linewidth)

    if legend_size[0] != 0:
        plt.legend(legend_name, loc=legend_position, fontsize=legend_size[0])
    
    # Save plot
    if output_name.lower().endswith('.eps'):
        plt.savefig(output_name, dpi=300, format='eps')
        jpg_output = output_name[:-4] + ".png"  # 去掉 .eps，换成 .png
        plt.savefig(jpg_output, dpi=300)
        write_txt(log_file, f"Saved as '{output_name}' and '{jpg_output}'")
    else:
        plt.savefig(output_name, dpi=300)
        write_txt(log_file, f"Saved as '{output_name}'")

    if not pltshow:
        plt.close()
    else:
        plt.show()
        

def plot_graph2(input_array, plot_number=1, dz=1, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, label1="x", label2="y", figsize=(10, 5), axis_width=1, axis_length=1, linewidth=1, linestyle=("-", "-", "-"), grid_linewidth=0, line_color=("k", "r", "b", ), FontSize=(9,9), fontsize=(9,9), legend_size=(9,9), legend_position="best", legend_name=("1","2","3", ), eps_dpi=300, output_name="tmp.eps", title="", plot_end=0, reverse_1=False, reverse_2=False, pltshow=False, output_info=False ):
    """
    input_array: input_arry, list of numpy array, list[0]=np.array
    plot_number: how many number of array is plotted in a figure. We can plot different length (but the same original position and the different end position)
    dz:
    x1beg:
    d1num:
    x2beg:
    d2num:
    
    legend_position: supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    
    pltshow: two options: True: plotshow and save figure, False: only save
    example1: P.plot_graph(in_arr, plot_number=plot_number, dz=dz, x1beg=f1, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, label1=label1, label2=label2, figsize=figsize1, axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=10, FontSize=10, legend_size=12, legend_position=legend_position, legend_name=legend_name, eps_dpi=300, output_name=ou_file);
    
    P.plot_graph(in_arr, plot_number=plot_number, dz=dz, x1beg=f1, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, label1=label1, label2=label2, figsize=figsize1, axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=10, FontSize=10, legend_size=12, legend_position=legend_position, legend_name=legend_name, eps_dpi=300, output_name=ou_file, plot_end=plot_end);
    """
    
    x_list=[]
    for i in range(0, plot_number):
        shape = input_array[i].shape
        nz    = shape[0]

        if     x1beg != 0 and x1end==0:
            x_array=np.linspace(x1beg, x1beg+nz*dz, nz)
        elif   x1beg != 0 and x1end!=0:
            x_array=np.linspace(x1beg, x1end, nz)
        elif   x1beg == 0 and x1end!=0:
            x_array=np.linspace(x1beg, x1end, nz)
        else:
            x_array=np.linspace(0, nz*dz, nz)
    
        x_list.append(x_array);
    
	# print_numpy_array_info(x_array, "x_array of plot_graph")
    print("max of x_array is", np.max(x_array) ); print("min of x_array is", np.min(x_array) );
    
    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=(10,5))
    
    # Remove the margins of the subplot
    ax.margins(0)
    # Remove all the whitespace around the figure
    fig.set_constrained_layout(True)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05, top=0.85, wspace=0.2, hspace=0.2)
    
    if plot_number >= 1:
        plt.semilogy(x_list[0], input_array[0],  line_color[0]+linestyle[0] , linewidth=linewidth)
    if plot_number >= 2:
        plt.semilogy(x_list[1], input_array[1],  line_color[1]+linestyle[1] , linewidth=linewidth)
    if plot_number >= 3:
        plt.semilogy(x_list[2], input_array[2],  line_color[2]+linestyle[2] , linewidth=linewidth)
    if plot_number >= 4:
        plt.semilogy(x_list[3], input_array[3],  line_color[3]+linestyle[3] , linewidth=linewidth)
    if plot_number >= 5:
        plt.semilogy(x_list[4], input_array[4],  line_color[4]+linestyle[4] , linewidth=linewidth)


    plt.xlabel(label1, fontsize=FontSize[0], color="k")
    plt.ylabel(label2, fontsize=FontSize[1], color="k")
    
    
    if d1num!=0:
        plt.xlim(x1beg, x1end)
        num1 = int((x1end - x1beg) / d1num )
        xtick = np.linspace(x1beg,  x1beg + num1*d1num, num1+1)
        plt.xticks(xtick, fontsize=fontsize[0])
        
    if x2beg!=0 and x2end!=0 and d2num!=0:
        plt.ylim(x2beg, x2end)
        num2 = int((x2end - x2beg) / d2num ) 
        ytick = np.linspace(x2beg, x2beg + num2*d2num, num2+1)
        plt.yticks(ytick, fontsize=fontsize[1])
    
    if plot_end!=0:
        plt.xlim(x1beg, plot_end)
    
    # get current axis
    ax = plt.gca()
    # get current axis
    ax = plt.gca()
    xtick = ax.get_xticks()
    plt.xticks(fontsize=fontsize[0])
    ytick = ax.get_yticks()
    plt.yticks(fontsize=fontsize[1]) 
    
    # x y
    if axis_length!=0 and axis_width!=0:
        ax.tick_params(axis="both", direction="out", length=axis_length, width=axis_width, colors='k')
        ax.xaxis.set_ticks_position("bottom") #"top","bottom",
        ax.yaxis.set_ticks_position("left")  #"left","right"
        # ax.spines['right'].set_position(("data", 0.01))  #y轴在x轴的0.01

    #  #"top","bottom","left","right"
    if axis_width==0:
        ax.spines["top"].set_linewidth(axis_width)
        ax.spines["bottom"].set_linewidth(axis_width)
        ax.spines["left"].set_linewidth(axis_width)
        ax.spines["right"].set_linewidth(axis_width)
    
    if reverse_1:
        # Reverse the x-axis
        ax.invert_xaxis()
    if reverse_2:
        # Reverse the x-axis
        ax.invert_yaxis()    
	
    if title:
        plt.title(title);
	
    # grid parameters
    if grid_linewidth!=0:
        plt.grid(True, which='major', axis='both', color='k', linestyle='-',  linewidth=grid_linewidth)
        ax.xaxis.grid(color='k', linestyle='--', linewidth=grid_linewidth)
        ax.yaxis.grid(color='k', linestyle='--', linewidth=grid_linewidth)

    if legend_size != 0:
        plt.legend(legend_name, loc=legend_position, fontsize=legend_size)
    
    if output_name.lower().endswith('.eps'):
        plt.savefig(output_name, dpi=300, format='eps')
        jpg_output = output_name[:-4] + ".png"  # 去掉 .eps，换成 .png
        plt.savefig(jpg_output, dpi=300)
        print(f"Saved as '{output_name}' and '{jpg_output}'")
    else:
        plt.savefig(output_name, dpi=300)
        print(f"Saved as '{output_name}'")

    
    if not pltshow:
        plt.close();        



def imshow3D(data_3d, 
             xlabel="Inline (km)", ylabel="Crossline (km)", zlabel="Depth (km)",
             x1beg=0, x1end=0, d1num=0, d1=0.01,
             x2beg=0, x2end=0, d2num=0, d2=0.01,
             x3beg=0, x3end=0, d3num=0, d3=0.01,
             tranpose_2D_list = [False, False, False],
             height_width_ratio=[0.7, 0.7, 0.7],
             
             slice_positions=None, slice_round_num=2,
             vmin=0, vmax=0,
             total_title="",
             title_size=(11, 11, 11, 14),
             FontSize=(12, 12), fontsize=(11, 11), legend_size=(11, 11, 11),
             
             figsize=(12, 8), 
             output_name="output_3d.eps",
             
             xtop=True, pltshow=False, 
             
             colorbar=True, colorbar_label="Relative amplitude", cmap="gray",
             
             colorbar_text=('left', 'bottom'),  colorbar_ticks="", colorbar_ticks_num=5, powerlimits=(-1,1),
             output_info=False,
             log_file="imshow3D.txt"
             ):
    """
    Display three 2D slices of a 3D array at default or specified positions.

    Parameters:
    - data_3d: 3D array [Z, Y, X] (numpy array or torch.Tensor or CuPy array)
    - xlabel, ylabel, zlabel: Labels for each axis
    - slice_positions: A tuple of 3 values indicating the slice positions in X, Y, and Z directions
    - title: A tuple containing titles for the three subplots
    - figsize: Size of the figure
    - output_name: Output filename for saving the plot
    - cmap: Color map for the plots
    - colorbar: Boolean to enable/disable colorbars
    - FontSize: Font size for axis labels
    - fontsize: Font size for tick labels
    - xtop: Boolean to place the x-axis on the top (for each plot)
    - pltshow: Boolean to show the plots or not
    """
    
    data_3d = WR.array_squeeze(data_3d);
    
    data_3d = TF.array_to_numpy(data_3d)

    # Handle numpy array or CuPy array
    if vmin == 0 and vmax == 0:
        vmin = np.min(data_3d)
        vmax = np.max(data_3d)

    nx, ny, nz = data_3d.shape

    # Calculate the range of each axis if not specified
    if x1beg == 0 and x1end == 0:
        x1end = nx * d1
    elif x1beg != 0 and x1end == 0:
        x1end = x1beg + nx * d1

    if x2beg == 0 and x2end == 0:
        x2end = ny * d2
    elif x2beg != 0 and x2end == 0:
        x2end = x2beg + ny * d2

    if x3beg == 0 and x3end == 0:
        x3end = nz * d3
    elif x3beg != 0 and x3end == 0:
        x3end = x3beg + nz * d3

    # Default slice positions to the center of each dimension if not provided
    if slice_positions is None:
        x_mid, y_mid, z_mid = ( int(nx // 2), int(ny // 2), int(nz // 2) )
    else:
        x_mid, y_mid, z_mid = slice_positions

    

    title=[0, 1, 2];
    if title_size:
        xlabel_unit = re.search(r'\(.*?\)', xlabel).group(0) if re.search(r'\(.*?\)', xlabel) else ""
        ylabel_unit = re.search(r'\(.*?\)', ylabel).group(0) if re.search(r'\(.*?\)', ylabel) else ""
        zlabel_unit = re.search(r'\(.*?\)', zlabel).group(0) if re.search(r'\(.*?\)', zlabel) else ""
        
        xlabel_remove = re.sub(r'\s*\(.*?\)', '', xlabel)
        ylabel_remove = re.sub(r'\s*\(.*?\)', '', ylabel)
        zlabel_remove = re.sub(r'\s*\(.*?\)', '', zlabel)
        
        title[0] = zlabel_remove + "=" + str( round(z_mid*d3 + x3beg, slice_round_num) ) + " " + zlabel_unit
        title[1] = ylabel_remove + "=" + str( round(y_mid*d2 + x2beg, slice_round_num) ) + " " + ylabel_unit
        title[2] = xlabel_remove + "=" + str( round(x_mid*d1 + x1beg, slice_round_num) ) + " " + xlabel_unit
        
    else:
        title[0] = ""
        title[1] = ""
        title[2] = ""
    
    ###default, I will tranpose
    data1 = data_3d[:, :, z_mid].T;
    data1_x_label=xlabel
    data1_y_label=ylabel
    extent1=(x1beg, x1end, x2end, x2beg)
    aspect1=abs((x1end - x1beg) / (x2end - x2beg)) * height_width_ratio[0]
    
    
    data2 = data_3d[:, y_mid, :].T;
    data2_x_label=xlabel
    data2_y_label=zlabel
    extent2=(x1beg, x1end, x3end, x3beg)
    aspect2=abs((x1end - x1beg) / (x3end - x3beg)) * height_width_ratio[1]
    
    
    data3 = data_3d[x_mid, :, :].T;
    data3_x_label=ylabel
    data3_y_label=zlabel
    extent3=(x2beg, x2end, x3end, x3beg)
    aspect3=abs((x2end - x2beg) / (x3end - x3beg))  * height_width_ratio[2]
    
    
    if tranpose_2D_list[0]:
        data1 = data1.T;
        data1_x_label, data1_y_label = data1_y_label, data1_x_label  # Swap labels
        extent1=(x2beg, x2end, x1end, x1beg)
        aspect1=abs((x2end - x2beg) / (x1end - x1beg)) * height_width_ratio[0]
        
    if tranpose_2D_list[1]:
        data2 = data2.T;
        data2_x_label, data2_y_label = data2_y_label, data2_x_label  # Swap labels
        extent2=(x3beg, x3end, x1end, x1beg)
        
        aspect2=abs((x3end - x3beg) / (x1end - x1beg)) * height_width_ratio[1]
        
    if tranpose_2D_list[2]:
        data3 = data3.T;
        data3_x_label, data3_y_label = data3_y_label, data3_x_label  # Swap labels
        extent3=(x3beg, x3end, x2end, x2beg)
        
        aspect3=abs((x3end - x3beg) / (x2end - x2beg))  * height_width_ratio[2]
    
    
     # Create a gridspec layout for 3 subplots (no colorbars)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.1, hspace=0.3)


    
    # XY Slice (Z方向中间位置的切片，放在左上方)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(data1, vmin=vmin, vmax=vmax, cmap=cmap, extent=extent1, origin='upper')
    ax1.set_xlabel(data1_x_label, fontsize=FontSize[0])
    ax1.set_ylabel(data1_y_label, fontsize=FontSize[1])
    ax1.set_aspect( aspect1 ) 
    # ax1.set_title(title[0])
    ax1.text(0.5, -0.05, title[0], ha='center', va='center', transform=ax1.transAxes, fontsize=title_size[0])
    
    if d1num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d1num))
    else:
        # ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.xaxis.set_major_locator( ticker.MaxNLocator(nbins=5) )
        ax1.yaxis.set_major_locator( ticker.MaxNLocator(nbins=5) )

    if xtop:
        ax1.xaxis.set_label_position("top")
        ax1.xaxis.tick_top()

    
    
    # XZ Slice (Y方向中间位置的切片，放在左下方)
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(data2, vmin=vmin, vmax=vmax, cmap=cmap, extent=extent2, origin='upper')
    ax2.set_xlabel(data2_x_label, fontsize=FontSize[0])
    ax2.set_ylabel(data2_y_label, fontsize=FontSize[1])
    ax2.set_aspect( aspect2 )  
    # ax2.set_title(title[1])
    ax2.text(0.5, -0.05, title[1], ha='center', va='center', transform=ax2.transAxes, fontsize=title_size[1])
    
    if d2num!=0:
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(d2num))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        # ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax2.xaxis.set_major_locator( ticker.MaxNLocator(nbins=5) )
        ax2.yaxis.set_major_locator( ticker.MaxNLocator(nbins=5) )
    
    if xtop:
        ax2.xaxis.set_label_position("top")
        ax2.xaxis.tick_top()


    # YZ Slice (X方向中间位置的切片，放在右下方)
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(data3, vmin=vmin, vmax=vmax, cmap=cmap, extent=extent3, origin='upper')
    ax3.set_xlabel(data3_x_label, fontsize=FontSize[0])
    ax3.set_ylabel(data3_y_label, fontsize=FontSize[1])
    ax3.set_aspect( aspect3 )
    # ax3.set_title(title[2])
    ax3.text(0.5, -0.05, title[2], ha='center', va='center', transform=ax3.transAxes, fontsize=title_size[2])
    
    if d2num!=0:
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(d3num))
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(d3num))
    else:
        # ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax3.xaxis.set_major_locator( ticker.MaxNLocator(nbins=5) )
        ax3.yaxis.set_major_locator( ticker.MaxNLocator(nbins=5) )

    if xtop:
        ax3.xaxis.set_label_position("top")
        ax3.xaxis.tick_top()
    
    
    
    
    
    if colorbar:  
        cbar = plt.colorbar(im1, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        ##label size 
        
        
        if output_info:
            print("len(colorbar_ticks) is {}", len(colorbar_ticks) )
        if len(colorbar_ticks)==0:
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=colorbar_ticks_num)) ## 5
        if len(colorbar_ticks)!=0:
            ticks=cbar.set_ticks(colorbar_ticks)
        
        cbar.ax.tick_params(labelsize=legend_size[1]) ##ticks size
        
        if output_info:
            print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        if np.fabs(vmin)>=0.09999999 and np.fabs(vmax)<9.9999999 :
            tick_format.set_powerlimits( (-2,2) )
        else:
            tick_format.set_powerlimits( powerlimits )
        
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1f'
        cbar.ax.yaxis.set_major_formatter(tick_format)
       
        if output_info:
            print("final cbar1.get_ticks() is", cbar.get_ticks())
        ticks=cbar.get_ticks()
        
        
        # cbar.set_ticks(ticks)
        # ticks=cbar.get_ticks()
        # for i in range(0, len(ticks) ):
        #     # if len( str(abs(ticks)[0]) ) <2:
        #     if ticks[i] <:
        #         tick_format = ticker.ScalarFormatter(useMathText=True)
        #         tick_format.set_powerlimits( -2, 2 )
        #         tick_format.set_scientific(True)
        #         tick_format.set_useOffset(False)
        #         tick_format.format = '%.1f'
        #         cbar.ax.yaxis.set_major_formatter(tick_format)
                
        
        if output_info:
            print("final cbar2.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        if output_info:
            print("cbar_box is", cbar_box)
        
        
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_horizontalalignment(colorbar_text[0]) #'center', 'right', 'left'
        offset_text.set_verticalalignment(colorbar_text[1]) #supported values are 'top', 'bottom', 'center', 'baseline', 'center_baseline'

        if output_info:
            print("offset_text.set_position 1 is", offset_text.set_position)
        
        
    # Add the overall title at the top of the figure
    if total_title:
        fig.suptitle(total_title, fontsize=title_size[3], y=0.97)

    # Adjust layout
    # fig.tight_layout(rect=[0, 0, 1, 0.95])  
    # Leave space for the suptitle    
    # Adjust layout
    fig.tight_layout()

    # Save plot
    if output_name.lower().endswith('.eps'):
        plt.savefig(output_name, dpi=300, format='eps')
        jpg_output = output_name[:-4] + ".png"  # 去掉 .eps，换成 .png
        plt.savefig(jpg_output, dpi=300)
        write_txt(log_file, f"Saved as '{output_name}' and '{jpg_output}'")
    else:
        plt.savefig(output_name, dpi=300)
        write_txt(log_file, f"Saved as '{output_name}'")

    if not pltshow:
        plt.close()
    else:
        plt.show()



def imshow(x_data, 
            
            x1beg=0, x1end=0, d1num=0, d1=0.01,
            
            x2beg=0, x2end=0, d2num=0, d2=0.01,
            
            xlabel="Distance (km)", ylabel="Depth (km)", 
            
            xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", 
            
            vmin=0,  vmax=0, 

            FontSize=(12, 12), fontsize=(11, 11), legend_size=(10, 10, 8), 
            figsize=(6, 3.5),  
            
            title="", output_name="tmp.eps", eps_dpi=200, 
            
            
            colorbar=True, colorbar_label="Relative amplitude", cmap="gray",
            
            colorbar_text=('left', 'bottom'),  colorbar_ticks="", colorbar_ticks_num=5, powerlimits=(-1,1), 
            
            
            axis_mark='tight', xtop=True, pltshow=False, gca_remove=False, output_info=False,
            
            log_file="imshow.txt"):
    """
    input array is [nz,nx]
    
    
    case 3:
    when I want to use xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", I recommend that I can use set d1=1 d2=1         
    """
    x_data = WR.array_squeeze(x_data);
    
    input_array = TF.array_to_numpy(x_data);
    
    if vmin==0 and vmax==0:
        vmin = 1.0 * input_array.min().item()
        vmax = 1.0 * input_array.max().item()
    
    ###
    if   x1beg ==0 and x1end ==0:
        x1end = input_array.shape[1] * d1
    elif x1beg !=0 and x1end ==0:
        x1end = x1beg + input_array.shape[1] * d1
        
        
    if  x2beg ==0 and x2end ==0:
        x2end = input_array.shape[0] * d2 
    elif x2beg !=0 and x2end ==0:
        x2end = x2beg + input_array.shape[0] * d2
    

    
    fig, ax1 = plt.subplots(figsize=figsize)


    im1   = ax1.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2end, x2beg), origin='upper');

    
    if d1num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
    else:
        # ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        
    
    if d2num!=0:
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        # ax1.yaxis.set_major_locator(ticker.AutoLocator())
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5)) 
    
    
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        plt.xticks(xtick_positions, xtick_lables)
        if output_info:
            print("xtick_positions is ", xtick_positions)
            print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        plt.yticks(ytick_positions, ytick_lables)
        if output_info:
            print("ytick_positions is ", ytick_positions)
            print("ytick_lables is ", ytick_lables)    
    
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    
    plt.axis(axis_mark);
    ax1.set_xlabel(xlabel, fontsize=FontSize[0])
    ax1.set_ylabel(ylabel, fontsize=FontSize[1])
    
    ax1.tick_params(axis='x', labelsize=fontsize[0])
    ax1.tick_params(axis='y', labelsize=fontsize[1])
    
    # Remove the margins of the subplot
    # fig.set_constrained_layout(True) ###?
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05, top=0.85, wspace=0.2, hspace=0.2)
    
    if colorbar:  
        cbar = plt.colorbar(im1, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        ##label size 
        
        
        if output_info:
            print("len(colorbar_ticks) is {}", len(colorbar_ticks) )
        if len(colorbar_ticks)==0:
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=colorbar_ticks_num)) ## 5
        if len(colorbar_ticks)!=0:
            ticks=cbar.set_ticks(colorbar_ticks)
        
        cbar.ax.tick_params(labelsize=legend_size[1]) ##ticks size
        
        if output_info:
            print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        if np.fabs(vmin)>=0.09999999 and np.fabs(vmax)<9.9999999 :
            tick_format.set_powerlimits( (-2,2) )
        else:
            tick_format.set_powerlimits( powerlimits )
        
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1f'
        cbar.ax.yaxis.set_major_formatter(tick_format)
       
        if output_info:
            print("final cbar1.get_ticks() is", cbar.get_ticks())
        ticks=cbar.get_ticks()
        
        
        # cbar.set_ticks(ticks)
        # ticks=cbar.get_ticks()
        # for i in range(0, len(ticks) ):
        #     # if len( str(abs(ticks)[0]) ) <2:
        #     if ticks[i] <:
        #         tick_format = ticker.ScalarFormatter(useMathText=True)
        #         tick_format.set_powerlimits( -2, 2 )
        #         tick_format.set_scientific(True)
        #         tick_format.set_useOffset(False)
        #         tick_format.format = '%.1f'
        #         cbar.ax.yaxis.set_major_formatter(tick_format)
                
        
        if output_info:
            print("final cbar2.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        if output_info:
            print("cbar_box is", cbar_box)
        
        
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_horizontalalignment(colorbar_text[0]) #'center', 'right', 'left'
        offset_text.set_verticalalignment(colorbar_text[1]) #supported values are 'top', 'bottom', 'center', 'baseline', 'center_baseline'

        if output_info:
            print("offset_text.set_position 1 is", offset_text.set_position)
        
    
    if gca_remove:
        plt.gca().remove()
    
    
    # Save plot
    if output_name.lower().endswith('.eps'):
        plt.savefig(output_name, dpi=300, format='eps')
        jpg_output = output_name[:-4] + ".png"  # 去掉 .eps，换成 .png
        plt.savefig(jpg_output, dpi=300)
        write_txt(log_file, f"Saved as '{output_name}' and '{jpg_output}'")
    else:
        plt.savefig(output_name, dpi=300)
        write_txt(log_file, f"Saved as '{output_name}'")

    if not pltshow:
        plt.close()
    else:
        plt.show()
        




def imshow_kxkz(x_data, 
            kx_max      = 20,
            kz_max      = 20,
            x1beg=0, x1end=0, d1num=0, d1=0.01,
            
            x2beg=0, x2end=0, d2num=0, d2=0.01,
            
            xlabel=r"Horizontal wavenumber (km$^{-1}$)", ylabel=r"vertical wavenumber (km$^{-1}$)", 
            
            xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", 
            
            vmin=0,  vmax=0, 

            FontSize=(12, 12), fontsize=(11, 11), legend_size=(10, 10, 8), 
            figsize=(6, 3.5),  
            
            title="", output_name="tmp.eps", eps_dpi=200, 
            
            
            colorbar=True, colorbar_label="Normalized amplitude", cmap="gray",
            
            colorbar_text=('left', 'bottom'),  colorbar_ticks="", colorbar_ticks_num=5, powerlimits=(-1,1), 
            
            
            axis_mark='tight', xtop=True, pltshow=False, gca_remove=False, output_info=False,
            
            log_file="imshow.txt"):
    """
    input array is [nz,nx]
    
    
    kx_max = 20
    kz_max = 20
    name_list=["kxkz-tru-ip",
                 "kxkz-mig-ip",
                 "kxkz-L2-ip",
                 "kxkz-TV-ip",]
    for idx, (arr, name) in enumerate( zip(plot_arr3_list, name_list)  ):
        
        eps_name = inv_path   + name + "-kxkz.png"
        PF.imshow_kxkz( arr.T, 
                  x1beg=-kx_max, x1end=kx_max, d1=hes_inv_obj.dx * 0.001,
                  
                  x2beg=-kz_max, x2end=kz_max, d2=hes_inv_obj.dx * 0.001,
                  xlabel=r"Horizontal wavenumber ($km^{-1}$)",
                  ylabel=r"vertical wavenumber ($km^{-1}$)",   
                  output_name=eps_name, 
                  vmin=0, 
                  vmax=1.0, 
                  cmap='seismic', 
                  colorbar_label='Relative impedance perturbation');
    """
    x_data = WR.array_squeeze(x_data);
    
    input_array = TF.array_to_numpy(x_data);
    
    
    [nz, nx]    = list( input_array.shape )
    
    dkx         = 1.0 /(d1 * nx) 
    dkz         = 1.0 /(d2 * nz) 
    kx_max_id   = kx_max//dkx
    kz_max_id   = kz_max//dkz
    
    arr_fft     = np.abs(   np.fft.fftn(input_array)   )
    
    arr_fft     =  arr_fft/ np.max(  arr_fft  );
    arr_fft     =  np.fft.fftshift(arr_fft)
    slice_index = slice(int(nz//2-kz_max_id), int(nz//2+kz_max_id)), slice(int(nx//2-kx_max_id), int(nx//2+kx_max_id))
        
    input_array    =  arr_fft[slice_index]
    
    
    if vmin==0 and vmax==0:
        vmin = 1.0 * input_array.min().item()
        vmax = 1.0 * input_array.max().item()
    
    ###
    if   x1beg ==0 and x1end ==0:
        x1end = input_array.shape[1] * d1
    elif x1beg !=0 and x1end ==0:
        x1end = x1beg + input_array.shape[1] * d1
        
        
    if  x2beg ==0 and x2end ==0:
        x2end = input_array.shape[0] * d2 
    elif x2beg !=0 and x2end ==0:
        x2end = x2beg + input_array.shape[0] * d2
    

    
    fig, ax1 = plt.subplots(figsize=figsize)


    im1   = ax1.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2end, x2beg), origin='upper');

    
    if d1num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
    else:
        # ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        
    
    if d2num!=0:
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        # ax1.yaxis.set_major_locator(ticker.AutoLocator())
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5)) 
    
    
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        plt.xticks(xtick_positions, xtick_lables)
        if output_info:
            print("xtick_positions is ", xtick_positions)
            print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        plt.yticks(ytick_positions, ytick_lables)
        if output_info:
            print("ytick_positions is ", ytick_positions)
            print("ytick_lables is ", ytick_lables)    
    
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    
    plt.axis(axis_mark);
    ax1.set_xlabel(xlabel, fontsize=FontSize[0])
    ax1.set_ylabel(ylabel, fontsize=FontSize[1])
    
    ax1.tick_params(axis='x', labelsize=fontsize[0])
    ax1.tick_params(axis='y', labelsize=fontsize[1])
    
    # Remove the margins of the subplot
    # fig.set_constrained_layout(True) ###?
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05, top=0.85, wspace=0.2, hspace=0.2)
    
    if colorbar:  
        cbar = plt.colorbar(im1, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        ##label size 
        
        
        if output_info:
            print("len(colorbar_ticks) is {}", len(colorbar_ticks) )
        if len(colorbar_ticks)==0:
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=colorbar_ticks_num)) ## 5
        if len(colorbar_ticks)!=0:
            ticks=cbar.set_ticks(colorbar_ticks)
        
        cbar.ax.tick_params(labelsize=legend_size[1]) ##ticks size
        
        if output_info:
            print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        if np.fabs(vmin)>=0.09999999 and np.fabs(vmax)<9.9999999 :
            tick_format.set_powerlimits( (-2,2) )
        else:
            tick_format.set_powerlimits( powerlimits )
        
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1f'
        cbar.ax.yaxis.set_major_formatter(tick_format)
       
        if output_info:
            print("final cbar1.get_ticks() is", cbar.get_ticks())
        ticks=cbar.get_ticks()
        
        
        # cbar.set_ticks(ticks)
        # ticks=cbar.get_ticks()
        # for i in range(0, len(ticks) ):
        #     # if len( str(abs(ticks)[0]) ) <2:
        #     if ticks[i] <:
        #         tick_format = ticker.ScalarFormatter(useMathText=True)
        #         tick_format.set_powerlimits( -2, 2 )
        #         tick_format.set_scientific(True)
        #         tick_format.set_useOffset(False)
        #         tick_format.format = '%.1f'
        #         cbar.ax.yaxis.set_major_formatter(tick_format)
                
        
        if output_info:
            print("final cbar2.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        if output_info:
            print("cbar_box is", cbar_box)
        
        
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_horizontalalignment(colorbar_text[0]) #'center', 'right', 'left'
        offset_text.set_verticalalignment(colorbar_text[1]) #supported values are 'top', 'bottom', 'center', 'baseline', 'center_baseline'

        if output_info:
            print("offset_text.set_position 1 is", offset_text.set_position)
        
    
    if gca_remove:
        plt.gca().remove()
    
    
    # Save plot
    if output_name.lower().endswith('.eps'):
        plt.savefig(output_name, dpi=300, format='eps')
        jpg_output = output_name[:-4] + ".png"  # 去掉 .eps，换成 .png
        plt.savefig(jpg_output, dpi=300)
        write_txt(log_file, f"Saved as '{output_name}' and '{jpg_output}'")
    else:
        plt.savefig(output_name, dpi=300)
        write_txt(log_file, f"Saved as '{output_name}'")

    if not pltshow:
        plt.close()
    else:
        plt.show()
        



def imshow1(x_data, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="Distance (km)", ylabel="Depth (km)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(12,12,10), xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", axis_mark='tight', gca_remove=False, colorbar_text=('left', 'bottom'), colorbar_ticks="", colorbar_ticks_num=5, powerlimits=(-1,1), output_info=False ):
    """
    input_array:input_arry, numpy array
    
    plot [nz, nx]
    
    pltshow: two options: True: plotshow and save figure, False: only save
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr, x1beg=fx, x1end=fx+nx*dz, d1num=0, x2beg=fz+nz*dz, x2end=fz, d2num=0, xlabel=label1, ylabel=label2, vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    input_array = TF.array_to_numpy(x_data);
    
    if vmin==0 and vmax==0:
        vmin = 0.8*input_array.min()
        vmax = 0.8*input_array.max()
        
        
    if x1beg ==0 and x1end ==0:
        x1end = input_array.shape[1]
        
    if  x2beg ==0 and x2end ==0:
        x2beg = input_array.shape[0]
    
    if figsize:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        fig, ax1 = plt.subplots(figsize=(10,5))


    im1   = ax1.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2beg, x2end), origin='upper');

    
    if d1num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
    else:
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
    if d2num!=0:
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        ax1.yaxis.set_major_locator(ticker.AutoLocator())
    
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        plt.xticks(xtick_positions, xtick_lables)
        if output_info:
            print("xtick_positions is ", xtick_positions)
            print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        plt.yticks(ytick_positions, ytick_lables)
        if output_info:
            print("ytick_positions is ", ytick_positions)
            print("ytick_lables is ", ytick_lables)    
    
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    plt.axis(axis_mark);
    ax1.set_xlabel(xlabel, fontsize=FontSize[0])
    ax1.set_ylabel(ylabel, fontsize=FontSize[1])
    
    ax1.tick_params(axis='x', labelsize=fontsize[0])
    ax1.tick_params(axis='y', labelsize=fontsize[1])
    
    # Remove the margins of the subplot
    # fig.set_constrained_layout(True) ###?
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05, top=0.85, wspace=0.2, hspace=0.2)  #检查边距设置：如果使用 subplots_adjust，确保边距设置合理（如 left=0.05, right=0.95），避免图像内容被边界遮挡。
    
    if colorbar:  
        cbar = plt.colorbar(im1, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) ##label size 
        
        if output_info:
            print("len(colorbar_ticks) is {}", len(colorbar_ticks) )
        if len(colorbar_ticks)==0:
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=colorbar_ticks_num)) ## 5
        if len(colorbar_ticks)!=0:
            ticks=cbar.set_ticks(colorbar_ticks)
        
        cbar.ax.tick_params(labelsize=legend_size[1]) ##ticks size
        
        if output_info:
            print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        if np.fabs(vmin)>=0.09999999 and np.fabs(vmax)<9.9999999 :
            tick_format.set_powerlimits( (-2,2) )
        else:
            tick_format.set_powerlimits( powerlimits )
        
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1f'
        cbar.ax.yaxis.set_major_formatter(tick_format)
       
        if output_info:
            print("final cbar1.get_ticks() is", cbar.get_ticks())
        ticks=cbar.get_ticks()
        
        
        # cbar.set_ticks(ticks)
        # ticks=cbar.get_ticks()
        # for i in range(0, len(ticks) ):
        #     # if len( str(abs(ticks)[0]) ) <2:
        #     if ticks[i] <:
        #         tick_format = ticker.ScalarFormatter(useMathText=True)
        #         tick_format.set_powerlimits( -2, 2 )
        #         tick_format.set_scientific(True)
        #         tick_format.set_useOffset(False)
        #         tick_format.format = '%.1f'
        #         cbar.ax.yaxis.set_major_formatter(tick_format)
                
        
        if output_info:
            print("final cbar2.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        if output_info:
            print("cbar_box is", cbar_box)
        
        
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_horizontalalignment(colorbar_text[0]) #'center', 'right', 'left'
        offset_text.set_verticalalignment(colorbar_text[1]) #supported values are 'top', 'bottom', 'center', 'baseline', 'center_baseline'

        if output_info:
            print("offset_text.set_position 1 is", offset_text.set_position)
        
    
    if gca_remove:
        plt.gca().remove()
    
    
    if output_name.lower().endswith('.eps'):
        plt.savefig(output_name, dpi=300, format='eps')
        jpg_output = output_name[:-4] + ".png"  # 去掉 .eps，换成 .png
        plt.savefig(jpg_output, dpi=300)
        print(f"Saved as '{output_name}' and '{jpg_output}'")
    else:
        plt.savefig(output_name, dpi=300)
        print(f"Saved as '{output_name}'")
 
    if not pltshow:
        plt.close();
        
        
def imshow2(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(12,12,10), xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", axis_mark='tight', gca_remove=False, colorbar_text=('left', 'bottom') ):
    """
    input_array:input_arry, numpy array
   
    pltshow: two options: True: plotshow and save figure, False: only save
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr, x1beg=fx, x1end=fx+nx*dz, d1num=0, x2beg=fz+nz*dz, x2end=fz, d2num=0, xlabel=label1, ylabel=label2, vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    if vmin==0 and vmax==0:
        vmin = input_array.min()
        vmax = input_array.max()
    if x1beg ==0 and x1end ==0:
        x1end = input_array.shape[1]
        
    if  x2beg ==0 and x2end ==0:
        x2beg = input_array.shape[0]
    
    if figsize:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        fig, ax1 = plt.subplots(figsize=(10,5))


    im1   = ax1.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2beg, x2end), origin='upper');

    
    if d1num!=0 and d2num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.yaxis.set_major_locator(ticker.AutoLocator())
    
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.xticks(xtick_positions, xtick_lables)
        print("xtick_positions is ", xtick_positions)
        print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.yticks(ytick_positions, ytick_lables)
        print("ytick_positions is ", ytick_positions)
        print("ytick_lables is ", ytick_lables)    
    
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    plt.axis(axis_mark);
    ax1.set_xlabel(xlabel, fontsize=FontSize[0])
    ax1.set_ylabel(ylabel, fontsize=FontSize[1])
    
    ax1.tick_params(axis='x', labelsize=fontsize[0])
    ax1.tick_params(axis='y', labelsize=fontsize[1])
    
    # Remove the margins of the subplot
    fig.set_constrained_layout(True)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05, top=0.85, wspace=0.2, hspace=0.2) 
    
    if colorbar:  
        cbar = plt.colorbar(im1, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar.ax.tick_params(labelsize=legend_size[1])
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        tick_format.set_powerlimits( (-2, 2) )
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1E'
        cbar.ax.yaxis.set_major_formatter(tick_format)
        
        # Set the number of ticks on the colorbar to 5
        # num_ticks = 5
        # ticks = ticker.LinearLocator(num_ticks).tick_values(vmin, vmax)
        # cbar.set_ticks(ticks)
        
        print("final cbar.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        
        offset_text.set
        
        offset_text.set_horizontalalignment(colorbar_text[0]) #'center', 'right', 'left'
        offset_text.set_verticalalignment(colorbar_text[1]) #supported values are 'top', 'bottom', 'center', 'baseline', 'center_baseline'

        print("offset_text.set_position 1 is", offset_text.set_position)
        
        # (x0, y0) = offset_text.get_position
        # offset_text.set_position( ( offset_text.x0 +0.2,  offset_text.y0 +0.2) )
        # print("offset_text.set_position 2 is", offset_text.set_position)
        
        # offset_text.set_position( ( (cbar_box.x0+cbar_box.x1)/2, cbar_box.ymax + 0.5) )
        # print("offset_text.set_position 2 is", offset_text.set_position)
        
        # canvas = fig.canvas
        # # create Agg
        # width, height = canvas.get_width_height()
        # dpi = canvas.get_renderer().dpi
        # renderer = agg.RendererAgg( width, height, dpi )
        
        # # get tex box
        # offset_text_bb = offset_text.get_window_extent(renderer=renderer)
        # offset_text_width  = offset_text_bb.width
        # offset_text_height = offset_text_bb.height
        
        # x = cbar_box.x0 + cbar_box.width + offset_text_width
        # y = cbar_box.y1 + offset_text_bb.height
        # offset_text.set_position((x, y))
        # print("offset_text_height is", offset_text_height)
        # print("offset_text_width is", offset_text_width)
        # print("offset_text.set_position is", offset_text.set_position)
    
    if gca_remove:
        plt.gca().remove()
    # plt.tight_layout()
    # fig.set_constrained_layout(True)
    
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".png", dpi=eps_dpi)
    else:
        plt.savefig(output_name, dpi=300)
 
    if not pltshow:
        plt.close();
  

def imshow222(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(12,12,10), xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", axis_mark='tight'):
    """
    input_array:input_arry, numpy array
   
    pltshow: two options: True: plotshow and save figure, False: only save
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr, x1beg=fx, x1end=fx+nx*dz, d1num=0, x2beg=fz+nz*dz, x2end=fz, d2num=0, xlabel=label1, ylabel=label2, vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    if vmin==0 and vmax==0:
        vmin = input_array.min()
        vmax = input_array.max()
    if x1beg ==0 and x1end ==0:
        x1end = input_array.shape[1]
        
    if  x2beg ==0 and x2end ==0:
        x2beg = input_array.shape[0]
    
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(10,5))

    

    ax1   = plt.subplot(121)
    # ax1 =fig.add_axes([0.1, 0.1, 0.4, 0.8])
    
    im1   = ax1.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2beg, x2end), origin='upper');

    
    if d1num!=0 and d2num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.yaxis.set_major_locator(ticker.AutoLocator())
    
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.xticks(xtick_positions, xtick_lables)
        print("xtick_positions is ", xtick_positions)
        print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.yticks(ytick_positions, ytick_lables)
        print("ytick_positions is ", ytick_positions)
        print("ytick_lables is ", ytick_lables)    
    
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    plt.axis(axis_mark);
    ax1.set_xlabel(xlabel, fontsize=FontSize[0])
    ax1.set_ylabel(ylabel, fontsize=FontSize[1])
    
    ax1.tick_params(axis='x', labelsize=fontsize[0])
    ax1.tick_params(axis='y', labelsize=fontsize[1])
    
    # Remove the margins of the subplot
    fig.set_constrained_layout(True)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05, top=0.85, wspace=0.2, hspace=0.2) 
    
    if colorbar:  
        # cbar_ax = plt.subplot(122)
        cbar_ax =fig.add_axes([0.6, 0.1, 0.6, 0.8])
        cbar = plt.colorbar(im1, cax=cbar_ax, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar.ax.tick_params(labelsize=legend_size[1])
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        tick_format.set_powerlimits( (-2, 2) )
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1E'
        cbar.ax.yaxis.set_major_formatter(tick_format)
        
        # Set the number of ticks on the colorbar to 5
        # num_ticks = 5
        # ticks = ticker.LinearLocator(num_ticks).tick_values(vmin, vmax)
        # cbar.set_ticks(ticks)
        
        print("final cbar.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_verticalalignment('center')
        offset_text.set_horizontalalignment('center')
        print("offset_text.set_position is", offset_text.set_position)
        # canvas = fig.canvas
        # # create Agg
        # width, height = canvas.get_width_height()
        # dpi = canvas.get_renderer().dpi
        # renderer = agg.RendererAgg( width, height, dpi )
        
        # # get tex box
        # offset_text_bb = offset_text.get_window_extent(renderer=renderer)
        # offset_text_width  = offset_text_bb.width
        # offset_text_height = offset_text_bb.height
        
        # x = cbar_box.x0 + cbar_box.width + offset_text_width
        # y = cbar_box.y1 + offset_text_bb.height
        # offset_text.set_position((x, y))
        # print("offset_text_height is", offset_text_height)
        # print("offset_text_width is", offset_text_width)
        # print("offset_text.set_position is", offset_text.set_position)
    
    # plt.gca().remove()
    # plt.tight_layout()
    # fig.set_constrained_layout(True)
    # plt.tight_layout()
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".png", dpi=eps_dpi)
    else:
        plt.savefig(output_name, dpi=300)
 
    if not pltshow:
        plt.close();



def imshow44(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(12,12,10), xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", axis_mark='tight', GridSpec_ratio=(5, 2)):
    """
    input_array:input_arry, numpy array
   
    pltshow: two options: True: plotshow and save figure, False: only save
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr, x1beg=fx, x1end=fx+nx*dz, d1num=0, x2beg=fz+nz*dz, x2end=fz, d2num=0, xlabel=label1, ylabel=label2, vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    if vmin==0 and vmax==0:
        vmin = input_array.min()
        vmax = input_array.max()
    if x1beg ==0 and x1end ==0:
        x1end = input_array.shape[1]
        
    if  x2beg ==0 and x2end ==0:
        x2beg = input_array.shape[0]
    
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(10,5))

    
    # gs = GridSpec(1,2, figure=fig)
    

    
    gs = GridSpec(1,1, figure=fig)
    ax_main    = plt.subplot(gs[0])
    # ax_cbar    = plt.subplot(gs[1])
    
    
    # Plot the 2D array in the left subplot
    ax1   = ax_main.axes

    
    im1   = ax_main.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2beg, x2end), origin='upper');
    
    if d1num!=0 and d2num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.yaxis.set_major_locator(ticker.AutoLocator())
    
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.xticks(xtick_positions, xtick_lables)
        print("xtick_positions is ", xtick_positions)
        print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.yticks(ytick_positions, ytick_lables)
        print("ytick_positions is ", ytick_positions)
        print("ytick_lables is ", ytick_lables)    
    
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    plt.axis(axis_mark);
    ax1.set_xlabel(xlabel, fontsize=FontSize[0])
    ax1.set_ylabel(ylabel, fontsize=FontSize[1])
    
    ax1.tick_params(axis='x', labelsize=fontsize[0])
    ax1.tick_params(axis='y', labelsize=fontsize[1])
    
    # Remove the margins of the subplot
    fig.set_constrained_layout(True)
    # fig.subplots_adjust(left=-0.05, right=0.95, bottom=-0.05, top=0.95, wspace=0.2, hspace=0.2)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05, top=0.85, wspace=0.2, hspace=0.2) 
    
    if colorbar:  
        # cbar = plt.colorbar(im1, cax=ax_cbar, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar = plt.colorbar(im1, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar.ax.tick_params(labelsize=legend_size[1])
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        tick_format.set_powerlimits( (-2, 2) )
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1E'
        cbar.ax.yaxis.set_major_formatter(tick_format)
        
        # Set the number of ticks on the colorbar to 5
        # num_ticks = 5
        # ticks = ticker.LinearLocator(num_ticks).tick_values(vmin, vmax)
        # cbar.set_ticks(ticks)
        
        print("final cbar.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_verticalalignment('center')
        offset_text.set_horizontalalignment('center')
        print("offset_text.set_position is", offset_text.set_position)
        canvas = fig.canvas
        # create Agg
        width, height = canvas.get_width_height()
        dpi = canvas.get_renderer().dpi
        renderer = agg.RendererAgg( width, height, dpi )
        
        # get tex box
        offset_text_bb = offset_text.get_window_extent(renderer=renderer)
        offset_text_width  = offset_text_bb.width
        offset_text_height = offset_text_bb.height
        
        x = cbar_box.x0 + cbar_box.width + offset_text_width
        y = cbar_box.y1 + offset_text_bb.height
        offset_text.set_position((x, y))
        print("offset_text_height is", offset_text_height)
        print("offset_text_width is", offset_text_width)
        print("offset_text.set_position is", offset_text.set_position)
    
    # plt.gca().remove()
    # plt.tight_layout()
    # fig.set_constrained_layout(True)
    
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".png", dpi=eps_dpi)
    else:
        plt.savefig(output_name, dpi=300)
 
    if not pltshow:
        plt.close();    
 
    
 
    
 
def imshow3(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(12,12,10), xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", axis_mark='tight', GridSpec_ratio=(9, 1) ):
    """
    input_array:input_arry, numpy array
   
    pltshow: two options: True: plotshow and save figure, False: only save
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr, x1beg=fx, x1end=fx+nx*dz, d1num=0, x2beg=fz+nz*dz, x2end=fz, d2num=0, xlabel=label1, ylabel=label2, vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    
    
    fig = plt.figure(figsize=(10,5))
    gs = GridSpec(1, 2, width_ratios=[GridSpec_ratio[0], GridSpec_ratio[1]])
    
    ax_main    = plt.subplot(gs[0])
    ax_cbar    = plt.subplot(gs[1])
    
    # Plot the 2D array in the left subplot
    
    im1   = ax_main.imshow(input_array);
    ax1   = ax_main.axes
    
        
    if colorbar:  
        cbar = plt.colorbar(im1, cax=ax_cbar, label=colorbar_label, pad=0.01, shrink=0.7)
        cbar.ax.tick_params(labelsize=legend_size[1])
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        tick_format.set_powerlimits( (-2, 2) )
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1E'
        cbar.ax.yaxis.set_major_formatter(tick_format)
        
        # Set the number of ticks on the colorbar to 5
        # num_ticks = 5
        # ticks = ticker.LinearLocator(num_ticks).tick_values(vmin, vmax)
        # cbar.set_ticks(ticks)
        
        print("final cbar.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_verticalalignment('center')
        offset_text.set_horizontalalignment('center')
        print("offset_text.set_position is", offset_text.set_position)
        # canvas = fig.canvas
        # # create Agg
        # width, height = canvas.get_width_height()
        # dpi = canvas.get_renderer().dpi
        # renderer = agg.RendererAgg( width, height, dpi )
        
        # # get tex box
        # offset_text_bb = offset_text.get_window_extent(renderer=renderer)
        # offset_text_width  = offset_text_bb.width
        # offset_text_height = offset_text_bb.height
        
        # x = cbar_box.x0 + cbar_box.width + offset_text_width
        # y = cbar_box.y1 + offset_text_bb.height
        # offset_text.set_position((x, y))
        # print("offset_text_height is", offset_text_height)
        # print("offset_text_width is", offset_text_width)
        # print("offset_text.set_position is", offset_text.set_position)
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".png", dpi=eps_dpi)
    else:
        plt.savefig(output_name, dpi=300)
 
    if not pltshow:
        plt.close();
        
# def imshow2(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables=""):
def imshow2_old2(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(14,14,10), xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables=""):
    """
    input_array:input_arry, numpy array
   
    pltshow: two options: True: plotshow and save figure, False: only save
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr, x1beg=fx, x1end=fx+nx*dz, d1num=0, x2beg=fz+nz*dz, x2end=fz, d2num=0, xlabel=label1, ylabel=label2, vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    if vmin==0 and vmax==0:
        vmin = input_array.min()
        vmax = input_array.max()
    if x1beg ==0 and x1end ==0:
        x1end = input_array.shape[1]
        
    if  x2beg ==0 and x2end ==0:
        x2beg = input_array.shape[0]
    
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(10,5))

    
    # Plot the 2D array in the left subplot
    ax1   = fig.add_subplot( )
    
    im1   = plt.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2beg, x2end), origin='upper');
    
    if d1num!=0 and d2num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.yaxis.set_major_locator(ticker.AutoLocator())
        
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.xticks(xtick_positions, xtick_lables)
        print("xtick_positions is ", xtick_positions)
        print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.yticks(ytick_positions, ytick_lables)
        print("ytick_positions is ", ytick_positions)
        print("ytick_lables is ", ytick_lables)
        
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    plt.axis('tight');
    ax1.set_xlabel(xlabel, fontsize=FontSize[0])
    ax1.set_ylabel(ylabel, fontsize=FontSize[1])
    
    ax1.tick_params(axis='x', labelsize=fontsize[0])
    ax1.tick_params(axis='y', labelsize=fontsize[1])
    
    # Remove the margins of the subplot
    fig.set_constrained_layout(True)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05, top=0.85, wspace=0.2, hspace=0.2) 
    
    if colorbar:  
        cbar = plt.colorbar(label=colorbar_label, pad=0.01, shrink=0.7)
        cbar.ax.tick_params(labelsize=legend_size[1])
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        tick_format.set_powerlimits( (-2, 2) )
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1E'
        cbar.ax.yaxis.set_major_formatter(tick_format)
        
        # Set the number of ticks on the colorbar to 5
        # num_ticks = 5
        # ticks = ticker.LinearLocator(num_ticks).tick_values(vmin, vmax)
        # cbar.set_ticks(ticks)
        
        print("final cbar.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_verticalalignment('center')
        offset_text.set_horizontalalignment('center')
        print("offset_text.set_position is", offset_text.set_position)
        # canvas = fig.canvas
        # # create Agg
        # width, height = canvas.get_width_height()
        # dpi = canvas.get_renderer().dpi
        # renderer = agg.RendererAgg( width, height, dpi )
        
        # # get tex box
        # offset_text_bb = offset_text.get_window_extent(renderer=renderer)
        # offset_text_width  = offset_text_bb.width
        # offset_text_height = offset_text_bb.height
        
        # x = cbar_box.x0 + cbar_box.width + offset_text_width
        # y = cbar_box.y1 + offset_text_bb.height
        # offset_text.set_position((x, y))
        # print("offset_text_height is", offset_text_height)
        # print("offset_text_width is", offset_text_width)
        # print("offset_text.set_position is", offset_text.set_position)
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".png", dpi=eps_dpi)
    else:
        plt.savefig(output_name, dpi=300)
 
    if not pltshow:
        plt.close();
        

def format_tick(tick):
    if tick>0:
        exponent = int(np.floor(np.log10(float(tick))))
    elif tick<0:
        exponent =  int(np.floor(np.log10(float(-tick))))
    else:
        exponent = 0;
    coeff   = round(tick / 10**exponent, 2)
    
    if exponent == 0:
        return f"{coeff:g}"
    else:
        return f"{coeff:g} $\\times$ 10$^{{{exponent:d}}}$"





def imshow2_old(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(10,5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False):
    """
    input_array:input_arry, numpy array
    pltshow: two options: True: plotshow and save figure, False: only save    
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    if vmin==0 and vmax==0:
        vmin = input_array.min()
        vmax = input_array.max()
    if x1beg ==0 and x1end ==0 and x2beg ==0 and x2end ==0:
        x1end = input_array.shape[1]
        x2beg = input_array.shape[0]
    
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(10,5))

    
    # Plot the 2D array in the left subplot
    ax1   = fig.add_subplot( )
    
    im1   = plt.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2beg, x2end), origin='upper');
    
    if d1num!=0 and d2num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.yaxis.set_major_locator(ticker.AutoLocator())
        
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    plt.axis('tight');
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    
    # Remove the margins of the subplot
    fig.set_constrained_layout(True)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05, top=0.85, wspace=0.2, hspace=0.2) 
    
    if colorbar:
        cbar = plt.colorbar(label=colorbar_label, pad=0.01, shrink=0.7)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_ylabel(colorbar_label, fontsize=8) 
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        tick_format.set_powerlimits( (-1, 1) )
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.2E'
        cbar.ax.yaxis.set_major_formatter(tick_format)
        
        
        # Set the number of ticks on the colorbar to 5
        # num_ticks = 5
        # ticks = ticker.LinearLocator(num_ticks).tick_values(vmin, vmax)
        # cbar.set_ticks(ticks)
        
        print("final cbar.get_ticks() is", cbar.get_ticks())
        
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(6)
        offset_text.set_verticalalignment('center')
        offset_text.set_horizontalalignment('center')
        print("offset_text.set_position is", offset_text.set_position)
        
        # canvas = fig.canvas
        # # create Agg
        # width, height = canvas.get_width_height()
        # dpi = canvas.get_renderer().dpi
        # renderer = agg.RendererAgg( width, height, dpi )
        
        # # get tex box
        # offset_text_bb = offset_text.get_window_extent(renderer=renderer)
        # offset_text_width  = offset_text_bb.width
        # offset_text_height = offset_text_bb.height
        
        # x = cbar_box.x0 + cbar_box.width + offset_text_width
        # y = cbar_box.y1 + offset_text_bb.height
        # offset_text.set_position((x, y))
        # print("offset_text_height is", offset_text_height)
        # print("offset_text_width is", offset_text_width)
        # print("offset_text.set_position is", offset_text.set_position)
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".png", dpi=eps_dpi)
        plt.close();
    else:
        plt.savefig(output_name, dpi=300)
