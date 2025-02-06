import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import multiprocessing
########################################################################
def plotResults(Lx,Ly,psi,psi_exact,metrics,index,special):
    Ny = psi.shape[1]

    levels = np.linspace(-2.5, 2.5, 10)
    plt.set_cmap('bwr')
    # Create a figure
    fig = plt.figure(figsize=(12, 6))
    day = int(index/5)
    fig.suptitle(f'Day {day+1}', fontsize=16)
    if special==1:
        fig.patch.set_facecolor('xkcd:mint green')
    else:
        fig.patch.set_facecolor('xkcd:white')

    # Create a main GridSpec layout with 1 row and 3 columns
    #main_gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 0.05,0.07,1], wspace=0.15)
    main_gs = gridspec.GridSpec(2, 3, wspace=0.15)

    x = np.linspace(0,Lx,psi.shape[0])
    y = np.linspace(0,Ly,psi.shape[1])
    X,Y = np.meshgrid(x,y)
    
    # Plot in the first cell (1st column)
    ax1 = fig.add_subplot(main_gs[0, 0])
    ax1.set_title('Psi_ens 1')
    contour1 = plt.contourf(X,Y,np.transpose(psi[:,:,0]),levels=levels)    

    # Plot in the second cell (2nd column)
    ax4 = fig.add_subplot(main_gs[0, 1])
    ax4.set_title('Psi_exc 1')
    contour2 = plt.contourf(X,Y,np.transpose(psi_exact[:,:,0]),levels=levels)

    # Plot in the first cell (1st column)
    ax2 = fig.add_subplot(main_gs[1, 0])
    ax2.set_title('Psi_ens 2')
    contour1 = plt.contourf(X,Y,np.transpose(psi[:,:,1]),levels=levels)    

    # Plot in the second cell (2nd column)
    ax5 = fig.add_subplot(main_gs[1, 1])
    ax5.set_title('Psi_exc 2')
    contour2 = plt.contourf(X,Y,np.transpose(psi_exact[:,:,1]),levels=levels)

    days = np.linspace(0.25,(index+1)/5,index+1)
    # Plot in the second cell (2nd column)
    ax3 = fig.add_subplot(main_gs[0, 2])    
    ax3.set_title('Ek')
    #ax3.set_xlim(Ly/2-20,Ly/2+20)
    #ax3.set_ylim(-0.06,0.06)
    ax3.plot(days,metrics[0:index+1,0],'b')
    ax3.plot(days,metrics[0:index+1,1],'r')

    # Plot in the second cell (2nd column)
    ax6 = fig.add_subplot(main_gs[1, 2])    
    ax6.set_title('Acc')
    #ax6.set_xlim(Ly/2-20,Ly/2+20)
    #ax6.set_ylim(-0.06,0.06)
    ax6.plot(days,metrics[0:index+1,2],'b')
    ax6.plot(days,metrics[0:index+1,3],'r')   


    plt.tight_layout()

    # Display the plots
    plt.savefig(f'imagess/Solver_{index:0{6}}.png', dpi=300)
    plt.close()

    return
########################################################################

