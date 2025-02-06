import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
########################################################################
def plotField(psi,Lx,Ly,fileName):
    x = np.linspace(0,Lx,psi1.shape[0])
    y = np.linspace(0,Ly,psi1.shape[1])
    X,Y = np.meshgrid(x,y)
    
    levels = np.linspace(-2.5, 2.5, 10)
    plt.set_cmap('bwr')
    contour = plt.contourf(X,Y,np.transpose(psi),levels=levels)
    plt.colorbar(contour, label='Value')
    plt.savefig(f'images/plot_{fileName}.png', dpi=300)
    plt.show()
    return

########################################################################
def plotAverages(data1,data2,L,fileName):
    y = np.linspace(0,L,data1.shape[0])
    plt.clf()
    plt.plot(y,data1,'b')
    plt.plot(y,data2,'r')
    plt.savefig(f'images/{fileName}.png', dpi=300)
    plt.show()
    return

########################################################################
def fourPlots(Lx,Ly,psi1,psi2,u1,u2,uv1,uv2,index):
    Ny = psi1.shape[1]

    levels = np.linspace(-2.5, 2.5, 10)
    plt.set_cmap('bwr')
    # Create a figure
    fig = plt.figure(figsize=(12, 6))

    # Create a main GridSpec layout with 1 row and 3 columns
    main_gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 0.05,0.07,1], wspace=0.15)

    # Plot in the first cell (1st column)
    ax1 = fig.add_subplot(main_gs[0, 0])
    ax1.set_title('Psi 1')

    x = np.linspace(0,Lx,psi1.shape[0])
    y = np.linspace(0,Ly,psi1.shape[1])
    X,Y = np.meshgrid(x,y)
    
    #ax1.set_cmap('bwr')
    contour1 = plt.contourf(X,Y,np.transpose(psi1),levels=levels)    

    # Plot in the second cell (2nd column)
    ax2 = fig.add_subplot(main_gs[0, 1])
    #ax2.plot([1, 2, 3], [1, 2, 3])
    contour2 = plt.contourf(X,Y,np.transpose(psi2),levels=levels)
    #plt.colorbar(contour2, label='Value')
    ax2.set_title('Psi 2')

    # Add colorbar in the fourth cell (4th column)
    cbar_ax = fig.add_subplot(main_gs[:, 2])
    #cbar = plt.colorbar(contour, cax=cbar_ax)
    cbar = plt.colorbar(contour2, cax=cbar_ax)
    #cbar.set_label('Colorbar')

    # Create a nested GridSpec in the third cell (3rd column)
    nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[0, 4], height_ratios=[1, 1])

    # Plot in the top half of the third cell (1st row of nested GridSpec)
    ax3 = fig.add_subplot(nested_gs[0, 0])
    ax3.set_xlim(Ly/2-20,Ly/2+20)
    ax3.set_ylim(-0.5,1.2)
    ax3.plot(y,u1,'b')
    ax3.plot(y,u2,'r')
    ax3.yaxis.tick_right()
    #ax3.plot([1, 2, 3], [3, 2, 1])
    ax3.set_title('Mean Zonal Wind')

    # Plot in the bottom half of the third cell (2nd row of nested GridSpec)
    ax4 = fig.add_subplot(nested_gs[1, 0])
    ax4.set_xlim(Ly/2-20,Ly/2+20)
    ax4.set_ylim(-0.06,0.06)
    ax4.plot(y,uv1,'b')
    ax4.plot(y,uv2,'r')
    ax4.yaxis.tick_right()
    ax4.set_title('Zonal-mean EMF')

    # Adjust the layout to prevent overlap
    #plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)


    # Display the plots
    plt.savefig(f'images/fig_{index:0{4}}.png', dpi=300)
    plt.show()

    return
########################################################################
month = 105
day = 1
days = np.load(f'ICs/month_{month}.npy')

psi1 = days[day,:,:,0]
psi2 = days[day,:,:,1]

#plotField(psi2,46,68,f'day{day}_month{month}')
########################################################################
u1 = np.load(f'Averages/u1_month_95.npy')
u2 = np.load(f'Averages/u2_month_95.npy')
uv1 = np.load(f'Averages/uv1_month_95.npy')
uv2 = np.load(f'Averages/uv2_month_95.npy')
#plotAverages(data1,data2,68,"test")
########################################################################
#for month in range(1:355)
fourPlots(46,68,psi1,psi2,u1,u2,uv1,uv2,0)



########################################################################
import concurrent.futures

# Define a function to perform the task for each iteration
def task(index):
    # Your task logic here
    month = int(index/ 150) + 1
    day = index - (month-1)*150 + 1
    print(f"Task {index} {month} {day} is running")
    
    data = np.load(f'ICs/month_{month}.npy')
    psi1 = data[day,:,:,0]
    psi2 = data[day,:,:,1]

    u1 = np.load(f'Averages/u1_month_{month}.npy')
    u2 = np.load(f'Averages/u2_month_{month}.npy')
    uv1 = np.load(f'Averages/uv1_month_{month}.npy')
    uv2 = np.load(f'Averages/uv2_month_{month}.npy')

    #fourPlots(46,68,psi1,psi2,u1,u2,uv1,uv2,index)

    return index

# Number of iterations in the loop
num_iterations = 355*150

# Create a ThreadPoolExecutor with a maximum of 4 threads
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    # Submit tasks to the executor for each iteration
    future_to_index = {executor.submit(task, index): index for index in range(num_iterations)}
    
    # Wait for all tasks to complete
    for future in concurrent.futures.as_completed(future_to_index):
        index = future_to_index[future]
        try:
            result = future.result()
            #print(f"Task {index} completed with result: {result}")
        except Exception as e:
            print("X")
            #print(f"Task {index} generated an exception: {e}")
########################################################################
