import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

def finer(array, scale=3, rows=10, columns=10):
    columnCenter = 0
    rowCenter = 0
    columnCount = 0
    rowCount = 0

    fine = np.zeros((rows*scale,columns*scale))

    for r in range(0,rows*scale):
        for c in range(0,columns*scale):
            
            fine[r,c] = array[rowCenter, columnCenter]
            
            columnCount += 1
            if columnCount == scale:
                columnCenter += 1
                columnCount = 0
        rowCount += 1
        if rowCount == scale:
            rowCenter += 1
            rowCount = 0
        columnCenter = 0
        
    return(fine)

def initialFinal(data1, data2, title1="initial", title2="final"):
    fig, (ax, ax2, cax) = plt.subplots(1,3,        #subplot 3 to include color bar
     gridspec_kw={'width_ratios':[1,1, 0.05]})     #creates space allocated to each subplot
    fig.subplots_adjust(wspace=0.3)                #space btw subplots

    colors = ['pink', 'lightgreen','darkgreen']
    levels = [0, 1, 2]

    cmap, norm = clt.from_levels_and_colors(levels=levels, colors=colors, extend='max')

    im = ax.imshow(data1, cmap=cmap,norm=norm)
    im2 = ax2.imshow(data2,cmap=cmap,norm=norm)

    ip = InsetPosition(ax2, [1.05,0,0.05,1]) 
    cax.set_axes_locator(ip)

    fig.colorbar(im, cax=cax, ax=[ax,ax2])

    ax.set_title(title1)
    ax2.set_title(title2)
    plt.show()

def averageHists(x, coralCount, turfCount, algaeCount, ylabel):
    fig, (ax, ax2, ax3) = plt.subplots(1,3, facecolor = 'w', sharey='row')
    im = ax.bar(x, coralCount.mean(axis=1), color='pink')
    im2 = ax2.bar(x, turfCount.mean(axis=1), color='lightgreen')
    im2 = ax3.bar(x, algaeCount.mean(axis=1), color='darkgreen')
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    ax2.set_xlabel("Time")
    ax3.set_xlabel("Time")
    ax.set_facecolor('white')
    ax2.set_facecolor('white')
    ax3.set_facecolor('white')
    plt.show()
    
def timeSeries(x, x1, x2, x3,legend1='Coral', legend2='Turf', legend3='Algae'):
    fig1 = plt.figure()
    x = np.arange(NumberOfRuns)
    im = plt.plot(x, coralCount[:,1], color='pink')
    im2 = plt.plot(x, turfCount[:,1], color='lightgreen')
    im2 = plt.plot(x, algaeCount[:,1], color='darkgreen')
    plt.legend(['Coral', 'Turf', 'Algae'], loc='upper left', fontsize = 'medium')
    plt.ylabel("Percent")
    plt.xlabel("Time")
    plt.show()
    
    
def instance(reef, timestep, simulation, grid = True, rows=10):
    composition = np.zeros(len(reef.nodes))
    for n, val in enumerate(reef.nodes):
        composition[n] = reef.nodes[n].type[timestep,simulation]
    if grid == True:
        composition = np.reshape(composition, (-1, rows))
    return(composition)
          
def speciesCount(reef, type, simulation):
    count = np.zeros((len(reef.nodes[0].type[:,0]),simulation))
    for s in range(0,simulation):
        for r in range(0,len(count)):
            count[r,s] = np.count_nonzero(instance(reef, r, s, False)== type)
    return(count)
        
        
        
        #for n in enumerate(reef.nodes):
         #   if reef.nodes[n].type[r, simulation] == type:
          #      count[r] += 1
    
#coralCount[n,s] = np.count_nonzero(types[n,:,s] == 0)