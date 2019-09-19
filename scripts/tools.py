import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clt
import PIL
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

## Plotting Tools

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
    
def timeSeries(x, x1, x2, x3, legend1='Coral', legend2='Turf', legend3='Algae'):
    fig1 = plt.figure()
    im = plt.plot(x, x1, color='pink')
    im2 = plt.plot(x, x2, color='lightgreen')
    im2 = plt.plot(x, x3, color='darkgreen')
    plt.legend([legend1, legend2, legend3], loc='upper left', fontsize = 'medium')
    plt.ylabel("Percent")
    plt.xlabel("Time")
    plt.show()

## TDA Tools

def binaryizer(df):
    with np.errstate(divide='ignore',invalid='ignore'):
        df = np.nan_to_num(df/df)
    return(df)
    
def shaper(df, rows):
    df = np.reshape(df, (-1, rows))
    return(df)

def finer(df, rows=10, columns=10, scale=3):
    columnCenter, rowCenter = 0, 0
    columnCount, rowCount = 0, 0
    fine = np.zeros((rows*scale,columns*scale))
    for r in range(0,rows*scale):
        for c in range(0,columns*scale):            
            fine[r,c] = df[rowCenter, columnCenter]           
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

def tda_prep(df, columns, rows, fineness, refined_grid = True):
    df = binaryizer(df)
    dfi = 1-df
    
    df = shaper(df,rows)
    dfi = shaper(dfi,rows)
    
    if refined_grid == True:
        df = finer(df, rows, columns, scale = fineness)
        dfi = finer(dfi, rows, columns, scale = fineness)
    return(df, dfi)  


## From SciKit https://ripser.scikit-tda.org/Lower%20Star%20Image%20Filtrations.html


from ripser import ripser, lower_star_img
from persim import plot_diagrams
from scipy import ndimage

def prep2(df):
    cells_grey = np.asarray(PIL.Image.fromarray(df).convert('L'))
    smoothed = ndimage.uniform_filter(cells_grey.astype(np.float64), size=10)
    smoothed += 0.01 * np.random.randn(*smoothed.shape)
    return(smoothed)


def pointDef(dgm, threshold):
    idxs = np.arange(dgm.shape[0])
    idxs = idxs[np.abs(dgm[:, 1] - dgm[:, 0]) > threshold]
    return(idxs)

def plotPoints(points, df, dgm):
    X, Y = np.meshgrid(np.arange(df.shape[1]), np.arange(df.shape[0]))
    X = X.flatten()
    Y = Y.flatten()
    for idx in points:
        bidx = np.argmin(np.abs(df + dgm[idx, 0]))
        plt.scatter(X[bidx], Y[bidx], 20, 'k')

def plotTDA(points, yo, df, dgm):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(yo)
    plt.colorbar()
    plotPoints(points, df, dgm)
    plt.title("Test Image")
    plt.subplot(122)
    plot_diagrams(dgm)
    plt.title("0-D Persistence Diagram")

    plt.tight_layout()
    plt.show()
    
    
    
## Outdated
    
    
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