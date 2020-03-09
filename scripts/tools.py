import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clt
import PIL
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from ripser import ripser, lower_star_img
from persim import plot_diagrams







## Functions

def dictToNumpy(dictionary):
    output = list(dictionary.values())
    output = np.array(output)
    return(output)

def shaper(df, rows):
    df = np.reshape(df, (-1, rows))
    return(df)

def binary(df,ones='coral'):
    with np.errstate(divide='ignore',invalid='ignore'):
        if ones == 'coral':
            df = np.nan_to_num(df/df)
        elif ones == 'turf':
            df = (df-1)**2
        elif ones == 'macro':
            df = np.nan_to_num((df-2)**2/(df-2)**2)
    return(df)

def tdaPrep(df, rows, ones='coral'):
    df = binary(df,ones)
    dfi = 1-df   
    df = shaper(df,rows)
    dfi = shaper(dfi,rows)
    return(df, dfi)

def readPrep(simulation, timestep, rows):
    
    df = np.genfromtxt('coralModelStatsReadyOutput/modelOutput_switching'
                         +str(simulation)+'.csv', delimiter=',')
    df = np.reshape(df, (-1, rows**2))
    df = df[timestep,:]
    return(df)

def patchCounts(sim, rows):

# TDA measure for coral:

    df, dfi = tdaPrep(sim, rows, ones='coral')
    countCoralPatches = len(lower_star_img(df))
    countAlgaePatches = len(lower_star_img(dfi))

# TDA meaure for turf:

    df, dfi = tdaPrep(sim, rows, ones='turf')
    countTurfPatches = len(lower_star_img(df))

# TDA measure for macroalgae:

    df, dfi = tdaPrep(sim, rows, ones='macro')
    countMacroPatches = len(lower_star_img(df))
    
    return(countCoralPatches, countAlgaePatches, countTurfPatches, countMacroPatches)






def genOut(grazesim):
    x = list(grazesim.simulation[1].coralNodeCount.keys())

    multsimCoralCounts = np.array([tl.dictToNumpy(grazesim.simulation[i].coralNodeCount) 
                                   for i,val in enumerate(grazesim.simulation)]).transpose()

    multsimNeighborCounts = np.array([tl.dictToNumpy(grazesim.simulation[i].coralNeighborCount) 
                                   for i,val in enumerate(grazesim.simulation)]).transpose()

    C = multsimCoralCounts
    N = multsimNeighborCounts
    CN = np.divide(N, C, out=np.zeros_like(N), where=C!=0)
    CNr = np.divide(CN, C, out=np.zeros_like(N), where=C!=0)

    multsimMacroCounts = np.array([tl.dictToNumpy(grazesim.simulation[i].macroNodeCount) 
                                   for i,val in enumerate(grazesim.simulation)]).transpose()

    multsimNeighborCounts = np.array([tl.dictToNumpy(grazesim.simulation[i].macroNeighborCount) 
                                   for i,val in enumerate(grazesim.simulation)]).transpose()
    M = multsimMacroCounts
    mN = multsimNeighborCounts
    MN = np.divide(mN, M, out=np.zeros_like(mN), where=M!=0)
    MNr = np.divide(MN, M, out=np.zeros_like(mN), where=M!=0)


    multsimTurfCounts = np.array([tl.dictToNumpy(grazesim.simulation[i].turfNodeCount) 
                                   for i,val in enumerate(grazesim.simulation)]).transpose()

    multsimNeighborCounts = np.array([tl.dictToNumpy(grazesim.simulation[i].turfNeighborCount) 
                                   for i,val in enumerate(grazesim.simulation)]).transpose()
    T = multsimTurfCounts
    tN = multsimNeighborCounts
    TN = np.divide(tN, T, out=np.zeros_like(tN), where=T!=0)
    TNr = np.divide(TN, T, out=np.zeros_like(tN), where=T!=0)
    
    return(C,CN,CNr,T,TN,TNr,M,MN,MNr)




## Plotting Tool


def CountNeighborsPlots(x,C,CN,T,TN,M,MN,ti,tf,s):
    plt.figure(figsize=(20, 15))

    plt.subplot(321)

    plt.title("Subset Coral Count")
    plt.plot(x[ti:tf],C[ti:tf,s], alpha=0.4)
    plt.plot(x[ti:tf],C[ti:tf,:].mean(axis=1), 'black')

    plt.subplot(322)

    plt.title("Subset Coral-Coral Neighbor Average")
    plt.plot(x[ti:tf],CN[ti:tf,s], alpha=0.4)
    plt.plot(x[ti:tf],CN[ti:tf,:].mean(axis=1), 'black')

    plt.subplot(323)

    plt.title("Subset Turf Count")
    plt.plot(x[ti:tf],T[ti:tf,s], alpha=0.4)
    plt.plot(x[ti:tf],T[ti:tf,:].mean(axis=1), 'black')

    plt.subplot(324)

    plt.title("Subset Turf-Turf Neighbor Average")
    plt.plot(x[ti:tf],TN[ti:tf,s], alpha=0.4)
    plt.plot(x[ti:tf],TN[ti:tf,:].mean(axis=1), 'black')

    plt.subplot(325)

    plt.title("Subset Macroalgae Count")
    plt.plot(x[ti:tf],M[ti:tf,s], alpha=0.4)
    plt.plot(x[ti:tf],M[ti:tf,:].mean(axis=1), 'black')

    plt.subplot(326)

    plt.title("Subset Macroalgae-Macroalgae Neighbor Average")
    plt.plot(x[ti:tf],MN[ti:tf,s], alpha=0.4)
    plt.plot(x[ti:tf],MN[ti:tf,:].mean(axis=1), 'black')
    
def RCountNeighborsPlots(x,C,CNr,T,TNr,M,MNr,ti,tf,s):

    plt.figure(figsize=(20, 15))

    plt.subplot(321)

    plt.title("Subset Coral Count")
    plt.plot(x[ti:tf],C[ti:tf,s], alpha=0.4)
    plt.plot(x[ti:tf],C[ti:tf,:].mean(axis=1), 'black')

    plt.subplot(322)

    plt.title("Subset CNr index")
    plt.plot(x[ti:tf],CNr[ti:tf,s], alpha=0.4)
    plt.plot(x[ti:tf],CNr[ti:tf,:].mean(axis=1), 'black')


    plt.subplot(323)

    plt.title("Subset Turf Count")
    plt.plot(x[ti:tf],T[ti:tf,s], alpha=0.4)
    plt.plot(x[ti:tf],T[ti:tf,:].mean(axis=1), 'black')

    plt.subplot(324)

    plt.title("Subset TNr")
    plt.plot(x[ti:tf],TNr[ti:tf,s], alpha=0.4)
    plt.plot(x[ti:tf],TNr[ti:tf,:].mean(axis=1), 'black')

    plt.subplot(325)

    plt.title("Subset Macroalgae Count")
    plt.plot(x[ti:tf],M[ti:tf,s], alpha=0.4)
    plt.plot(x[ti:tf],M[ti:tf,:].mean(axis=1), 'black')

    plt.subplot(326)

    plt.title("Subset MNr")
    plt.plot(x[ti:tf],MNr[ti:tf,s], alpha=0.4)
    plt.plot(x[ti:tf],MNr[ti:tf,:].mean(axis=1), 'black')


def initialFinal(data1, data2, title1="initial", title2="final"):
    
    fig, (ax, ax2, cax) = plt.subplots(1,3, gridspec_kw= {'width_ratios':[1,1, 0.05]}, figsize=(10, 10))     
                                           #3 for colorbar
    fig.subplots_adjust(wspace=0.3)        #space btw subplots

    colors = ['pink', 'lightgreen','darkgreen']
    levels = [0, 1, 2]
    cmap, norm = clt.from_levels_and_colors(levels=levels, colors=colors, extend='max')

    im = ax.imshow(data1, cmap=cmap, norm=norm)
    im2 = ax2.imshow(data2, cmap=cmap, norm=norm)

    ip = InsetPosition(ax2, [1.05,0,0.05,1]) 
    cax.set_axes_locator(ip)

    fig.colorbar(im, cax=cax, ax=[ax,ax2])

    ax.set_title(title1)
    ax2.set_title(title2)
    plt.show()

def averageTimeSeries(x, coral, turf, algae, ylabel):
    
    x = np.arange(len(coral))
    
    fig, (ax, ax2, ax3) = plt.subplots(3,1, facecolor = 'w', sharey='row',
                                       figsize=(5, 14))
       
    im = ax.plot(x, coral.mean(axis=1), color='pink')
    im = ax.plot(x, coral.max(axis=1), color='pink')
    im = ax.plot(x, coral.min(axis=1), color='pink')

    im2 = ax2.plot(x, turf.mean(axis=1), color='lightgreen')
    im2 = ax2.plot(x, turf.max(axis=1), color='lightgreen')
    im2 = ax2.plot(x, turf.min(axis=1), color='lightgreen')

    im2 = ax3.plot(x, algae.mean(axis=1), color='green')
    im2 = ax3.plot(x, algae.max(axis=1), color='green')
    im2 = ax3.plot(x, algae.min(axis=1), color='green')
    
    title = 'Averages, Max, and Mins, of' + ylabel + 'over' + str(len(coral[1]))+ 'Simulations'
        
    fig.suptitle(title)
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


## Old TDA Tools


def finer(df, scale=3):
    columnCenter, rowCenter = 0, 0
    columnCount, rowCount = 0, 0
    columns, rows = len(df[0]), len(df)
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

def tda_prep_fineness(df, rows, fineness=2, refined_grid=False):
    df = binary(df)
    dfi = 1-df
    
    df = shaper(df,rows)
    dfi = shaper(dfi,rows)
    
    if refined_grid == True:
        df = finer(df, scale = fineness)
        dfi = finer(dfi, scale = fineness)

    return(df, dfi) 

## From SciKit https://ripser.scikit-tda.org/Lower%20Star%20Image%20Filtrations.html


from ripser import ripser, lower_star_img
from persim import plot_diagrams
from scipy import ndimage

def prep2(df):
    cells_grey = np.asarray(PIL.Image.fromarray(df).convert('L'))
    smoothed = ndimage.uniform_filter(cells_grey.astype(np.float64), size=10)
    #smoothed += 0.01 * np.random.randn(*smoothed.shape)
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


   

