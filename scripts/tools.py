import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clt
import PIL
import os
import pandas as pd
import re
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from ripser import ripser, lower_star_img
from persim import plot_diagrams
from coralModel import Reef

## For Model Runs

#Model Setup


def generate_checker_board(rows, columns):
    m = rows + 2
    n = columns + 2
    checker_board = np.tile(np.array([[0,1,2],[1,2,0],[2,0,1]]),
                           ((m+2)//3, (n+2)//3))
    return(checker_board)

def generate_blob(blob_percent, blob_value, rows, columns, number_of_nodes):
    not_blob = [a for a in [0,1,2] if a != blob_value]
    center = (rows/2, columns/2)
    distance_grid = np.array([Reef.distance([i+.5,j+.5], center)
                             for i in range(0,rows)
                             for j in range(0,columns)])
    max_distance = np.sort(distance_grid)[round(blob_percent * number_of_nodes)]
    blob_locations = (np.where(distance_grid.reshape(rows, columns) < max_distance))
    blob_locations = [(blob_locations[0][n],blob_locations[1][n])
                     for n in range(0,len(blob_locations[0]))]
    return(blob_locations, not_blob)

#Data Pulling

def density_extract(Moorea, node, count):
    if count == 0:
        neighbors = 0
    else:
        neighbors = np.array([Moorea.nodes[n].density/
                              Moorea.nodes[n].density.sum()
                              for n,val in enumerate(Moorea.nodes)
                              if Moorea.nodes[n].type == node]).mean(axis=0)
    return(neighbors)

def density_extract_specific(Moorea, node_type, neighbor_type, count):
    if Count == 0:
        neighbors = 0
    else:
        neighbors = np.array([Moorea.nodes[n].density/
                              Moorea.nodes[n].density.sum()
                              for n,val in enumerate(Moorea.nodes)
                              if Moorea.nodes[n].type == node_type]).mean(axis=0)[neighbor_type]
    return(neighbors)


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

def tda_prep(df, rows, ones='coral'):
    df = binary(df,ones)
    dfi = 1-df
    df = shaper(df,rows)
    dfi = shaper(dfi,rows)
    return(df, dfi)

def patch_counts(sim, rows):

# TDA measure for coral:

    df, dfi = tda_prep(sim, rows, ones='coral')
    count_coral_patches = len(lower_star_img(df))
    count_algae_patches = len(lower_star_img(dfi))

# TDA meaure for turf:

    df, dfi = tda_prep(sim, rows, ones='turf')
    count_turf_patches = len(lower_star_img(df))

# TDA measure for macroalgae:

    df, dfi = tda_prep(sim, rows, ones='macro')
    count_macro_patches = len(lower_star_img(df))

    return(count_coral_patches, count_algae_patches, count_turf_patches, count_macro_patches)

def pandas_histogram(dataframe, variable, by):
    test = dataframe.hist(column=variable,
                          by=by, figsize=(3,10),
                          layout=(3,1), sharex=True, sharey=True)
    return(test)


## For Model Analysis

def view_runs(top_directory):

    ## Get path + name for all csvs within top directory
    files = [path+'/'+file for path, d, f in os.walk(top_directory) for
             file in f if file.endswith(".csv") and path.find('images') == -1]

    ## Pull parameter information from path + name
    overview_of_runs = pd.concat([pd.DataFrame([np.array(re.split('[a-z-/.]+',files[n])[1:-1], dtype='int')]) for n in range(0,len(files))])

    ## Assign column names + create file index
    overview_of_runs.columns = ['Rows', 'Columns', 'Grid Option', 'Grazing', 'Initial Coral Percent', 'Initial Macroalgae Percent', 'r', 'd', 'a', 'y', 'Time', 'Record Rate', 'Number of Simulations']

    overview_of_runs = overview_of_runs.set_index([pd.Series([n for n in range(0,len(files))])])
    overview_of_runs['File']=overview_of_runs.index

    return(files, overview_of_runs)

def view_images(top_directory, overview_of_runs):

    ## Get path + name for all csvs within top directory
    files = [path+'/'+file for path, d, f in os.walk(top_directory) for file in f if file.endswith(".csv") and path.find('images') != -1]

    ## Pull parameter information from path + name
    overview_of_images = pd.concat([pd.DataFrame([np.array(re.split('[a-z-/.]+',files[n])[1:-1], dtype='int')]) for n in range(0,len(files))])

    ## Assign column names + create file index
    overview_of_images.columns = ['Rows', 'Columns', 'Grid Option', 'Grazing',
                                  'Simulation', 'Timestep','Initial Coral Percent',
                                  'Initial Macroalgae Percent', 'r', 'd', 'a', 'y',
                                  'Time', 'Record Rate', 'Number of Simulations']

    overview_of_images = overview_of_images.set_index([pd.Series([n for n in range(0,len(files))])])
    overview_of_images['Image File']=overview_of_images.index

    overview_of_images = pd.merge(overview_of_images, overview_of_runs)

    return(files, overview_of_images)

def load_runs(files, subset):

    #subset = needs to be a subset of viewRuns, output
    simulation_data = pd.concat([pd.DataFrame(pd.read_csv(
        files[f])).assign(File = np.repeat(f, len(pd.read_csv(files[f])))) for f in list(subset.index)])
    dataframe = pd.merge(simulation_data, subset, on='File')

    return(dataframe)

def add_percent(df):

    df['CoralPercent'] = df['CoralCount']/(df['Rows']*df['Columns']) * 100
    df['TurfPercent'] = df['TurfCount']/(df['Rows']*df['Columns']) * 100
    df['MacroalgaePercent'] = df['MacroalgaeCount']/(df['Rows']*df['Columns']) * 100

    return(df)


def label_crash_statistics(df, coral_success):

    timestep = df['Timestep'].max()

    df['Coral Success'] = -999

    df.loc[(df['Timestep']==timestep) & (df['Coral Percent'] > coral_success), 'Coral Success'] = 1
    df.loc[(df['Timestep']==timestep) & (df['Coral Percent'] < coral_success), 'Coral Success'] = 0
    df.loc[(df['Timestep']==timestep) & (df['Coral Percent'] == 0), 'Coral Success'] = -1

    df.loc[:,'Coral Success'] = df.groupby(['File','Simulation'])['Coral Success'].transform('max')

    return(df)

def scaled_by_number_of_nodes(df):

    df['CoralPatchCount_Scaled'] = df['CoralPatchCount']/df['CoralCount']
    df['TurfPatchCount_Scaled'] = df['TurfPatchCount']/df['TurfCount']
    df['MacroPatchCount_Scaled'] = df['MacroPatchCount']/df['MacroalgaeCount']
    df['AlgaePatchCount_Scaled'] = df['AlgaePatchCount']/(df['MacroalgaeCount']+df['TurfCount'])
    df['AlgaePatchCount_MScaled'] = df['AlgaePatchCount']/(df['MacroalgaeCount'])
    df['AlgaePatchCount_TScaled'] = df['AlgaePatchCount']/(df['TurfCount'])

    df['CoralNeighbors_Scaled'] = df['Coral-CoralNeighbors']/df['CoralCount']
    df['TurfNeighbors_Scaled'] = df['Turf-TurfNeighbors']/df['TurfCount']
    df['MacroNeighbors_Scaled'] = df['Macro-MacroNeighbors']/df['MacroalgaeCount']

    return(df.replace(np.inf, 0))

def normalize(df):

    df_n=(df-df.min())/(df.max()-df.min())
    df['coral_coral_neighbors']=df_n['Coral-CoralNeighbors']
    df['coral_turf_neighbors']=df_n['Coral-TurfNeighbors']
    df['coral_macroalgae_neighbors']=df_n['Coral-MacroNeighbors']

    df['turf_coral_neighbors']=df_n['Turf-CoralNeighbors']
    df['turf_turf_neighbors']=df_n['Turf-TurfNeighbors']
    df['turf_macroalgae_neighbors']=df_n['Turf-MacroNeighbors']

    df['macro_coral_neighbors']=df_n['Macro-CoralNeighbors']
    df['macro_turf_neighbors']=df_n['Macro-TurfNeighbors']
    df['macro_macroalgae_neighbors']=df_n['Macro-MacroNeighbors']

    df['turf_percent']=df_n['TurfPercent']
    df['macroalgae_percent']=df_n['MacroalgaePercent']
    df['coral_percent']=df_n['CoralPercent']

    df['coral_patches']=df_n['CoralPatchCount']
    df['turf_patches']=df_n['TurfPatchCount']
    df['macroalgae_patches']=df_n['MacroPatchCount']
    df['patches_algae']=df_n['AlgaePatchCount']

    return(df)

def lag(df, column_name, shift_value):
    record_rate = df['Record Rate'][0]
    df[column_name+'_lag'+str(shift_value*record_rate)] = df.groupby(
        ['File','Simulation'])[column_name].shift(shift_value)
    return(df)

def add_lag(df, lag_column_names, shift_values):
    for column_name in lag_column_names:
        for shift_value in shift_values:
            df = lag(df, column_name, shift_value)
    return(df)


def add_crash_time(df):
    df['TotalTimeToCrash'] = 0
    for File in df['File'].unique():
        for Simulation in df[df['File']==File]['Simulation'].unique():
            time = df[(df['File']==File) &
                      (df['Simulation']==Simulation)].shape[0] * df['Record Rate']
            df.loc[(df['File']==File) &
                   (df['Simulation']==Simulation),'TotalTimeToCrash'] = time

    df['TimeToCrash'] = df['TotalTimeToCrash'] - df['Timestep']

    df.loc[dfcrash['Coral Success'] != -1, 'Time To Crash'] = -100

    return(df)

# for new names including "threshold"

def viewRuns2(topDirectory):

    ## Get path + name for all csvs within top directory
    files = [path+'/'+file for path, d, f in os.walk(topDirectory) for file in f if file.endswith(".csv") and path.find('images') == -1]

    ## Pull parameter information from path + name
    overviewOfRuns = pd.concat([pd.DataFrame([np.array(re.split('[a-z-/.]+', files[n])[1:-1], dtype='int')]) for n in range(0,len(files))])

    ## Assign column names + create file index
    overviewOfRuns.columns = ['Rows', 'Columns', 'Grid Option', 'Grazing',
                              'Initial Coral Percent', 'Initial Macroalgae Percent',
                              'r', 'd', 'a', 'y', 'Time', 'Record Rate',
                              'Number of Simulations', 'Threshold']

    overviewOfRuns = overviewOfRuns.set_index([pd.Series([n for n in range(0,len(files))])])
    overviewOfRuns['File']=overviewOfRuns.index

    return(files, overviewOfRuns)

def viewImages2(topDirectory, overviewOfRuns):

    ## Get path + name for all csvs within top directory
    files = [path+'/'+file for path, d, f in os.walk(topDirectory) for file in f if file.endswith(".csv") and path.find('images') != -1]

    ## Pull parameter information from path + name
    overviewOfImages = pd.concat([pd.DataFrame([np.array(re.split('[a-z-/.]+',files[n])[1:-1], dtype='int')]) for n in range(0,len(files))])

    ## Assign column names + create file index
    overviewOfImages.columns = ['Rows', 'Columns', 'Grid Option', 'Grazing', 'Simulation', 'Timestep','Initial Coral Percent', 'Initial Macroalgae Percent', 'r', 'd', 'a', 'y', 'Time', 'Record Rate', 'Number of Simulations','Threshold']

    overviewOfImages = overviewOfImages.set_index([pd.Series([n for n in range(0,len(files))])])
    overviewOfImages['ImageFile']=overviewOfImages.index

    overviewOfImages = pd.merge(overviewOfImages, overviewOfRuns)

    return(files, overviewOfImages)




'''

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



def dictToNumpy(dictionary):
    output = list(dictionary.values())
    output = np.array(output)
    return(output)


## single images
def pullImageIndex(overviewOfImages, file, simulation, timestep):
    imageFileIndex = overviewOfImages[(overviewOfImages['File']==file)&
                                      (overviewOfImages['Simulation']==simulation)&
                                      (overviewOfImages['Timestep']==timestep)]['ImageFile']
    return(int(imageFileIndex))

def loadImage(imageFiles, overviewOfImages, file, simulation, timestep, nrows):
    index = pullImageIndex(overviewOfImages, file, simulation, timestep)
    image = np.genfromtxt(imageFiles[index])
    return(shaper(image, nrows))

## multiple images

def pullImageIndexes(overviewOfImages, file, simulation):

    imageFileIndexes = [index for index in overviewOfImages[(overviewOfImages['File']==file)&
                                                            (overviewOfImages['Simulation']==simulation)]['ImageFile']]
    return(imageFileIndexes)

def loadImages(imageFiles, overviewOfImages, file, simulation, nrows):

    indexes = pullImageIndexes(overviewOfImages, file, simulation)
    images = {int(overviewOfImages[overviewOfImages['ImageFile']==index]['Timestep']):
              shaper(np.genfromtxt(imageFiles[index]),nrows) for index in indexes}

    return(images)

    '''
