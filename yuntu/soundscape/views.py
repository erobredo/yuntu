import numpy as np
import matplotlib.pyplot as plt


def plot_concat(x,figsize=(20,10),cmap="terrain",interpolation="sinc",xlabel=None,ylabel=None,xlabels=[],ylabels=[],colorbar=True,compose=False,path=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    if compose:
        plot = plt.imshow(x,interpolation=interpolation,extent=[0,1,0,1],aspect="auto")
    else:
        plot = plt.imshow(x,interpolation=interpolation,extent=[0,1,0,1],cmap=cmap,aspect="auto")
    
    if len(xlabels):
        xtickpos = np.append(np.arange(0,1,1/(len(xlabels)-1)),[1])
        ax.set_xticks(xtickpos)
        ax.set_xticklabels(xlabels)
        
    if len(ylabels):
        ytickpos = np.append(np.arange(0,1,1/(len(ylabels)-1)),[1])
        ax.set_yticks(ytickpos)
        ax.set_yticklabels(ylabels)
        
    if not compose and colorbar:
        fig.colorbar(plot, ax=ax)


    plt.xticks(rotation=90)

    if xlabel:
        ax.set_xlabel(xlabel)
        
    if ylabel:
        ax.set_ylabel(ylabel)

    if path is not None:
        plt.savefig(path)
        
    return fig,ax,plot
        
def plot_aggr(x,figsize=(20,10),cmap="terrain",interpolation="sinc",xlabel=None,ylabel=None,xlabels=[],ylabels=[],aggr="mean",colorbar=True,compose=False,path=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    if compose:
        plot = plt.imshow(x,interpolation=interpolation,extent=[0,1,0,1],aspect="auto")
    else:
        plot = plt.imshow(x,interpolation=interpolation,extent=[0,1,0,1],cmap=cmap,aspect="auto")
    
    if len(xlabels):
        xtickpos = np.append(np.arange(0,1,1/(len(xlabels)-1)),[1])
        ax.set_xticks(xtickpos)
        ax.set_xticklabels(xlabels)
        
    if len(ylabels):
        ytickpos = np.append(np.arange(0,1,1/(len(ylabels)-1)),[1])
        ax.set_yticks(ytickpos)
        ax.set_yticklabels(ylabels)

    if not compose and colorbar:
        fig.colorbar(plot, ax=ax)


    
    
    if xlabel:
        ax.set_xlabel(xlabel)
        
    if ylabel:
        ax.set_ylabel(ylabel)
    
    plt.xticks(rotation=90)

    if path is not None:
        plt.savefig(path)
        
    return fig,ax,plot



    
        



