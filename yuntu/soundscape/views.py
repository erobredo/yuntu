import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

def boxPlot(groupData,freqBins=None,timeCells=None,):
    dNames = list(groupData.keys())
    columns = groupData[dNames[0]].columns
    allEnergies = []

    i = 1
    ecol = "e_0"
    while ecol in columns:
        allEnergies.append(ecol)
        i += 1
        ecol = "e_"+str(i)

    data = []

    dEnergies = allEnergies
    if freqBins is not None:
        dEnergies = ["e_"+str(x) for x in freqBins]

    
    colors = cl.scales[str(len(dNames))]

    c = 0
    data = []
    for g in dNames:
    
        y = groupData[g][dEnergies[0]].values
        x = ["e_0" for d in range(y.size)]
        for i in range(1,len(dEnergies)):
            y_ =  groupData[g][dEnergies[i]].values
            x += ["e_"+str(i) for d in range(y_.size)]
            y = np.concatenate((y,y_))

        box = go.Box(y=y,x=x,name=g,marker={"color":colors[c]})
        data.append(box)
        c += 1

    layout = go.Layout(yaxis=dict(title='',zeroline=False),boxmode='group')
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)




    
        



