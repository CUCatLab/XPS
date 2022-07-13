import numpy as np
import scipy
import pandas as pd
from pandas import DataFrame as df
import yaml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import ipywidgets as ipw
from ipywidgets import Button, Layout
from lmfit import model, Model
import re
import os
from os import listdir
from os.path import isfile, join, dirname
import sys
from pathlib import Path
from IPython.display import clear_output
from pylab import rc

pio.renderers.default = 'notebook+plotly_mimetype'
pio.templates.default = 'simple_white'
pio.templates[pio.templates.default].layout.update(dict(
    title_y = 0.95,
    title_x = 0.5,
    title_xanchor = 'center',
    title_yanchor = 'top',
    legend_x = 0,
    legend_y = 1,
    legend_traceorder = "normal",
    legend_bgcolor='rgba(0,0,0,0)',
    margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=50, #top margin
        )
))

##### XPS #####

class XPS () :

    def __init__(self,folder='') :

        self.folder = folder
        self.data = {}

    def Normalize(self,data) :
        baseline = np.mean(data[-10:-1])
        data = data - baseline
        data = data/max(data)
        return data

    def BE(self,x,Ef) :
        x -= Ef
        x = -x
        return x

    def PlotRegion(self,region) :

        data = self.data

        fig = go.Figure()
        fig.update_layout(xaxis_title="Energy (eV)",yaxis_title="Intensity (au)",title=region,font=dict(size=18),
            autosize=False,width=1000,height=600)

        with open('XPS Parameters.yaml', 'r') as stream:
            par = yaml.safe_load(stream)
        filenames = list()
        for filename in par :
            filenames.append(filename)

        for filename in filenames :
            if filename in par and region in par[filename] :
                newdata = pd.read_csv(Path(self.folder+'/'+filename+'_'+str(par[filename][region]['Channel'])+'.csv'))
                data[filename] = {}
                data[filename][region] = {}
                for idx, col in enumerate(newdata) :
                    if idx == 0 :
                        x = newdata[col]
                        if 'xOffset' in par[filename] :
                            x = self.BE(x,par[filename]['xOffset'])
                    elif idx == 1 :
                        y = newdata[col]
                    else :
                        y += newdata[col]
                if 'Normalize' in par[filename][region] :
                    if par[filename][region]['Normalize'] :
                        y = self.Normalize(y)
                else: y = self.Normalize(y)
                if 'yFactor' in par[filename][region] :
                    y = y/par[filename][region]['yFactor']
                name = filename
                if 'Temperature' in par[filename] :
                    name += ', '+str(par[filename]['Temperature'])+' K'
                fig.add_trace(go.Scatter(x=x, y=y, name=name))
                data[filename][region]['x'] = x
                data[filename][region]['y'] = y

        self.data = data
        fig.show()

    def PlotFile(self,filename,region,runs=range(0,10000),normalize=False) :

        with open('XPS Parameters.yaml', 'r') as stream:
            par = yaml.safe_load(stream)

        fig = go.Figure()
        fig.update_layout(xaxis_title="Energy (eV)",yaxis_title="Intensity (au)",title=filename,font=dict(size=18),
            autosize=False,width=1000,height=600)
        data = pd.read_csv(Path(self.folder+'/'+filename+'_'+str(par[filename][region]['Channel'])+'.csv'))
        for idx, col in enumerate(data) :
            if idx == 0 :
                x = data[col]
                if 'xOffset' in par[filename] :
                    x = self.BE(x,par[filename]['xOffset'])
            else :
                if idx in runs :
                    y = data[col]
                    if normalize :
                        y = self.Normalize(y)
                    fig.add_trace(go.Scatter(x=x, y=y, name=idx))

        fig.show()