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
from lmfit.models import GaussianModel, SkewedGaussianModel, VoigtModel, ConstantModel, LinearModel, QuadraticModel, PolynomialModel
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

class DataTools () :

    def __init__(self,folder='') :

        self.folder = folder
    
    def FileList (self) :

        with open('XPS Parameters.yaml', 'r') as stream:
            par = yaml.safe_load(stream)

        filenames = list()
        for filename in par :
            filenames.append(filename)
        
        return filenames
    
    def LoadData(self,filename,normalize=False) :

        with open('XPS Parameters.yaml', 'r') as stream:
            par = yaml.safe_load(stream)

        regions = ['C1s','O1s','Pt4f','Ef','Overview']
        
        data = {}
        if filename in par :
            for region in par[filename] :
                if region in regions :
                    data[region] = {}
                    channel = par[filename][region]['Channel']
                    newdata = pd.read_csv(Path(self.folder+'/'+filename+'_'+str(channel)+'.csv'))
                    if not filename in data :
                        data[filename] = {}
                    for idx, col in enumerate(newdata) :
                        if idx == 0 :
                            data[region]['x'] = newdata[col].values
                            if 'xOffset' in par[filename] :
                                data[region]['x'] -= par[filename]['xOffset']
                                data[region]['x'] = -data[region]['x']
                        else :
                            if idx == 1 :
                                data[region]['y'] = newdata[col].values.copy()
                            else :
                                data[region]['y'] += newdata[col].values.copy()
                            data[region]['y'+str(idx)] = newdata[col].values
                    data[region]['y'] /= idx
                    if normalize :
                        for col in data[region] :
                            if 'y' in col :
                                baseline = np.mean(data[region][col][-10:-1])
                                data[region][col] -= baseline
                        normalization = max(data[region]['y'])
                        for col in data[region] :
                            if 'y' in col :
                                data[region][col] /= normalization
                    if 'yFactor' in par[filename][region] :
                        for y in data[region] :
                            if 'y' in y :
                                data[region][y] /= par[filename][region]['yFactor']

        return data

    def PlotRegion(self,region,normalize=True) :

        with open('XPS Parameters.yaml', 'r') as stream:
            par = yaml.safe_load(stream)

        fig = go.Figure()
        fig.update_layout(xaxis_title="Energy (eV)",yaxis_title="Intensity (au)",title=region,font=dict(size=18),
            autosize=False,width=1000,height=600)
        filenames = list()
        for filename in par :
            filenames.append(filename)

        data = {}
        for filename in filenames :
            data[filename] = {}
            newdata = self.LoadData(filename,normalize)
            if region in newdata :
                data[filename][region] = {}
                data[filename][region]['x'] = newdata[region]['x']
                data[filename][region]['y'] = newdata[region]['y']
                name = filename
                if 'Temperature' in par[filename] :
                    name += ', '+str(par[filename]['Temperature'])+' K'
                fig.add_trace(go.Scatter(x=data[filename][region]['x'], y=data[filename][region]['y'], name=name))

        self.data = data
        fig.show()

    def PlotFile(self,filename,region,normalize=True) :

        data = self.LoadData(filename,normalize)

        fig = go.Figure()
        fig.update_layout(xaxis_title="Energy (eV)",yaxis_title="Intensity (au)",title=filename,font=dict(size=18),
            autosize=False,width=1000,height=600)
        for col in data[region] :
            if 'y' in col :
                fig.add_trace(go.Scatter(x=data[region]['x'], y=data[region][col], name=col))

        self.data = data
        fig.show()

##### XPS Tools #####

class XPS :
    
    def __init__(self,folder='') :

        self.dt = DataTools(folder)
        self.folder = folder
        self.par = self.dt.par
    
    def SetModel(self, Data, par) :
        
        ModelString = list()
        for Peak in par :
            ModelString.append((Peak,par[Peak]['model']))
        
        for Model in ModelString :
            try :
                FitModel
            except :
                if Model[1] == 'Constant' :
                    FitModel = ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    FitModel = LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    FitModel = GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    FitModel = SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    FitModel = VoigtModel(prefix=Model[0]+'_')
            else :
                if Model[1] == 'Constant' :
                    FitModel = FitModel + ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    FitModel = FitModel + LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    FitModel = FitModel + GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    FitModel = FitModel + SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    FitModel = FitModel + VoigtModel(prefix=Model[0]+'_')
        
        ModelParameters = FitModel.make_params()
        names = list()
        for col in Data :
            if 'y' in col :
                names.append(col)
        FitsParameters = df(index=ModelParameters.keys(),columns=names)
        
        self.FitModel = FitModel
        self.ModelParameters = ModelParameters
        self.FitsParameters = FitsParameters
    
    def SetParameters(self, par, Value=None) :
        
        ModelParameters = self.ModelParameters
        
        ParameterList = ['intercept','offset','amplitude','center','sigma']
        Parameters = {'Standard': par}

        for Dictionary in Parameters :
            for Peak in Parameters[Dictionary] :
                for Parameter in Parameters[Dictionary][Peak] :
                    if Parameter in ParameterList :
                        for Key in Parameters[Dictionary][Peak][Parameter] :
                            if Key != 'set' :
                                exec('ModelParameters["'+Peak+'_'+Parameter+'"].'+Key+'='+str(Parameters[Dictionary][Peak][Parameter][Key]))
                            else :
                                exec('ModelParameters["'+Peak+'_'+Parameter+'"].'+Key+str(Parameters[Dictionary][Peak][Parameter][Key]))
                                    
        self.ModelParameters = ModelParameters
        self.FitsParameters = df(index=ModelParameters.keys(),columns=self.names)
    
    def Fit(self,Data,par,**kwargs) :

        self.SetModel(Data,par)
        ModelParameters = self.ModelParameters
        FitsParameters = self.FitsParameters

        self.names = list()
        for col in Data :
            if 'y' in col :
                self.names.append(col)
        names = self.names
        
        FitModel = self.FitModel
        
        Fits = df(index=Data['x'],columns=names)
        FitsResults = list()
        FitsComponents = list()
        
        for idx,col in enumerate(Data) :

            if 'y' in col :
                self.SetParameters(par)
                x = Data['x']
                y = Data[col]
                FitResults = FitModel.fit(y, ModelParameters, x=x, nan_policy='omit')
                fit_comps = FitResults.eval_components(FitResults.params, x=x)
                fit_y = FitResults.eval(x=x)
                ParameterNames = [i for i in FitResults.params.keys()]
                for Parameter in (ParameterNames) :
                    FitsParameters[col][Parameter] = FitResults.params[Parameter].value
                Fits[names[idx-1]] = fit_y
                FitsResults.append(FitResults)
                FitsComponents.append(fit_comps)
                
                sys.stdout.write(("\rFitting %i out of "+str(len(Data)-1)) % (idx))
                sys.stdout.flush()
            
        self.Fits = Fits
        self.FitsParameters = FitsParameters
        self.FitsResults = FitsResults
        self.FitsComponents = FitsComponents
    
    def ShowFits(self,xLabel='',yLabel='') :
        
        Data = self.data
        Fits = self.Fits
        
        FitsParameters = self.FitsParameters
        FitsComponents = self.FitsComponents
        
        for idx,Column in enumerate(Data) :
            
            plt.figure(figsize = [6,4])
            plt.plot(Data[region]['x'], Data[region][col],'k.', label='Data')
            plt.plot(Fits.index, Fits[col], 'r-', label='Fit')
            for Component in FitsComponents[idx] :
                if not isinstance(FitsComponents[idx][Component],float) :
                    plt.fill(Fits.index, FitsComponents[idx][Component], '--', label=Component, alpha=0.5)
            plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
            plt.xlabel(xLabel), plt.ylabel(yLabel)
            plt.title(str(Column))
            plt.show()
            
            Peaks = list()
            for Parameter in FitsParameters.index :
                Name = Parameter.split('_')[0]
                if Name not in Peaks :
                    Peaks.append(Name)

            string = ''
            for Peak in Peaks :
                string = string + Peak + ' | '
                for Parameter in FitsParameters.index :
                    if Peak == Parameter.split('_')[0] : 
                        string = string + Parameter.split('_')[1] + ': ' + str(round(FitsParameters[Column][Parameter],2))
                        string = string + ', '
                string = string[:-2] + '\n'
            print(string)
            print(75*'_')
    
    def FitData(self) :
        
        with open('XPS Parameters.yaml', 'r') as stream:
            self.par = yaml.safe_load(stream)
        
        Data = self.data
        par = self.par
        DataName = str(self.Files.value)
        
        print('Data: '+DataName)
        
        ##### Fit Data #####

        regions = ['C1s','O1s','Pt4f','Ef']
        
        for region in par[self.Files.value] :
            if region in regions :
                if 'Models' in par[self.Files.value][region] :
                    self.Fit(Data[region],par[DataName][region]['Models'])
                    Fits = self.Fits
                    FitsParameters = self.FitsParameters
        
                    print('\n'+100*'_')
        
                    for idx,col in enumerate(Data[region]) :

                        if 'y' in col :

                            plt.figure(figsize = [12,4])
                            plt.plot(Data[region]['x'], Data[region][col],'k.', label='Data')
                            plt.plot(Fits.index, Fits[col], 'r-', label='Fit')
                            plt.xlabel('Energy (eV)'), plt.ylabel('Intensity (au)')
                            plt.title(region+', '+str(col))
                            for Component in self.FitsComponents[idx-1] :
                                if not isinstance(self.FitsComponents[idx-1][Component],float) :
                                    plt.fill(Fits.index, self.FitsComponents[idx-1][Component], '--', label=Component, alpha=0.5)
                            plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
                            plt.show()
                            
                            Peaks = list()
                            for Parameter in FitsParameters.index :
                                Name = Parameter.split('_')[0]
                                if Name not in Peaks :
                                    Peaks.append(Name)
                            string = ''
                            for Peak in Peaks :
                                if 'assignment' in par[self.Files.value][region]['Models'][Peak] :
                                    string += par[self.Files.value][region]['Models'][Peak]['assignment'] + ' | '
                                else :
                                    string += Peak + ' | '
                                for Parameter in FitsParameters.index :
                                    if Peak == Parameter.split('_')[0] : 
                                        string += Parameter.split('_')[1] + ': ' + str(round(FitsParameters[col][Parameter],2))
                                        string += ', '
                                string = string[:-2] + '\n'
                            print(string)
                            print(100*'_')
                    FitsParameters = FitsParameters.T
                    FitsParameters = FitsParameters[np.concatenate((FitsParameters.columns.values[1:],FitsParameters.columns.values[0:1]))]
        
        # Plot Trends
        
        # UniqueParameters = []
        # [UniqueParameters.append(x.split('_')[1]) for x in FitsParameters.columns if x.split('_')[1] not in UniqueParameters][0]
        # for uniqueParameter in UniqueParameters :
        #     fig = go.Figure()
        #     for parameter in FitsParameters :
        #         if uniqueParameter in parameter :
        #             Name = parameter.split('_')[0]
        #             if 'assignment' in par['Fit']['Models'][Name] :
        #                 Name = par['Fit']['Models'][Name]['assignment']
        #             fig.add_trace(go.Scatter(x=FitsParameters.index,y=FitsParameters[parameter],name=Name,mode='lines+markers'))
        #     fig.update_layout(xaxis_title='Temperature (K)',yaxis_title=uniqueParameter,title=DataName,legend_title='',width=800,height=400)
        #     fig.show()
        
        ##### Store Fits ####
        
        # self.Fits = Fits
        # self.FitsData = Data
        # self.FitsBackground = Background
        # self.FitsParameters = FitsParameters
        # self.FitsAssignments = FitsAssignments
        
        # ##### Widgets #####

        # def CopyData_Clicked(b) :
        #     Data.to_clipboard()
        # CopyData = ipw.Button(description="Copy Data")
        # CopyData.on_click(CopyData_Clicked)

        # def CopyFits_Clicked(b) :
        #     Fits.to_clipboard()
        # CopyFits = ipw.Button(description="Copy Fits")
        # CopyFits.on_click(CopyFits_Clicked)

        # def CopyParameters_Clicked(b) :
        #     FitsParameters.to_clipboard()
        # CopyParameters = ipw.Button(description="Copy Parameters")
        # CopyParameters.on_click(CopyParameters_Clicked)

        # def Save2File_Clicked(b) :
        #     os.makedirs(Folders['Fits'], exist_ok=True)
        #     FitsFile = Folders['Fits'] +'/' + DataName + '.hdf'
        #     Data.to_hdf(FitsFile,'Data')
        #     Fits.to_hdf(FitsFile,'Fits',mode='a')
        #     FitsParameters.to_hdf(FitsFile,'Fits_Parameters',mode='a')
        #     FitsAssignments.to_hdf(FitsFile,'Fits_Assignments',mode='a')
        # Save2File = ipw.Button(description="Save to File")
        # Save2File.on_click(Save2File_Clicked)

        # display(ipw.Box([CopyData,CopyFits,CopyParameters,Save2File]))
    
    def UI(self) :

        folder = self.folder

        dt = DataTools (folder)
        
        out = ipw.Output()
        
        self.Files = ipw.Dropdown(
            options=dt.FileList(),
            description='Select File',
            layout=Layout(width='70%'),
            style = {'description_width': '150px'},
            disabled=False,
        )

        # def ShowData_Clicked(b) :
        #     with out :
        #         clear_output(True)
        #         self.LoadData(Folders['Parameters'],self.ParametersFiles.value+'.yaml')
        #         plt.figure(figsize = [8,6])
        #         x = self.Data.index.values
        #         y = self.Data.columns.values
        #         z = np.transpose(self.Data.values)
        #         plt.xlabel('Wavenumber (cm$^-$$^1$)', fontsize=16)
        #         plt.ylabel('Temperature (K)', fontsize=16)
        #         plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
        #         plt.title(self.DataName, fontsize=16)
        #         pcm = plt.pcolor(x, y, z, cmap='jet', shading='auto')
        #         plt.show()
        # ShowData = ipw.Button(description="Show Data")
        # ShowData.on_click(ShowData_Clicked)
        
        def FitData_Clicked(b) :
            with out :
                clear_output(True)
                self.data = dt.LoadData(self.Files.value,normalize=True)
                self.FitData()
        FitData = ipw.Button(description="Fit Data")
        FitData.on_click(FitData_Clicked)
        
        display(self.Files)
        # display(ipw.HBox([ShowData,FitData]))
        display(FitData)
        display(out)