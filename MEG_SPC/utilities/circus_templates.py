#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: vagechirkov@gmail.com
"""
import matplotlib.pyplot as plt

class Templates:
    """ Here we read templates from the folder and create plots for them"""

    def __init__(self, paths, params, sensors='grad', 
                 cc_merge=0.9, cut_off=9,cut_off_top=200, N_t=80, 
                 MAD=6.0):
        '''
        Set parameters to open the folder with Syking Circus results

        '''
        import os
        import traceback
        #from circus.shared.parser import CircusParser
        
        self.case = paths.case
        self.case_dir = str(paths.case_meg)
        self.folder = str(paths.SPC)
        self.filename = paths.npy_file_name
        
        self.sensors = sensors
        
        self.params = params#CircusParser('%s/%s/%s'%(self.folder, output_dir, self.filename))
        self.spike_thresh = float(self.params.get('detection','spike_thresh'))
        self.N_t = self.params.getint('detection','N_t')
        self.cut_off = 0.1 #int(self.params.get('filtering','cut_off')[0])
        self.cut_off_top = cut_off_top
        self.cc_merge = float(self.params.get('clustering','cc_merge'))
        self.output_dir = 'cut_off_(%s,%s)_MAD_%s_N_t_%s_cc_merge_%s_%s'%(self.cut_off,self.cut_off_top, self.spike_thresh, self.N_t, self.cc_merge, self.sensors)
        
        try:
            os.makedirs('%s%s/Templates_plots'%(self.folder, self.output_dir),exist_ok=True)
            os.makedirs('%s%s/Templates_waveforms'%(self.folder, self.output_dir),exist_ok=True)
            os.makedirs('%s%s/SVG_plots'%(self.folder, self.output_dir),exist_ok=True)
            self.png_path = '%s%s/Templates_plots/'%(self.folder, self.output_dir)
            self.waveforms_path = '%s%s/Templates_waveforms'%(self.folder, self.output_dir)
            self.svg_path = '%s%s/SVG_plots/'%(self.folder, self.output_dir)
        except Exception: traceback.print_exc() #print('Wrong path!')
        Templates.read_templates(self)
        
    def read_templates(self):
        '''
        Read templates from the folder
        '''
        import pandas as pd        
        fname = '%s%s/Results/Templates_%s_all.xlsx'%(self.folder, self.output_dir, 
                                                      self.case)
        self.templates = pd.read_excel(fname)
        self.templates['Time'] = self.templates['Spiketimes']
        
    def save_template_for_aspire(self):
        import os 
        templates_for_aspire = self.templates.copy()
        templates_for_aspire['Template'] = templates_for_aspire['Template'].map(lambda x: int(x.split('_')[1]))
        os.makedirs(self.folder + 'Aspire/',exist_ok=True)
        templates_for_aspire.to_csv('%sAspire/Templates_%s_%s.csv'%(self.folder, self.filename[:-6], self.sensors), index=False)
        

    def templates_array(self):
        '''
        Extracts templates' data from Spyking Circus
        For even N_t don't forget +1  

        '''
        from circus.shared.files import load_data
        import numpy as np
        
        # The number of channels
        N_e = self.params.getint('data', 'N_e')
        # The temporal width of the template #sometimes needs +1 !!!!!
        N_t = self.params.getint('detection', 'N_t')+1 
        templates_cir = load_data(self.params, 'templates')
        self.temp_array = np.zeros((templates_cir.shape[1], N_e, N_t))
        
        for temp_n in range(templates_cir.shape[1]):
           self.temp_array[temp_n]  = templates_cir[:, temp_n].toarray().reshape(N_e, N_t)
    
    def spikes(self, first_sample):
        '''
        Make dic[template number] = list of aligned spikes
        
        In the dictionary, the spikes are sorted in 
        descending order of amplitude (GOF)
        
        Parameters
        ----------
        first_sample : int
            first timepoint from the 'fif' file in milliseconds

        Returns
        -------
        None.

        '''
        self.aligned_spikes = {}
        
        ### delete duplicates
        self.templates.sort_values('Amplitudes',ascending=False,inplace=True)
        self.templates.reset_index(inplace=True,drop=True)
        self.templates.drop_duplicates('Spiketimes',inplace=True)
        #self.templates.sort_values('Spiketimes',ascending=True,inplace=True)
        #self.templates.reset_index(inplace=True,drop=True)
        
        ### adding the first sample
        self.templates['Aligned_spikes'] = self.templates.Spiketimes + first_sample
        
        for temp_n in self.templates.Template.unique():
            temp_n_list = self.templates.loc[self.templates.Template==temp_n,"Aligned_spikes"].values.tolist()
            self.aligned_spikes[int(temp_n.split('_')[1])] = temp_n_list
    
    def masking_templates_data(self, temp_array,n_max_peaks=10):
        '''
        This function removes noise from the templates.

        Parameters
        ----------
        temp_array : numpy.ndarray
            array with templates data. 
            Shape -- templates, chanels, times
        n_max_peaks : int, optional
            Number of channels that will be not zero. The default is 10.

        Returns
        -------
        numpy.ndarray
            masked array with templates data (only n_max_peaks not 0). 
            Shape -- templates, chanels, times

        '''
        import numpy as np
        
        N_temp = temp_array.shape[0]
        N_e = temp_array.shape[1]
        N_t = temp_array.shape[2]
        
        temp_array = temp_array.reshape(N_temp,(N_e*N_t))
        temp_mask = np.zeros_like(temp_array)
    
        for temp_inx in range(temp_array.shape[0]):
            peaks_all = np.argsort(np.abs(temp_array[temp_inx,:]))
            peaks_max = np.zeros(n_max_peaks*N_t)
            n=0
            for i in peaks_all:
                if (i-(N_t//2+5)>0) & (i+(N_t//2+5)<len(peaks_all))&(i not in peaks_max):
                    peaks_max[n*N_t:(n+1)*N_t] = np.arange(i-N_t//2, i+N_t//2+1)
                    n+=1
                if n == n_max_peaks:
                    break

            temp_mask[temp_inx] = np.isin(np.arange(0,N_e*N_t),peaks_max)
           
    
        templates_c = temp_array.copy()
        templates_c[temp_mask==0] = 0        
        
        return templates_c.reshape(N_temp, N_e, N_t)


        
        
        
        
        
        
        
        
        
        
        