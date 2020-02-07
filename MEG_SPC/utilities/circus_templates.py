#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: vagechirkov@gmail.com
"""

class Templates:
    """ Here we read templates from the folder and create plots for them"""

    def __init__(self, paths, params, sensors='grad'):
        '''
        Set parameters to open the folder with Syking Circus results

        '''
        self.results_path = paths['SPC_results']        
        self.sensors      = sensors
        self.params       = params
        self.N_t          = self.params.getint('detection','N_t')
        self.output_dir   = paths['SPC_output']
        
        ### 
        self.read_templates()
        self.save_template_for_aspire()
        self.templates_array()
        
    def read_templates(self):
        '''
        Read templates from the folder
        '''
        import pandas as pd        
        self.excel_path = self.results_path/'Templates_{}.xlsx'.format(self.sensors)
        self.templates = pd.read_excel(self.excel_path)
        self.templates['Time'] = self.templates['Spiketimes']
        
    def save_template_for_aspire(self):
        temp = self.templates.copy()
        temp['Template'] = temp['Template'].map(lambda x: int(x.split('_')[1]))
        temp.to_csv(self.excel_path.with_suffix('.scv'), index=False)
        del temp
        

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

        '''
        self.aligned_spikes = {}
        ### delete duplicates
        self.templates.sort_values('Amplitudes',ascending=False,inplace=True)
        self.templates.reset_index(inplace=True,drop=True)
        self.templates.drop_duplicates('Spiketimes',inplace=True)
        
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
