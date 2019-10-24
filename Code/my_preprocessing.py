class NoStdStreams(object):
    
    def __init__(self,stdout = None, stderr = None):
        import sys, os
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr
    def __enter__(self):
        import sys
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr
    def __exit__(self, exc_type, exc_value, traceback):
        import sys
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

class Preprocessing:
    """ Here we run all preprocessing steps"""

    def __init__(self, directory, case, dir_root='D:/Valerii/45_cases/'):
        from pathlib import Path
        import os
        import pandas as pd
        self.dir_root = dir_root
        self.dir_case = directory
        self.dir_tsss = directory + 'MEG_data/tsss_mc/'
        self.dir_tsss_filt_1Hz = directory + 'MEG_data/tsss_mc_1Hz/'
        self.dir_psd = directory + 'PSD/'
        self.dir_bdip = directory + 'VD_spikes/bdip_files/'
        self.dir_spc = directory + 'Spyking_circus/'
        self.dir_art_corr = directory + 'MEG_data/tsss_mc_artefact_correction'
        self.dir_ica = directory + 'ICA/'
        self.case = case
        self.fif_files = Preprocessing.file_names_fif(self, self.dir_tsss)
        os.makedirs(self.dir_case + 'MEG_data/tsss_mc_artefact_correction',exist_ok=True)
        os.makedirs(self.dir_case + 'Spyking_circus',exist_ok=True)
  
    
    def file_names(self, directory):
        """ Read all files for directory"""
        import os
        files = os.listdir(directory)
        all_files = [x for x in files]
        return all_files

    def file_names_fif(self, directory):
        """ Read all fif files for directory"""
        import os
        import re
        
        files = os.listdir(directory)
        fif_files = [x for x in files if x.split('.')[-1]=='fif']
        reg_exp = 'run\d+|_\d+_|rnu\d+|sleep\d+|_ictal\d+'
        run_numbers = [int(re.findall('\d+', re.findall(reg_exp, i)[0])[0]) if re.findall(reg_exp, i) != [] else 0 for i in fif_files]
        fif_files_in_order = [x for _,x in sorted(zip(run_numbers,fif_files))]
        return fif_files_in_order
    
    def _boundary_time(self, files_info, data):
        """ Create boundaries time between concatinated files"""
        import numpy as np
        if 'boundaries(s)' not in files_info:
            files_info['boundaries(s)'] = None
        boundary_time = []
        boundary_time.append(0.0)
        for i in range(len(data.annotations)):
            if data.annotations.description[i] == 'EDGE boundary':
                #print(data.annotations.onset[i])
                boundary_time.append(np.round(data.annotations.onset[i] - data.first_samp/1000, 3))
        ind_conc = files_info[files_info['file'] == 'concatenated_files'].index[0]
        files_info['boundaries(s)'].loc[ind_conc] = boundary_time
        
    def concatenate_files(self):
        """ Saving concatinated files """
        import os
        import pandas as pd
        import mne
        from ipypb import irange
        os.makedirs(self.dir_tsss_filt_1Hz,exist_ok=True)
        print('Readind, concatenating and saving tsss_mc_filt_1 ...')
        with NoStdStreams():
            data = mne.io.read_raw_fif(self.dir_tsss + self.fif_files[0], preload=True, verbose=False).filter(1., None, fir_design='firwin')
            data.save('%s%s_filt_1.fif'%(self.dir_tsss_filt_1Hz, self.fif_files[0][:-4]), overwrite=True, verbose=False)
        data.pick_types(meg=True, eeg=False, stim=False, eog=False)

        if len(self.fif_files)>1:
            for i in irange(1, len(self.fif_files), label='Files'):
                with NoStdStreams():
                    data_0 = mne.io.read_raw_fif(self.dir_tsss + self.fif_files[i], preload = True, verbose = False).filter(1., None, fir_design='firwin')
                    data_0.save('%s%s_filt_1.fif'%(self.dir_tsss_filt_1Hz, self.fif_files[i][:-4]), overwrite=True, verbose=False)
                data.append(data_0.pick_types(meg=True, eeg=False, stim=False, eog=False), preload=True)
                del data_0
                
        cnames = ('Patient','folder','file','length(s)','N_manual_spikes','ICA_bad_comp_auto','Manual_components','bad_annot_begin','bad_annot_duration','first_samp(s)','files_order','boundaries(s)')
        files_info = pd.DataFrame(columns=cnames)
        files_info.at[0,'Patient'] = self.case
        files_info.at[0,'folder'] = self.dir_case
        files_info.at[0,'file'] = 'concatenated_files'
        files_info.at[0,'first_samp(s)'] = data.first_samp/1000
        files_info.at[0,'length(s)'] = data.n_times/1000
        files_info.at[0,'files_order'] = self.fif_files
        Preprocessing._boundary_time(self, files_info, data)
        files_info.to_excel('%sfiles_info_%s.xlsx'%(self.dir_tsss, self.case), index=False)
        
        with NoStdStreams():
            data.save('%s%s_concatenated_files_tsss_mc_1_Hz_meg_ch.fif'%(self.dir_tsss_filt_1Hz, self.case), overwrite=True, verbose=False)
        del data, files_info
    
    def read_concatenated_file(self):
        import mne
        with NoStdStreams():
            self.concatenated_file = mne.io.read_raw_fif('%s%s_concatenated_files_tsss_mc_1_Hz_meg_ch.fif'%(self.dir_tsss_filt_1Hz, self.case), preload=False, verbose=False)

    def _plot_psd_one_file(self, fif_file, data):
        """ Plot PSD for one file"""
        import gc
        import mne
        import numpy as np
        import matplotlib.pyplot as plt
        
        picks = mne.pick_types(data.info, meg=True, eeg=False, eog=False, stim=False)
        fig = data.plot_psd(tmax=np.inf, picks=picks, fmax=150, show=False, verbose=False)
        fig.suptitle("%s_%s"%(self.case, fif_file), fontsize="x-large")
        fig.axes[0].set_xticks(np.arange(0, 150, step=10))
        fig.axes[1].set_xticks(np.arange(0, 150, step=10))

        fig.subplots_adjust(wspace=0.5, hspace=0.5, left=0.125, right=0.9, top=0.85, bottom=0.2)
        fname = '%s%s_%s.png'%(self.dir_psd, self.case, fif_file)
        fig.savefig(fname, format='png', dpi=300)
        fig.clf()
        plt.close(fig)
        del data, fname
        gc.collect()
    
    def plot_psd(self):
        """ Plot PSD for all files"""
        import mne
        from ipypb import track
        print('Plotting PSD ...') 

        for fif_file in track(self.fif_files, label='Plots '):
            with NoStdStreams():
                data = mne.io.read_raw_fif(self.dir_tsss + fif_file, preload=False, verbose=False)
            Preprocessing._plot_psd_one_file(self, fif_file[:-4], data)
        del data
    
    def _ICA_fit(self, data):
        """ Fit ICA """
        import mne
        
        ica = mne.preprocessing.ICA(n_components=0.999, method='fastica', random_state=0, max_iter=100, verbose=False)
        picks = mne.pick_types(data.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
        ica.fit(data, picks=picks, reject_by_annotation=True, verbose=False)
        return ica

    def _ICA_artifacts_det(self, data, ica, save_fname): 
        """ Automatic ICA detection """
        import mne
        #https://mne-tools.github.io/dev/auto_tutorials/preprocessing/plot_artifacts_correction_ica.html#tut-artifact-ica
        ecg_channels = mne.pick_types(data.info, ecg=True, meg = False, eeg = False, eog = False)
        if ecg_channels != []:
            ecg_channels = data.info['ch_names'][ecg_channels[0]]
        else: ecg_channels = None
        eog_channels = mne.pick_types(data.info, ecg=False, meg = False, eeg = False, eog = True)
        if eog_channels != []:
            eog_channels = [data.info['ch_names'][eog_channels[0]], data.info['ch_names'][eog_channels[1]]]
        else: eog_channels = None
        with NoStdStreams():    
            ica.detect_artifacts(data, ecg_ch=ecg_channels, eog_ch=eog_channels, ecg_criterion=0.5, eog_criterion=0.5)
        ica.save(save_fname)
        return ica

    def _plot_prop(self, directory, ica, data):
        import gc
        import matplotlib.pyplot as plt
        
        for i in range(ica.n_components_):
            with NoStdStreams():
                fig = ica.plot_properties(data, picks=i, show=False, psd_args={'fmax': 200.})
            fname = '%s%s_ICA_component_%s_properties.png'%(directory, self.case,i)
            fig[0].savefig(fname, format='png', dpi=150)
            [f.clf() for f in fig]
            [plt.close(f) for f in fig]
        del data, ica, fname
        gc.collect()

    def _plot_sources(self, directory, ica, data, iterations=4, window=5):
        import math
        import gc
        import matplotlib.pyplot as plt
        
        for i in range(math.ceil(ica.n_components_/10)):
            for ii in range(iterations):
                m = 0 + i*10
                n = 10 + i*10
                if (ica.n_components_ - n) <= 0:
                    n = ica.n_components_
                fig = ica.plot_sources(data, picks=range(m,n), start=0+window*ii, stop=window+window*ii, show=False)
                fname = '%s%s_ICA_sources_%s_%s_time_%s_%s_s.png'%(directory, self.case, m, n,0+window*ii, window+window*ii)
                fig.savefig(fname, format='png', dpi=150)
                fig.clf()
                plt.close(fig)
        del data, ica, fname
        gc.collect()
    
    def ICA_plots(self, fif_files):
        """ Fit ICA and detect automatic bad components 
        Plot components"""
        import gc
        import os
        import mne
        from ipypb import track
        
        print('ICA fitting, artefscts detecting, properties and sources plotting...')
        for fif_file in track(fif_files, label = 'Files '):
            with NoStdStreams():
                data = mne.io.read_raw_fif('%s%s_filt_1.fif'%(self.dir_tsss_filt_1Hz, fif_file[:-4]), preload = False, verbose = False)       
            ica = Preprocessing._ICA_fit(self, data)
            if len(data.times)/1000 > 600:
                data_0 = data.copy().crop(10, 600)
                ica = Preprocessing._ICA_artifacts_det(self, data_0, ica, '%s%s_ica.fif'%(self.dir_ica, fif_file[:-4]))
            else:
                ica = Preprocessing._ICA_artifacts_det(self, data, ica, '%s%s_ica.fif'%(self.dir_ica, fif_file[:-4]))
            print('Automatic bad components for %s:  %s'%(fif_file, ica.exclude))
            if len(data.times)/1000 > 100:
                data_0 = data.copy().crop(10, 100)
                os.makedirs('%scomponent_properties/%s/'%(self.dir_ica, fif_file[:-4]),exist_ok=True)
                Preprocessing._plot_prop(self, '%scomponent_properties/%s/'%(self.dir_ica, fif_file[:-4]), ica, data_0)
                os.makedirs('%ssources/%s/'%(self.dir_ica, fif_file[:-4]),exist_ok=True)
                Preprocessing._plot_sources(self, '%ssources/%s/'%(self.dir_ica, fif_file[:-4]), ica, data_0, iterations=6)
            else:
                print('File %s is too small (%s s)'%(fif_file, len(data.times)/1000))
                data_0 = data.copy()
                os.makedirs('%scomponent_properties/%s/'%(self.dir_ica, fif_file[:-4]),exist_ok=True)
                Preprocessing._plot_prop(self, '%scomponent_properties/%s/'%(self.dir_ica, fif_file[:-4]), ica, data_0)
            del data_0, data, ica
            gc.collect()
        gc.collect()

    def read_bdip(self):
        """ Read bdip files """
        import os
        import numpy as np
        import pandas as pd
        from ipypb import irange
        files = os.listdir(self.dir_bdip)
        dip_table = pd.DataFrame(columns=('file','begin','end','r0','rd','Q','goodness','errors_computed','noise_level','single_errors','error_matrix',
                                           'conf_vol','khi2','prob','noise_est', 'file_name'))
        line = 0
        if len(files)>1:
            for i in irange(len(files), label = 'Files'):
                if (files[i].split('.')[-1] == 'bdip'):
                    data_all = np.fromfile(self.dir_bdip + files[i], dtype='>f4', count=-1)
                    for ii in range(int(len(data_all)/49)):
                        data = data_all[49*ii:49 + 49*ii]
                        dip_table.loc[line] = [files[i], data[1], data[2], data[3:6], data[6:9], data[9:12], data[12], data[13], 
                                               data[14], data[15:20], data[20:45], data[45], data[46], data[47], data[48], files[i].split('.')[0]]
                        line += 1

        dip_table[['file','begin','end','r0','rd','Q','goodness','errors_computed','noise_level','single_errors','error_matrix',
                   'conf_vol','khi2','prob','noise_est', 'file_name']].to_excel('%sbdip_manual_spikes_%s.xlsx'%(self.dir_bdip,self.case), index=False)
        self.mspikes = dip_table
        
    def _add_ica_artefacts(self, files_info):
        """ Add ICA artefacts to files_info"""
        import mne 
        for ica_file in Preprocessing.file_names(self, self.dir_ica):
            if (ica_file[-4:] == '.fif'):
                if ica_file[-19:-4]=='_manual_bad_ica':
                    ica = mne.preprocessing.read_ica('%s%s'%(self.dir_ica, ica_file), verbose=False)
                    files_info.loc[files_info.file == ica_file[:-19]+'.fif', 'ICA_bad_comp_manual'] = str(ica.exclude)
                elif  ica_file[-8:-4]=='_ica':
                    ica = mne.preprocessing.read_ica('%s%s'%(self.dir_ica, ica_file), verbose=False)
                    files_info.loc[files_info.file == ica_file[:-8]+'.fif', 'ICA_bad_comp_auto'] = str(ica.exclude)
        return files_info
    
    
    def _add_bad_annot_files_info(self, files_info):
        """ Add bad annotation for each fif file """
        import mne
        import pandas as pd
        import re

        if 'bad_annot_begin' not in files_info:
            files_info['bad_annot_begin'] = None
        if 'bad_annot_duration' not in files_info:
            files_info['bad_annot_duration'] = None
        if 'N_manual_spikes' not in files_info:
            files_info['N_manual_spikes'] = None
        if 'ICA_bad_comp_manual' not in files_info:
            files_info['ICA_bad_comp_manual'] = None
        if 'ICA_bad_comp_auto' not in files_info:
            files_info['ICA_bad_comp_auto'] = None
        if 'Patient' not in files_info:
            files_info['Patient'] = None
        if 'file' not in files_info:
            files_info['file'] = None
        if 'first_samp(s)' not in files_info:
            files_info['first_samp(s)'] = None
        if 'length(s)' not in files_info:
            files_info['length(s)'] = None
        if 'files_order' not in files_info:
            files_info['files_order'] = None
        if 'folder' not in files_info:
            files_info['folder'] = None
        
        Preprocessing.read_bdip(self)
        ### Manual spikes cleaning
        # Should appear twice: in all_dip file and in the individual dipole file 
        self.final_manual_spikes = pd.DataFrame([], columns=self.mspikes.columns.values)
        for i, row in self.mspikes.iterrows():
            for fif_file in self.fif_files:
                if (re.findall('at\d+', row.file) != [])&(re.findall(fif_file[:-4], row.file) != [])&(fif_file[:-4] in self.mspikes.file_name.unique()):
                    if sum(self.mspikes.begin.values == row.begin)>1:
                        self.final_manual_spikes = self.final_manual_spikes.append(pd.DataFrame({'file':row.file, 'begin':row.begin, 'end':row.end, 'r0':[row.r0], 'rd':[row.rd], 'Q':[row.Q], 'goodness':row.goodness,'errors_computed':row.errors_computed, 'noise_level':row.noise_level, 'single_errors':[row.single_errors], 'error_matrix':[row.error_matrix],'conf_vol':row.conf_vol, 'khi2':row.khi2, 'prob':row.prob, 'noise_est':row.noise_est, 'file_name':row.file_name}), ignore_index=True)
        self.final_manual_spikes.sort_values("begin", inplace=True)
        self.final_manual_spikes.reset_index(inplace=True, drop=True)
        
        mspikes_nunique_by_file = pd.DataFrame(self.final_manual_spikes.groupby('file_name')['begin'].nunique())
        
        for fif_file in self.fif_files:
            if fif_file not in files_info.file.values:
                files_info.loc[len(files_info),'file'] = fif_file
            
            with NoStdStreams():    
                data = mne.io.read_raw_fif('%s%s'%(self.dir_tsss,fif_file), preload=False, verbose=False)
            duration = []
            begining = []
            if len(data.annotations) != 0:
                for i in range(len(data.annotations)):
                    if (data.annotations[i]['description'][:3] == 'BAD')&(data.annotations[i]['duration'] != 0.0):
                        duration.append(data.annotations[i]['duration'])
                        begining.append(data.annotations[i]['onset'] - data.first_samp/1000)

            files_info.loc[files_info.file == fif_file,'bad_annot_begin'] = str(begining)
            files_info.loc[files_info.file == fif_file,'bad_annot_duration'] =  str(duration)
            files_info.loc[files_info.file == fif_file,'Patient'] = self.case
            files_info.loc[files_info.file == fif_file,'folder'] = self.dir_case
            files_info.loc[files_info.file == fif_file,'file'] = fif_file
            files_info.loc[files_info.file == fif_file,'first_samp(s)'] = data.first_samp/1000
            files_info.loc[files_info.file == fif_file,'length(s)'] = data.n_times/1000
            files_info.loc[files_info.file == fif_file,'files_order'] = [fif_file]
            
            if fif_file[:-4] in self.final_manual_spikes.file_name.unique(): #Not all fif files have spikes
                files_info.loc[files_info.file == fif_file,'N_manual_spikes'] = mspikes_nunique_by_file.loc[fif_file[:-4]][0]
            del data
                
        return files_info
        
    def update_files_info(self):
        """ Update information in files_info """
        from pathlib import Path
        import pandas as pd
        import mne

        if not Path('%sfiles_info_%s.xlsx'%(self.dir_tsss, self.case)).is_file():
            cnames = ('Patient','file','length(s)','N_manual_spikes','ICA_bad_comp_auto','Manual_components','bad_annot_begin','bad_annot_duration','first_samp(s)','files_order','boundaries(s)')
            files_info = pd.DataFrame(columns=cnames)
        else: files_info = pd.read_excel('%sfiles_info_%s.xlsx'%(self.dir_tsss, self.case))
            
        files_info = Preprocessing._add_bad_annot_files_info(self, files_info)
        files_info = Preprocessing._add_ica_artefacts(self, files_info)

        files_info.to_excel('%sfiles_info_%s.xlsx'%(self.dir_tsss, self.case), index=False)
        self.files_info = files_info

    def _add_manual_artefacts(self, all_cases_files):
        """ Add manual artefactis from cases_files """
        from pathlib import Path
        import re
        import mne
        
        for file in all_cases_files.loc[all_cases_files.folder==self.dir_case,'file'].values:   
            manual_comp_str = str(all_cases_files.loc[((all_cases_files.folder == self.dir_case)|(all_cases_files.folder==self.dir_case[:-1]))&(all_cases_files.file == file),'Manual_components'].values)
            ICA_bad_comp_manual = str(self.files_info.loc[self.files_info.file == file,'ICA_bad_comp_manual'].values[0])
            
            if (set(re.findall("\d+", manual_comp_str)) != set(re.findall("\d+", ICA_bad_comp_manual)))&(file in self.files_info.file.values):
                m_comp = list(map(int, re.findall("\d+", manual_comp_str)))
                self.files_info.loc[self.files_info.file == file,'ICA_bad_comp_manual'] = str(m_comp)
                
                if Path('%s%s_ica.fif'%(self.dir_ica, file[:-4])).is_file():
                    with NoStdStreams():
                        ica = mne.preprocessing.read_ica('%s%s_ica.fif'%(self.dir_ica, file[:-4]), verbose=False)
                        ica.exclude = m_comp
                        ica.save('%s%s_manual_bad_ica.fif'%(self.dir_ica, file[:-4]))
                    del ica
                else: print('No ICA for file %s'%file)
                self.files_info.to_excel('%sfiles_info_%s.xlsx'%(self.dir_tsss, self.case), index=False)
    
    def _styler_highlight_not_skipped(self, skipped):
        return ['background-color: yellow' if v == False else '' for v in skipped]

    def _highlight_greater_than(s, threshold, column):
        import pandas as pd
        is_max = pd.Series(data=False, index=s.index)
        is_max[column] = s.loc[column] == threshold
        return ['' if is_max.any() else 'background-color: yellow' for v in is_max] #'background-color: white'        
        
    def update_cases_files(self):
        """ Update information in cases_files """
        from pathlib import Path
        import pandas as pd
        import shutil
        
        if not Path('%sall_cases_files.xlsx'%self.dir_root).is_file():
            cnames = ('Patient','folder','file','length(s)','N_manual_spikes','skipped','ICA_bad_comp_auto','ICA_bad_comp_manual','bad_annot_begin','bad_annot_duration','Manual_components')
            all_cases_files = pd.DataFrame(columns=cnames)
            all_cases_files.to_excel('%sall_cases_files.xlsx'%self.dir_root, index=False)
            print("No file 'all_cases_files.xlsx' in the root folder")
        
        Preprocessing.update_files_info(self)
        all_cases_files = pd.read_excel('%sall_cases_files.xlsx'%self.dir_root)
        Preprocessing._add_manual_artefacts(self, all_cases_files)
        
        all_cases_files_cols = ['Patient','folder','file','length(s)','N_manual_spikes','ICA_bad_comp_auto','ICA_bad_comp_manual','bad_annot_begin','bad_annot_duration']
        files_info_sel = self.files_info.loc[self.files_info.file != 'concatenated_files', all_cases_files_cols].copy()

        for index, row in files_info_sel.iterrows():
            row_file = pd.DataFrame(row).T
            all_cases_files.loc[((all_cases_files.folder==self.dir_case)|(all_cases_files.folder==self.dir_case[:-1]))&(all_cases_files.file==row_file.file.values[0]), all_cases_files_cols] = row_file.values.tolist()[0]
            
            if row_file.file.values[0] not in all_cases_files.loc[all_cases_files.folder==self.dir_case, 'file'].values:
                all_cases_files.loc[len(all_cases_files), all_cases_files_cols] = row_file.values.tolist()[0]
                all_cases_files.loc[len(all_cases_files)-1, 'folder'] = self.dir_case
                all_cases_files.loc[len(all_cases_files)-1, 'skipped'] = True
                all_cases_files.sort_values(['Patient','folder','N_manual_spikes'],inplace=True,ascending=[True,True,False])
                all_cases_files.reset_index(inplace=True, drop=True)
        
        all_cases_files.sort_values(['Patient','folder','N_manual_spikes'],inplace=True,ascending=[True,True,False])
        all_cases_files.reset_index(inplace=True, drop=True)
        all_cases_files.style.apply(Preprocessing._highlight_greater_than, threshold=True, column='skipped', axis=1).to_excel('%sall_cases_files.xlsx'%self.dir_root, index=False)
        #all_cases_files.to_excel('{}all_cases_files_190918.xlsx'.format(self.dir_root), index=False)
        self.all_cases_files = all_cases_files
        self.file_fif_for_circus = all_cases_files.loc[(all_cases_files.folder==self.dir_case)&(all_cases_files.skipped==False),'file'].values[0]
                
        #shutil.copyfile('{}all_cases_files_190918.xlsx'.format(self.dir_root), "C:/Users/TFedele/Google Drive/__PC_HSE/Results/MEG Valerii/Valerii/all_cases_files_190918.xlsx")
        
        
    def apply_ica(self):
        """ Apply ICA """
        import os
        import mne
        import pandas as pd
        import numpy as np
        import gc
        from pathlib import Path
        Preprocessing.update_cases_files(self)
    
        for file in self.all_cases_files.loc[(self.all_cases_files.folder==self.dir_case)&(self.all_cases_files.skipped == False),'file'].values:
            if Path('%s%s_manual_bad_ica.fif'%(self.dir_ica, file[:-4])).is_file():
                ica = mne.preprocessing.read_ica('%s%s_manual_bad_ica.fif'%(self.dir_ica, file[:-4]), verbose=False)
                with NoStdStreams():
                    data = mne.io.read_raw_fif(self.dir_tsss+file, preload=True, verbose=False)
                    ica.apply(data)
                    data.save('%s/%s_art_corr.fif'%(self.dir_art_corr, file[:-4]), overwrite=True, verbose=False)

                raw_data = data.pick_types(meg=True, eeg=False,stim=False, eog=False).get_data(reject_by_annotation = 'omit')
                os.makedirs('%s%s_%s'%(self.dir_spc,self.case,file[:-4]),exist_ok=True)
                np.save("%s%s_%s/%s_%s.npy"%(self.dir_spc,self.case,file[:-4], file[:-4], 0), (raw_data).astype(float))
                print('File %s saved as %s_%s.npy in the folder %s%s_%s'%(file, file[:-4], 0, self.dir_spc,self.case,file[:-4]))
                del ica
            else:
                print('No manual components ICA file %s'%file)
                with NoStdStreams():
                    data = mne.io.read_raw_fif(self.dir_tsss+file, preload=True, verbose=False)
                    data.save('%s/%s_art_corr.fif'%(self.dir_art_corr, file[:-4]), overwrite=True, verbose=False)
                
                raw_data = data.pick_types(meg=True, eeg=False,stim=False, eog=False).get_data(reject_by_annotation = 'omit')
                os.makedirs('%s%s_%s'%(self.dir_spc,self.case,file[:-4]),exist_ok=True)
                np.save("%s%s_%s/%s_%s.npy"%(self.dir_spc,self.case,file[:-4], file[:-4], 0), (raw_data).astype(float))
                print('File %s saved as %s_%s.npy in the folder %s%s%s'%(file, file[:-4], 0, self.dir_spc,self.case,file[:-4]))
                
        del raw_data, data
        gc.collect()
        
    def _read_array_xlsx_cell(self, string):
        """ Read array from excel cell """
        import re
        s = re.split('[|\'|\',|]', string)
        names = []
        for i in range(int((len(s)-3)/3)+1):
            names.append(s[1+i*3])
        return names

    def _add_boundary_bad_annotations_time(self):
        """ Add boundary time and bad components time for manual spikes """
        import pandas as pd
        import re
        
        Preprocessing.update_files_info(self)
        self.mspikes.drop_duplicates('begin', inplace=True)
        self.mspikes.reset_index(inplace=True, drop=True)
        self.mspikes_for_full_plot = pd.DataFrame(columns=self.mspikes.columns.values)
        bound = []
        bound.append(0.0)
        for i in range(len(re.findall("\d+\.\d+", self.files_info[self.files_info['file'] == 'concatenated_files']['boundaries(s)'].values[0]))):
            btime = float(re.findall("\d+\.\d+", self.files_info[self.files_info['file'] == 'concatenated_files']['boundaries(s)'].values[0])[i])
            if btime != 0.0:
                bound.append(btime)
        
        files_order = Preprocessing._read_array_xlsx_cell(self, self.files_info[self.files_info['file'] == 'concatenated_files'].files_order.values[0])

        lines = 0
        for i in range(len(self.mspikes)):
            fif_file = '%s.fif'%self.mspikes.at[i, 'file_name']
            if fif_file in files_order:
                self.mspikes.at[i, 'begin'] += bound[files_order.index(fif_file)]
                self.mspikes.at[i, 'begin'] *= 1000
                
                bad_annot_begin_str =  str(self.files_info.bad_annot_begin[self.files_info.file == fif_file])
                bad_annot_begin = list(map(float,re.findall("\d+\.\d+", bad_annot_begin_str)))
                bad_annot_duration_str = str(self.files_info.bad_annot_duration[self.files_info.file == fif_file])
                bad_annot_duration = list(map(float, re.findall("\d+\.\d+", bad_annot_duration_str)))

                if (bad_annot_duration != []):
                    self.mspikes.at[i, 'begin'] += bad_annot_duration[0]*1000
                self.mspikes_for_full_plot.loc[lines] = self.mspikes.loc[i]
                lines += 1
            
    def plot_full_case(self, duration=15, group_by='position', start=0.0, lowpass=45, highpass=1):
        """ Plot concatinated file """
        import mne
        import numpy as np
        
        Preprocessing._add_boundary_bad_annotations_time(self)
        Preprocessing.read_concatenated_file(self)
        self.concatenated_file
        
        onset = self.mspikes_for_full_plot.begin.values/1000 + self.concatenated_file.first_samp/1000
        self.concatenated_file.annotations.append(onset, np.repeat(0.00001, len(onset)), ['A']*len(onset))

        self.concatenated_file.plot(duration=duration, group_by=group_by, start=start, lowpass=lowpass, highpass=highpass)
    
    def cleaning_manual_spikes(self):
        """ Ckecking all manual spikes, if they are correct (they should be twice and whith 'at')"""
        import re
        import pandas as pd
        
        self.mspikes_for_circus = self.mspikes.copy()
        self.final_manual_spikes = pd.DataFrame([], columns=self.mspikes_for_circus.columns.values)
        for i, row in self.mspikes_for_circus.iterrows():
            if (re.findall('at\d+', row.file) != [])&(re.findall(self.file_fif_for_circus[:-4], row.file) != []):
                if sum(self.mspikes_for_circus.begin.values == row.begin)>1:
                    self.final_manual_spikes = self.final_manual_spikes.append(pd.DataFrame({'file':row.file, 'begin':row.begin, 'end':row.end, 'r0':[row.r0], 'rd':[row.rd], 'Q':[row.Q], 'goodness':row.goodness,'errors_computed':row.errors_computed, 'noise_level':row.noise_level, 'single_errors':[row.single_errors], 'error_matrix':[row.error_matrix],'conf_vol':row.conf_vol, 'khi2':row.khi2, 'prob':row.prob, 'noise_est':row.noise_est, 'file_name':row.file_name}), ignore_index=True)
        self.final_manual_spikes.sort_values("begin", inplace=True)
        self.final_manual_spikes.reset_index(inplace=True, drop=True)
        
        first_sample = self.all_cases_files.loc[(self.all_cases_files.folder==self.dir_case)&(self.all_cases_files.file==self.file_fif_for_circus),"Time_of_the_first_sample"].reset_index(drop=True)
        self.first_sample = first_sample[0]
        if pd.notnull(self.first_sample):
            self.final_manual_spikes.loc[:,"begin"] = self.final_manual_spikes.loc[:,"begin"] + self.first_sample
        
        self.final_manual_spikes.drop_duplicates(subset='begin',inplace=True)
        self.mspikes_for_circus = self.final_manual_spikes.copy()
    
    def information_for_circus(self):
        """ Creat all necessary for Spyking Circus files """
        import os
        import mne
        import pandas as pd

        
        Preprocessing.update_cases_files(self)
        Preprocessing.cleaning_manual_spikes(self)
        self.folder_for_circus = '%s%s_%s/'%(self.dir_spc,self.case,self.file_fif_for_circus[:-4])
        
        os.makedirs('{}Aspire'.format(self.folder_for_circus),exist_ok=True)
        self.mspikes_for_circus.to_csv(path_or_buf= '{}{}_{}/Aspire/Manual_spikes_{}.csv'.format(self.dir_spc,self.case,self.file_fif_for_circus[:-4],self.file_fif_for_circus[:-4]), columns=['begin'], index=False, header=False)
        
        
        
        