import matplotlib.pyplot as plt

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

class Circus:
    """ Here we run Spyking Circus, plot ROC etc."""

    def __init__(self, directory, case, folder, fif_file, mdip, cc_merge=0.9, cut_off=9, N_t=80, MAD=6.0, fname='_'):
        import os
        import traceback
        import pandas as pd
        
        self.case = case
        self.case_dir = directory
        self.sensors = ['grad','mag']
        self.folder = folder
        self.filename = '%s_0.npy'%fif_file[:-4]
        self.spike_thresh = MAD
        self.N_t = N_t
        self.cut_off = cut_off
        self.cc_merge = cc_merge
        self.mspikes = pd.DataFrame(mdip.begin.values, columns=['Time'])
        self.fname = fname
        return print('Folder: %s\nFile: %s\nNumber manual spikes: %s \nParameters: N_t = %s, Cut off = %s, Threshold = %s, cc_merge = %s\n'%(self.folder, self.filename, len(self.mspikes), self.N_t, self.cut_off, self.spike_thresh, self.cc_merge))

    def set_params_spc(self, sensor='grad'):
        """ Set parameters file for Spyking Circus"""
        import os 
        import shutil
        import pkg_resources
        from circus.shared.parser import CircusParser
        
        self.output_dir = 'cut_off_%s_spike_thresh_%s_N_t_%s_%s_(%s)'%(self.cut_off, self.spike_thresh, self.N_t, sensor, self.filename[:-4])
        os.makedirs(self.folder + self.output_dir,exist_ok=True)
        os.makedirs('%s%s/Results'%(self.folder, self.output_dir), exist_ok=True)
        self.output_dir_path = '%s%s/'%(self.folder, self.output_dir)
        self.output_dir_results_path = '%s%s/Results/'%(self.folder, self.output_dir)
        
        self.stream_mode = 'multi-files'
        self.path_params = '%s%s.params'%(self.folder,self.filename[:-4])
        config_file = os.path.abspath(pkg_resources.resource_filename('circus', 'config.params'))
        probe_file = os.path.abspath(pkg_resources.resource_filename('circus', 'meg_306.prb'))

        shutil.copyfile(config_file, self.path_params)
        shutil.copyfile(probe_file, '%smeg_306.prb'%self.folder);
        self.params = CircusParser(self.folder + self.filename, create_folders=False)

        self.params.write('data','file_format','numpy')
        self.params.write('data','stream_mode',self.stream_mode)
        self.params.write('data','mapping',self.folder + 'meg_306.prb')
        self.params.write('data','output_dir',self.folder + self.output_dir)
        self.params.write('data','sampling_rate','1000')
        #params.write('data','chunk_size','10')

        self.params.write('detection','N_t', str(self.N_t))
        self.params.write('detection','spike_thresh', str(self.spike_thresh))
        self.params.write('detection','peaks','both')
        self.params.write('detection','alignment','False')
        self.params.write('detection','isolation','False')

        if  (sensor == 'mag'):
            self.params.write('detection','dead_channels','{ 1 : [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 66, 67, 69, 70, 72, 73, 75, 76, 78, 79, 81, 82, 84, 85, 87, 88, 90, 91, 93, 94, 96, 97, 99, 100, 102, 103, 105, 106, 108, 109, 111, 112, 114, 115, 117, 118, 120, 121, 123, 124, 126, 127, 129, 130, 132, 133, 135, 136, 138, 139, 141, 142, 144, 145, 147, 148, 150, 151, 153, 154, 156, 157, 159, 160, 162, 163, 165, 166, 168, 169, 171, 172, 174, 175, 177, 178, 180, 181, 183, 184, 186, 187, 189, 190, 192, 193, 195, 196, 198, 199, 201, 202, 204, 205, 207, 208, 210, 211, 213, 214, 216, 217, 219, 220, 222, 223, 225, 226, 228, 229, 231, 232, 234, 235, 237, 238, 240, 241, 243, 244, 246, 247, 249, 250, 252, 253, 255, 256, 258, 259, 261, 262, 264, 265, 267, 268, 270, 271, 273, 274, 276, 277, 279, 280, 282, 283, 285, 286, 288, 289, 291, 292, 294, 295, 297, 298, 300, 301, 303, 304] }')
        else:
            self.params.write('detection','dead_channels','{ 1 : [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 206, 209, 212, 215, 218, 221, 224, 227, 230, 233, 236, 239, 242, 245, 248, 251, 254, 257, 260, 263, 266, 269, 272, 275, 278, 281, 284, 287, 290, 293, 296, 299, 302, 305] }')

        self.params.write('filtering','cut_off','%s, 100'%self.cut_off)

        self.params.write('whitening','safety_time','auto')
        self.params.write('whitening','max_elts','10000')
        self.params.write('whitening','nb_elts','0.1')

        self.params.write('clustering','safety_space','False')
        self.params.write('clustering','safety_time','1')
        self.params.write('clustering','max_elts','10000')
        self.params.write('clustering','nb_elts','0.1')
        self.params.write('clustering','nclus_min','0.02')
        self.params.write('clustering','smart_search','False')
        self.params.write('clustering','cc_merge', str(self.cc_merge))
        self.params.write('clustering','cc_mixtures','0.6')
        self.params.write('clustering','make_plots','png')

        self.params.write('fitting','amp_limits','(0.3,3)')
        self.params.write('fitting','amp_auto','True')
        self.params.write('fitting','collect_all','True')

        self.params.write('merging','cc_overlap','0.4')
        self.params.write('merging','cc_bin','200')
        
        self.params = CircusParser(self.folder + self.filename, create_folders=False)
        shutil.copyfile(self.path_params, '%s%s.params'%(self.output_dir_path,self.filename[:-4]))

    def run_command_and_print_output(command):
        from subprocess import Popen, PIPE, CalledProcessError, call
        output = []
        errors = []
        #command_list = shlex.split(command, posix="win" not in sys.platform)
        with Popen(command, stdout=PIPE, stderr=PIPE, shell=True) as process: #command_list, shell=False
            while True:
                output_stdout = process.stdout.readline()
                output_stderr = process.stderr.readline()
                if (not output_stdout) and (not output_stderr) and (process.poll() is not None):
                    break
                if output_stdout:
                    output.append(output_stdout.decode())
                if output_stderr:
                    errors.append(output_stderr.decode())
            rc = process.poll()
            return rc, output, errors
        
    def out_in_file(data_path, out, name='output.txt'):
        with open(data_path + name, 'w') as f:
            for item in out:
                f.write(item)

    def run_circus(self, n_cores=7, multi=False):
        """ Run Spyking Circus"""
        
        cmd = 'spyking-circus %s -m filtering,whitening,clustering,fitting -c %s'%(self.folder + self.filename, n_cores)
        cmd_multi = 'circus-multi %s'%(self.folder + self.filename)
        
        p, out, err = Circus.run_command_and_print_output(cmd)
        Circus.out_in_file(self.folder, out, 'output.txt')
        
        if err != []:
            Circus.out_in_file(self.folder, err, 'output_err.txt') 
        if multi:
            p, out_m, err_m = Circus.run_command_and_print_output(cmd_multi)
            if err_m != []:
                Circus.out_in_file(self.folder, err_m, 'output_err_multi.txt')
   
    def spc_results(self):
        """ Read results
        Not forget bad parts
        """
        import pandas as pd
        from circus.shared.files import load_data
        self.results = load_data(self.params, 'results')
        frames =[]
        for key in self.results['spiketimes'].keys():
            sp = self.results['spiketimes'][key]
            amp = self.results['amplitudes'][key][:,0]
            frames.append(pd.DataFrame(data={'Spiketimes': sp, 'Amplitudes': amp, 'Template': key}))
        self.templates=pd.concat(frames,ignore_index=True)
        #print('Adding bad annotations time')    
        #templates = add_bad_annot_time(directory, case, templates.sort_values('Spiketimes').reset_index(drop=True))
        self.templates.to_excel('%sTemplates_%s_all.xlsx'%(self.output_dir_results_path, self.case), index=False)

    def find_nearest(self, array, value):
        import numpy as np
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def nearest_mspike(self):
        self.mspikes['Temp'] = None
        self.mspikes['Temp_time'] = None
        self.mspikes['Difference'] = None
        self.mspikes['Temp_amp'] = None
        for i in range(len(self.mspikes.Time.values)):
            nearest = Circus.find_nearest(self, self.templates.Spiketimes.values, self.mspikes.Time[i])
            self.mspikes.at[i, 'Temp_time'] = self.templates.Spiketimes[self.templates.Spiketimes == nearest].values[0]
            self.mspikes.at[i, 'Temp'] = self.templates.Template[self.templates.Spiketimes == nearest].values[0]
            self.mspikes.at[i, 'Difference'] = -(self.mspikes.Time[i] - self.mspikes.at[i, 'Temp_time'])/1000
            self.mspikes.at[i, 'Temp_amp'] = self.templates.Amplitudes[self.templates.Spiketimes == nearest].values[0]
        self.mspikes.to_excel('%sMspikes_and_spyking_circus_%s.xlsx'%(self.output_dir_results_path, self.case), index=False)
   
    def ROC(self, sensor='grad', window=0.02, GoF=0.05):
        import numpy as np
        import pandas as pd
        self.roc_curve = pd.DataFrame(columns = ('Case','Sensors','Cut_off','MAD','N_t','Detected','Fitted','Best_fits','Manual_spikes','TP','FP','FN','Sensitivity','Specificity'))
        Detected = 'Not calculated'
        FP = 'Not calculated'
        Specificity = 'Not calculated'
        Circus.nearest_mspike(self)
        
        Fitted = len(self.templates)
        Best_fits = len(self.templates[np.abs(self.templates.Amplitudes-1)<GoF])
        Manual_spikes = len(self.mspikes)
        TP = len(self.mspikes[np.abs(self.mspikes.Difference) <= window])
        FN = len(self.mspikes[np.abs(self.mspikes.Difference) > window])
        if len(self.mspikes) >0:
            Sensitivity = TP/len(self.mspikes)
        else: Sensitivity = 'No information'
        self.roc_curve.loc[len(self.roc_curve)] = [self.case, sensor, self.cut_off, self.spike_thresh, self.N_t, Detected, Fitted, Best_fits, Manual_spikes, TP, FP, FN, Sensitivity, Specificity]
        self.roc_curve.to_excel('%sROC_curve_%s_%s_%s.xlsx'%(self.folder, self.case, self.fname, sensor), index=False)

    def params_iterations(self, n_cores=7, run_spc=True, ROC=False):
        from ipypb import track
        self.sensors_params = {}
        for sensor in track(self.sensors, label='Sensors '):    
            self.set_params_spc(sensor)
            if run_spc==True:
                self.run_circus(n_cores=n_cores, multi=True)
            
            self.spc_results()
            
            if ROC==True:
                self.ROC(sensor=sensor)
                    
            templates_for_aspire = self.templates.copy()
            templates_for_aspire['Template'] = templates_for_aspire['Template'].map(lambda x: int(x.split('_')[1]))
            templates_for_aspire.to_csv('%sAspire/Templates_%s_%s.csv'%(self.folder, self.filename[:-6], sensor), index=False)
            
            self.sensors_params[sensor] = self.params

            
            
class Templates:
    """ Here we read templates from the folder and create plots for them"""

    def __init__(self, directory, case, fif_file, params, sensors='grad', n_sp=100, cc_merge=0.9, cut_off=9, N_t=80, MAD=6.0):
        """ Set parameters to open
        the folder with Syking Circus results   
        """
        import os
        import traceback
        #from circus.shared.parser import CircusParser
        
        self.case = case
        self.case_dir = directory
        self.sensors = sensors
        self.n_sp = n_sp
        self.folder = '%sSpyking_circus/%s_%s/'%(self.case_dir, self.case, fif_file[:-4])
        output_dir = 'cut_off_%s_spike_thresh_%s_N_t_%s_%s_(%s)'%(cut_off, MAD, N_t, sensors, fif_file[:-4])
        self.filename = '%s_0.npy'%fif_file[:-4]
        self.params = params#CircusParser('%s/%s/%s'%(self.folder, output_dir, self.filename))
        
        self.spike_thresh = float(self.params.get('detection','spike_thresh'))
        self.N_t = self.params.getint('detection','N_t')
        self.cut_off = int(self.params.get('filtering','cut_off')[0])
        self.cc_merge = float(self.params.get('clustering','cc_merge'))
        self.output_dir = 'cut_off_%s_spike_thresh_%s_N_t_%s_%s_(%s)'%(self.cut_off, self.spike_thresh, self.N_t, self.sensors, self.filename[:-4])
        
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
        """ Here we read templates from the folder"""
        import pandas as pd
        import numpy as np
        self.templates = pd.read_excel('%s%s/Results/Templates_%s_all.xlsx'%(self.folder, self.output_dir, self.case))
        self.templates['Time'] = self.templates['Spiketimes']
        self.main_temp_names = np.unique(self.templates['Template'].value_counts()[self.templates['Template'].value_counts()>=self.n_sp].index.tolist())
        
    def all_temp_barplot(self, name=''):
        """ Here we plot barplot for all templates"""
        title = "Number of spikes for each template"
        fig = self.templates['Template'].value_counts()[self.templates['Template'].value_counts() >= self.n_sp].plot(kind="bar", title=title, figsize=(8,9))
        fig.axes.set_ylabel("Spikes")
        fig.axes.set_xlabel("Templates")
        fname = "%sall_templates_barplot_%s.png"%(self.png_path, name)
        plt.savefig(fname, dpi=200)
        plt.clf()
        plt.close()
    
    def best_spikes_temp_n(self, template):
        """ Here we selecte best spikes for one template"""
        import numpy as np
        
        self.template = template
        self.temp_n = int(self.template.split('_')[1])
        self.templates_n = self.templates[self.templates.Template == self.template]
        self.templates_n.reset_index(drop=True, inplace=True)
        self.n_spikes = len(self.templates_n)
        self.templates_n.Amplitudes = np.abs(self.templates_n.Amplitudes - 1)
        self.best_temp = self.templates_n.sort_values('Amplitudes').reset_index(drop=True).loc[0:200].copy()
        self.best_temp['Amplitudes'] = self.best_temp['Amplitudes'] + 1
    
    def create_epochs_temp_n(self, raw_filt, tmin=-1, tmax=1, picks_raw=True, loaded_data=True):
        """ Here we create epochs for template n"""
        import mne
        import numpy as np
        eve_id = self.temp_n+1
        eve_name = self.template
        
        if loaded_data != True:
            raw_filt.load_data()
        self.data_info = raw_filt.pick_types(meg=True, eeg=False,stim=False, eog=False).info
        #channel_indices_by_type = mne.io.pick.channel_indices_by_type(raw_filt.info)
        
        new_events, eve = [], []
        first_samp = raw_filt.first_samp

        for spike_time in self.best_temp['Spiketimes']: #self.templates_n['Spiketimes']:
            eve = [int(round(spike_time + first_samp)), 0, eve_id]
            new_events.append(eve)

        if 'STI_sp' not in raw_filt.info['ch_names']:
            stim_data = np.zeros((1, len(raw_filt.times)))
            info_sp = mne.create_info(['STI_sp'], raw_filt.info['sfreq'], ['stim'])
            stim_sp = mne.io.RawArray(stim_data, info_sp, verbose=False)
            raw_filt.add_channels([stim_sp], force_update_info=True)

        raw_filt.add_events(new_events, stim_channel='STI_sp', replace=True)
        self.events_temp = mne.find_events(raw_filt, stim_channel='STI_sp', verbose=False)
        event_id = {eve_name: eve_id}
        picks = mne.pick_types(raw_filt.info, meg=picks_raw, eeg=False, eog=False)
        self.epochs_temp = mne.Epochs(raw_filt, self.events_temp, event_id,  tmin, tmax, baseline=None, picks=picks, preload=True, verbose=False)
        self.data_info_sens = raw_filt.pick_types(meg=self.sensors, eeg=False,stim=False, eog=False).info
        del raw_filt, picks, event_id
    
    def largest_indices(ary, n):
        """ Returns the n largest indices from a numpy array."""
        import numpy as np
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape) # 0 - row, 1 - column

    def templates_topo_temp_n(self, save=True):
        """ Plot topography for template n"""
        """ For even N_t don't forget +1 """
        """
        This code is too slow....
        from mne.viz import iter_topography        
        
        ymin,ymax = temp_i.min(), temp_i.max()         
        for ax, idx in iter_topography(self.data_info_sens, fig_facecolor='white', axis_facecolor='white', axis_spinecolor='white', on_pick=None): #my_callback
            if idx in sensors: ax.plot(temp_i[idx],linewidth=0.5, color='red')
            else: ax.plot(temp_i[idx],linewidth=0.5, color='blue')
            ax.set_ylim([ymin+ymin/5,ymax+ymax/5])      
            
        sensors, time = Templates.largest_indices(np.abs(temp_i), 1)
        self.pick_index = sensors[0]
        self.pick = self.data_info['ch_names'].index(self.data_info_sens['ch_names'][self.pick_index])
        """
        
        from circus.shared.files import load_data
        import numpy as np
        import mne
        
        N_e = self.params.getint('data', 'N_e') # The number of channels
        N_t = self.params.getint('detection', 'N_t')+1 # The temporal width of the template #sometimes needs +1 !!!!!
        templates_cir = load_data(self.params, 'templates')

        temp_i = templates_cir[:, self.temp_n].toarray().reshape(N_e, N_t)
        self.temp_n_evocked = mne.EvokedArray(temp_i, self.data_info_sens)
        self.pick_name = self.temp_n_evocked.get_peak()[0]
        self.pick = self.data_info['ch_names'].index(self.pick_name)
        
        fig = self.temp_n_evocked.plot_topo(vline=None, legend=False, show=False)
        fig.suptitle('Template %s'%self.temp_n, x=0.42)
        if save:
            fname = "%s%s_templates_topo.svg"%(self.svg_path, self.temp_n)
            fig.savefig(fname, format='svg', papertype='legal')
        else: self.templates_topo_temp_n_fig = fig
        fig.clf()
        plt.close(fig)
        
        del templates_cir, N_e, N_t

    def power_plot_temp_n(self,save=False):
        """ Power plot for template n"""
        import numpy as np
        from mne.time_frequency import tfr_multitaper #(tfr_multitaper, tfr_stockwell, tfr_morlet, tfr_array_morlet)
        freqs = np.arange(1., 60., 1)
        n_cycles = freqs/10.
        time_bandwidth = 2.0
        power = tfr_multitaper(self.epochs_temp,freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, picks=[self.pick], return_itc=False, verbose=False)
        # Plot results. Baseline correct based on first 100 ms.
        fig = power.plot([0], baseline=(-1., -0.9), mode='mean', title='Time-Frequency Representation', show=False)
        #f = power.plot_topomap(baseline=(-1., -0.9), ch_type = 'mag')
        if save:
            fname = "%s%s_power_plot.svg"%(self.svg_path, self.temp_n)
            fig.savefig(fname, format='svg', papertype='legal')
        else: self.power_plot_temp_n_fig = fig
        #fig.clf()
        plt.close(fig)
        del power, time_bandwidth, n_cycles, freqs
        
    def _order_func(times, data):
        import numpy as np
        import random
        from sklearn.cluster.spectral import spectral_embedding  # noqa
        from sklearn.metrics.pairwise import rbf_kernel   # noqa
        this_data = data[:, (times > 0.0) & (times < 0.350)]
        this_data /= np.sqrt(np.sum(this_data ** 2, axis=1))[:, np.newaxis]
        return np.argsort(spectral_embedding(rbf_kernel(this_data, gamma=1.), n_components=1, random_state=0).ravel())

    def plot_epochs_image_temp_n(self, save=False):
        """ Epochs image for template n"""
        import mne
        #self.epochs_temp.filter(1., 45., fir_design='firwin')
        mne.viz.plot_epochs_image(self.epochs_temp, [self.pick], order=Templates._order_func, colorbar=True,  show=False)
        if save:
            fname = "%s%s_epochs_image.svg"%(self.svg_path, self.temp_n)
            plt.savefig(fname, dpi=50, format='svg', papertype='legal')
        else:
            fig = plt.gcf()
            self.plot_epochs_image_temp_n_fig = fig
        plt.close()
        
    def save_waveform_temp_n(self):
        """ Make average and save waveforms for epochs template n"""
        self.evoked_grad = self.epochs_temp.average().pick_types(meg='grad')
        self.evoked_mag = self.epochs_temp.average().pick_types(meg='mag')
        self.epochs_temp.average(picks=self.pick).save('%s%s/Templates_waveforms/%s.fif'%(self.folder, self.output_dir,self.template))
        
    def plot_evoked_joint_temp_n(self, save=False):
        """ Make evocked plot for epochs template n"""
        import mne
        import numpy as np
        mne.viz.plot_evoked_joint(self.evoked_mag, times=np.array([-0.1, -0.05, 0.0, 0.05, 0.1 ]), show=False)
        if save:
            fname = "%s%s_evoked_joint_mag.svg"%(self.svg_path, self.temp_n)
            plt.savefig(fname, format='svg', papertype='legal')
        else:
            fig = plt.gcf()
            self.plot_evoked_joint_temp_n_mag_fig = fig
        plt.close(fig)
        
        mne.viz.plot_evoked_joint(self.evoked_grad, times = np.array([-0.1, -0.05, 0.0, 0.05, 0.1 ]), show=False)
        if save:
            fname = "%s%s_evoked_joint_grad.svg"%(self.svg_path, self.temp_n)
            plt.savefig(fname, format='svg', papertype='legal')
        else:
            fig = plt.gcf()
            self.plot_evoked_joint_temp_n_grad_fig = fig
        plt.close(fig)
        
    def time_between_spikes_temp_n(self, save=False):
        """ Make time different for spikes template n"""
        from circus.shared.files import load_data
        import numpy as np
        results = load_data(self.params, 'results')
        fig, ax = plt.subplots(figsize=(4,2.9))
        spikes = results['spiketimes'][self.template]/1000
        isis = np.diff(spikes)

        plt.hist(isis, bins=100, range=(0,3))
        ax.set_ylabel("Spikes")
        ax.set_xlabel("Time, s")
        if save:
            fname = "%s%s_time_between_spikes.svg"%(self.svg_path, self.temp_n)
            plt.savefig(fname, format='svg', papertype='legal')
        else: self.time_between_spikes_temp_n_fig = fig
        plt.close(fig)

    def spikes_distribution_temp_n(self, save=False):
        """ Make distribution for spikes template n"""
        from circus.shared.files import load_data
        results = load_data(self.params, 'results')
        fig, ax = plt.subplots(figsize=(9,2.9))
        spikes = results['spiketimes'][self.template]/1000
        amps = results['amplitudes'][self.template][:, 0] # The second column are amplitude for orthogonal, not needed
        plt.plot(spikes, amps, '.')
        ax.set_ylabel("Amplitudes")
        ax.set_xlabel("Time, s")
        if save:
            fname = "%s%s_spikes_distribution.svg"%(self.svg_path, self.temp_n)
            plt.savefig(fname, format = 'svg', papertype= 'legal')
        else:
            fig = plt.gcf()
            self.spikes_distribution_temp_n_fig = fig
        plt.close(fig)
        
    def _waitForResponse(x): 
        out, err = x.communicate() 
        if x.returncode < 0: 
            r = "Popen returncode: "+str(x.returncode) 
            raise OSError(r)

    def plot_final_temp_n(self, system_type='Windows'):
        """ Plot all plots in one"""
        import svgutils.transform as sg
        from subprocess import Popen
        fig = sg.SVGFigure("24cm", "25cm")
        #plot1 = sg.fromfile("%s%s_epochs_image.svg"%(self.svg_path, self.temp_n)).getroot()
        plot1 = sg.from_mpl(self.plot_epochs_image_temp_n_fig).getroot()
        plot1.moveto(10, 10)
        #plot2 = sg.fromfile("%s%s_evoked_joint_mag.svg"%(self.svg_path, self.temp_n)).getroot()
        plot2 = sg.from_mpl(self.plot_evoked_joint_temp_n_mag_fig).getroot()
        plot2.moveto(20, 285)
        plot2.scale_xy(0.7, 0.7)
        #plot3 = sg.fromfile("%s%s_evoked_joint_grad.svg"%(self.svg_path, self.temp_n)).getroot()
        plot3 = sg.from_mpl(self.plot_evoked_joint_temp_n_grad_fig).getroot()
        plot3.moveto(20, 500)
        plot3.scale_xy(0.7, 0.7)
        plot4 = sg.fromfile("%s%s_templates_topo.svg"%(self.svg_path, self.temp_n)).getroot()
        #plot4 = sg.from_mpl(self.templates_topo_temp_n_fig).getroot()
        plot4.moveto(440, 20)
        plot4.scale_xy(1.2, 1.1)
        plot5 = sg.fromfile("%s%s_power_plot.svg"%(self.svg_path, self.temp_n)).getroot()
        #plot5 = sg.from_mpl(self.power_plot_temp_n_fig).getroot()
        plot5.moveto(440, 340)
        plot5.scale_xy(1.2, 1.3)
        #plot6 = sg.fromfile("%s%s_spikes_distribution.svg"%(self.svg_path, self.temp_n)).getroot()
        plot6 = sg.from_mpl(self.spikes_distribution_temp_n_fig).getroot()
        plot6.moveto(290, 715)
        #plot6.scale_xy(1.5, 1.)
        #plot7 = sg.fromfile("%s%s_time_between_spikes.svg"%(self.svg_path, self.temp_n)).getroot()
        plot7 = sg.from_mpl(self.time_between_spikes_temp_n_fig).getroot()
        plot7.moveto(20, 715)
        #plot7.scale_xy(1.5, 1.)

        params_plot_fin = ['channels = %s'%self.sensors, 'filtering = %s-100 Hz'%self.cut_off, 
                           'N_t = %s ms'%self.N_t, 'spike_thresh = %s'%self.spike_thresh,'isolation = False', 
                           'cc_merge = %s'%self.cc_merge, 'cc_overlap = 0.4', 'amp_limits = auto','safety_time = 1ms', 'cc_mixtures = 0.6']
        l = []
        for i in range(len(params_plot_fin)):
            l.append(sg.TextElement(815,27 + i*7, params_plot_fin[i], size=5))

        l.append(sg.TextElement(20,20, "A", size=12, weight="bold"))
        l.append(sg.TextElement(20,300, "B", size=12, weight="bold"))
        l.append(sg.TextElement(20,530, "C", size=12, weight="bold"))
        l.append(sg.TextElement(440,20, "D", size=12, weight="bold"))
        l.append(sg.TextElement(440,350, "E", size=12, weight="bold"))
        l.append(sg.TextElement(800,15, "N events: %s" %self.n_spikes, size=10) )
        l.append(sg.TextElement(20,730, "F", size=12, weight="bold"))
        l.append(sg.TextElement(310,730, "G", size=12, weight="bold"))

        fig.append([plot1, plot2, plot3, plot4, plot5, plot6, plot7])
        fig.append(l)
        fig.save("%s%s_fig_final.svg"%(self.svg_path, self.temp_n))

        your_svg_input = "%s%s_fig_final.svg"%(self.svg_path, self.temp_n)
        your_png_output = "%s%s_temp.png"%(self.png_path, self.temp_n)

        if system_type == 'Mac':
            x = Popen(['/Applications/Inkscape.app/Contents/Resources/bin/inkscape', your_svg_input, \
                       '--export-png=%s' % your_png_output, '-w2000 -h3000', '-b white'])
        elif system_type == 'Windows':
            x = Popen(['C:/Program Files/Inkscape/inkscape', your_svg_input, \
                       '--export-png=%s' % your_png_output, '-w2000 -h3000', '-b white'])
        try:
            Templates._waitForResponse(x)
        except OSError:
            return False
    
    def plot_all_temp_n(self, data, tmp_name, system_type='Windows'):
        """ Plot all plots for template n"""
        import traceback
        #with NoStdStreams():
        try:
            self.read_templates()
            self.all_temp_barplot()
            self.best_spikes_temp_n(tmp_name)
            self.create_epochs_temp_n(data.copy())
            self.templates_topo_temp_n()
            with NoStdStreams():
                self.power_plot_temp_n(save=True)
            self.plot_epochs_image_temp_n()
            self.save_waveform_temp_n()
            self.plot_evoked_joint_temp_n()
            self.spikes_distribution_temp_n()
            self.time_between_spikes_temp_n()
            self.plot_final_temp_n(system_type)
        except Exception: traceback.print_exc()

    def plot_all_templates(self, tsss_file):
        """ Plot all templates """
        from ipypb import track
        import mne
        
        data = mne.io.read_raw_fif(tsss_file, preload=True, verbose=False)
        for t_name in track(self.main_temp_names, label='Templates '):
            self.plot_all_temp_n(data, t_name, 'Windows')
        
        del data

            
            