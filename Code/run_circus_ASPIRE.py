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

    def __init__(self, directory, case, folder, fif_file, cc_merge=0.9, cut_off=9, N_t=80, MAD=6.0, fname='_'):
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
        self.fname = fname
        return print('Folder: %s\nFile: %s\nParameters: N_t = %s, Cut off = %s, Threshold = %s, cc_merge = %s\n'%(self.folder, self.filename, self.N_t, self.cut_off, self.spike_thresh, self.cc_merge))

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
        
        self.stream_mode = 'None' #'multi-files'
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

        self.params.write('detection','radius', '6') # Radius [in um] (if auto, read from the prb file)
        self.params.write('detection','N_t', str(self.N_t))
        self.params.write('detection','spike_thresh', str(self.spike_thresh))
        self.params.write('detection','peaks','both')
        self.params.write('detection','alignment','False')
        self.params.write('detection','isolation','False')

        if  (sensor == 'mag'):
            self.params.write('detection','dead_channels','{ 1 : [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 66, 67, 69, 70, 72, 73, 75, 76, 78, 79, 81, 82, 84, 85, 87, 88, 90, 91, 93, 94, 96, 97, 99, 100, 102, 103, 105, 106, 108, 109, 111, 112, 114, 115, 117, 118, 120, 121, 123, 124, 126, 127, 129, 130, 132, 133, 135, 136, 138, 139, 141, 142, 144, 145, 147, 148, 150, 151, 153, 154, 156, 157, 159, 160, 162, 163, 165, 166, 168, 169, 171, 172, 174, 175, 177, 178, 180, 181, 183, 184, 186, 187, 189, 190, 192, 193, 195, 196, 198, 199, 201, 202, 204, 205, 207, 208, 210, 211, 213, 214, 216, 217, 219, 220, 222, 223, 225, 226, 228, 229, 231, 232, 234, 235, 237, 238, 240, 241, 243, 244, 246, 247, 249, 250, 252, 253, 255, 256, 258, 259, 261, 262, 264, 265, 267, 268, 270, 271, 273, 274, 276, 277, 279, 280, 282, 283, 285, 286, 288, 289, 291, 292, 294, 295, 297, 298, 300, 301, 303, 304] }')
        else:
            self.params.write('detection','dead_channels','{ 1 : [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 206, 209, 212, 215, 218, 221, 224, 227, 230, 233, 236, 239, 242, 245, 248, 251, 254, 257, 260, 263, 266, 269, 272, 275, 278, 281, 284, 287, 290, 293, 296, 299, 302, 305] }')

        self.params.write('filtering','cut_off','%s, 200'%self.cut_off)

        self.params.write('whitening','safety_time','auto')
        self.params.write('whitening','max_elts','10000')
        self.params.write('whitening','nb_elts','0.1')

        self.params.write('clustering','safety_space','False')
        self.params.write('clustering','safety_time','1')
        self.params.write('clustering','max_elts','10000')
        self.params.write('clustering','nb_elts','0.001')
        self.params.write('clustering','nclus_min','0.0001')
        self.params.write('clustering','smart_search','False')
        self.params.write('clustering','sim_same_elec','1')
        self.params.write('clustering','sensitivity','2')
        self.params.write('clustering','cc_merge', str(self.cc_merge))
        self.params.write('clustering','dispersion','(1, 9)')
        self.params.write('clustering','cc_mixtures','0.6')
        self.params.write('clustering','make_plots','png')
    
        self.params.write('fitting','chunk_size','1')
        self.params.write('fitting','amp_limits','(0.1,8)')
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

    def run_circus(self, n_cores=7, multi=False, only_fitting=False):
        """ Run Spyking Circus"""
        
        if only_fitting:
            cmd = 'spyking-circus %s -m filtering,fitting -c %s'%(self.folder + self.filename, n_cores)
            cmd_multi = 'circus-multi %s'%(self.folder + self.filename)
        else:
            cmd = 'spyking-circus %s -m filtering,whitening,clustering,fitting -c %s'%(self.folder + self.filename, n_cores)
            cmd_multi = 'circus-multi %s'%(self.folder + self.filename)
        
        p, out, err = Circus.run_command_and_print_output(cmd)
        if only_fitting:
            Circus.out_in_file(self.folder, out, 'output_fitting.txt')
        else:
            Circus.out_in_file(self.folder, out, 'output_filtering_whitening_clustering.txt')
        
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

    def params_iterations(self, n_cores=7, run_spc=True, only_fitting = False, sensors=['grad','mag']):
        import os 
        from ipypb import track
        self.sensors_params = {}
        #sensors = self.sensors
        for sensor in track(sensors, label='Sensors '):    
            self.set_params_spc(sensor)
            if run_spc==True:
                self.run_circus(n_cores=n_cores, multi=True, only_fitting=only_fitting)
            
            self.spc_results()
                    
            templates_for_aspire = self.templates.copy()
            templates_for_aspire['Template'] = templates_for_aspire['Template'].map(lambda x: int(x.split('_')[1]))
            os.makedirs(self.folder + 'Aspire/',exist_ok=True)
            templates_for_aspire.to_csv('%sAspire/Templates_%s_%s.csv'%(self.folder, self.filename[:-6], sensor), index=False)
            
            self.sensors_params[sensor] = self.params
           