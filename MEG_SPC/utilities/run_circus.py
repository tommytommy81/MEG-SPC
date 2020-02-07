#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setting parameters and file system for Spyking Circus

@author: vagechirkov@gmail.com
"""

class Circus:
    """ Run Spyking Circus"""

    def __init__(self, paths, main_params, sensors):
        '''
        Set the minimum startup environment for Spyking Circus

        Parameters
        ----------
        paths : dictionary
            All necessary folders and files:
                -- paths['case']
                -- paths['npy_file']
                -- paths['SPC_output']
                -- paths['SPC_params']
                -- paths['circus_updates']
                -- paths['circus_pkg']
                -- paths['SPC']
                -- paths['SPC_output']
                -- paths['circus_updates']
        main_params : dictionary
            Some parameters that are useful for setting:
                -- 'N_t'
                -- 'cut_off'
                -- 'cc_merge'
        sensors : str
            'grad' or 'mag'

        '''
        import numpy as np
        from shutil import copyfile
        ### Set paths and parameters
        self.main_params = main_params        
        self.case        = paths['case']
        self.npy_file    = paths['npy_file']
        self.sensors     = sensors
        self.output      = paths['SPC_output']
        self.path_params = paths['SPC_params']
        
        ### Update circus package
        self._update_circus_package_for_meg(paths['circus_updates'], 
                                            paths['circus_pkg'])
                
        ### Copy files
        copyfile(paths['circus_updates']/'config.params', paths['SPC_params'])
        copyfile(paths['circus_updates']/'meg_306.prb', paths['SPC']/'meg_306.prb')
        grad_idx = paths['circus_updates']/'grad_sensors_idx.npy'
        mag_idx  = paths['circus_updates']/'mag_sensors_idx.npy'
        copyfile(grad_idx, paths['SPC']/'grad_idx.npy')
        copyfile(mag_idx, paths['SPC']/'mag_idx.npy')
        
        ### Load sensors indexes
        self.main_params['grad_idx']    = str(np.load(grad_idx).tolist())
        self.main_params['mag_idx']     = str(np.load(mag_idx).tolist())
        self.main_params['stream_mode'] = 'None' #'multi-files'
        
 
    def _update_circus_package_for_meg(self, code_source, circus_pkg):
        '''
        Updates the circus package version to use additional features:
            -- SPLINE
            -- fitting process
            -- probe file
            -- param file
        Parameters
        ----------
        code_source : pathlib.PosixPath
            The place where the modified files are located
        circus_pkg : pathlib.PosixPath
            Path to the package Spyking Circus
        '''
        from shutil import copyfile
        
        dst = circus_pkg
        copyfile(code_source / 'config.params', dst / 'config.params')
        copyfile(code_source / 'meg_306.prb', dst / 'meg_306.prb')
        copyfile(code_source / 'clustering.py', dst / 'clustering.py')
        copyfile(code_source / 'fitting.py', dst / 'fitting.py')
        copyfile(code_source / 'parser.py', dst / 'shared' / 'parser.py')
        copyfile(code_source / 'algorithms.py', dst / 'shared' / 'algorithms.py')

    def set_params_spc(self, main_params, npy_file, output):
        '''
        Set parameters file for Spyking Circus

        Parameters
        ----------
        main_params : dictionary
            Some parameters that are useful for setting:
                -- 'N_t'
                -- 'cut_off'
                -- 'stream_mode'
                -- dead_channels ('grad_idx'/'mag_idx')
                -- 'cc_merge'
        npy_file : pathlib.PosixPath
            The path to the numpy data file in the CIRCUS folder
        output : pathlib.PosixPath
            The directory where the results will be saved. 
            Different for magnetometers and gradiometers
        '''
        from shutil import copyfile
        from circus.shared.parser import CircusParser

        self.params = CircusParser(npy_file, create_folders=False)
        ### data
        self.params.write('data','file_format','numpy')
        self.params.write('data','stream_mode', main_params['stream_mode'])
        self.params.write('data','mapping', str(output.parent / 'meg_306.prb'))
        self.params.write('data','output_dir',str(output))
        self.params.write('data','sampling_rate','1000')
        ### detection
        self.params.write('detection','radius', '6')
        self.params.write('detection','N_t', str(main_params['N_t']))
        #self.params.write('detection','spike_thresh', '6')
        self.params.write('detection','peaks','both')
        self.params.write('detection','alignment','False')
        self.params.write('detection','isolation','False')
        if  (self.sensors == 'mag'):
            grad = '{ 1 : %s}'%main_params['grad_idx']
            self.params.write('detection','dead_channels', grad)
        else:
            mag = '{ 1 : %s}'%main_params['mag_idx']
            self.params.write('detection','dead_channels', mag)
        ### filtering
        filt_param = '{}, {}'.format(main_params['cut_off'][0], 
                                     main_params['cut_off'][1])
        self.params.write('filtering','cut_off',filt_param)
        ### whitening
        self.params.write('whitening','safety_time','auto')
        self.params.write('whitening','max_elts','10000')
        self.params.write('whitening','nb_elts','0.1')
        self.params.write('whitening','spatial','False')
        ### clustering
        self.params.write('clustering','extraction','mean-raw')
        self.params.write('clustering','safety_space','False')
        self.params.write('clustering','safety_time','1')
        self.params.write('clustering','max_elts','10000')
        self.params.write('clustering','nb_elts','0.001')
        self.params.write('clustering','nclus_min','0.0001')
        self.params.write('clustering','smart_search','False')
        self.params.write('clustering','sim_same_elec','1')
        self.params.write('clustering','sensitivity','5')
        self.params.write('clustering','cc_merge', str(main_params['cc_merge']))
        self.params.write('clustering','dispersion','(5, 5)')
        self.params.write('clustering','noise_thr','0.9')
        #self.params.write('clustering','remove_mixture','False')
        self.params.write('clustering','cc_mixtures','0.1')
        self.params.write('clustering','make_plots','png')
        ### fitting
        self.params.write('fitting','chunk_size','60')
        self.params.write('fitting','amp_limits','(0.01,10)')
        self.params.write('fitting','amp_auto','False')
        self.params.write('fitting','collect_all','True')
        ### merging
        self.params.write('merging','cc_overlap','0.4')
        self.params.write('merging','cc_bin','200')
        
        self.params = CircusParser(npy_file, create_folders=False)
        copyfile(self.path_params, output / 'config.param')

    def _run_command_and_print_output(self, command):
        from subprocess import Popen, PIPE
        output = []
        errors = []
        #command_list = shlex.split(command, posix="win" not in sys.platform)
        #command_list, shell=False
        with Popen(command, stdout=PIPE, stderr=PIPE, shell=True) as process:
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
        
    def _out_in_file(self, file, out):
        '''
        Write log information in file
        
        '''
        with open(file, 'w') as f:
            for item in out:
                f.write(item)

    def run_circus(self, output, file_npy, n_cores=4, only_fitting=False, multi=False):
        '''
        Run Spyking Circus

        Parameters
        ----------
        output : pathlib.PosixPath
            The directory where the results will be saved. 
            Different for magnetometers and gradiometers
        npy_file : pathlib.PosixPath
            The path to the numpy data file in the CIRCUS folder
        n_cores : int, optional
            Number of processor cores. The default is 4.
        only_fitting : bool, optional
            Run only fitting step. The default is False.
        multi : bool, optional
            Save results to different files if 'stream_mode' is 'multi-files'.
            The default is False.

        '''
        if only_fitting:
            methods = 'filtering,fitting'
            cmd = 'spyking-circus %s -m %s -c %s'%(str(file_npy), methods, n_cores)
            cmd_multi = 'circus-multi %s'%(str(file_npy))
        else:
            methods = 'filtering,whitening,clustering,fitting'
            cmd = 'spyking-circus %s -m %s -c %s'%(str(file_npy), methods, n_cores)
            cmd_multi = 'circus-multi %s'%(str(file_npy))
        
        p, out, err = self._run_command_and_print_output(cmd)
        self._out_in_file(output / 'output_log.txt', out)
        if err != []:
            self._out_in_file(output / 'error_log.txt', err) 
        if multi:
            p, out_m, err_m = self._run_command_and_print_output(cmd_multi)
            if err_m != []:
                self._out_in_file(output / 'err_multi_log.txt', err_m)
   
    def results_to_excel(self, circus_params, path_save_results):
        '''
        Convert results to excel table.

        Parameters
        ----------
        circus_params : circus.shared.parser.CircusParser 
            self.params
        path_save_results : pathlib.PosixPath
            The directory where the results will be saved. 

        '''
        import pandas as pd
        from circus.shared.files import load_data
        results = load_data(circus_params, 'results')
        frames =[]
        for key in results['spiketimes'].keys():
            sp = results['spiketimes'][key]
            amp = results['amplitudes'][key][:,0]
            frames.append(pd.DataFrame(data={'Spiketimes': sp, 'Amplitudes': amp, 'Template': key}))
        templates=pd.concat(frames,ignore_index=True)
        file = path_save_results / 'Templates_{}.xlsx'.format(self.sensors)
        templates.to_excel(file, index=False)
        del templates, results