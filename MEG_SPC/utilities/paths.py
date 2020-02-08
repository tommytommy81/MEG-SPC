#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:47:50 2020

@author: valery
"""
from pathlib import Path

def set_paths(root, case, run, sens='grad'):
    '''
    Prepare folders for Spyking Circus

    Parameters
    ----------
    root : str
        Full path to the case
    case : str
        Patient ID
    run : str
        Recording designation which will be analyzed
    sens : str, optional
        'grad' or 'mag'. The default is 'grad'.

    '''
    from pkg_resources import resource_filename
    ### Main
    paths = {}
    paths['code_source'] = Path.cwd()
    paths['case_meg']    = Path(root)
    paths['case']        = case
    paths['SPC_root']    = paths['case_meg'] / 'CIRCUS'
    if not paths['SPC_root'].is_dir():
        paths['SPC_root'].mkdir(exist_ok=True)
    
    ### Package paths
    paths['circus_pkg'] = Path(resource_filename('circus','clustering.py')).parent
    paths['circus_updates'] =  paths['code_source'] / 'spyking_circus_updates'
    
    ### Files
    paths['art_cor']         = paths['case_meg']/'MEG_data'/'tsss_mc_artefact_correction'
    if sorted(paths['art_cor'].glob('*{}*'.format(run))) != []:
        paths['fif_file']    = sorted(paths['art_cor'].glob('*{}*'.format(run)))[0]
        paths['SPC']         = paths['SPC_root'] / paths['fif_file'].stem #folder
        paths['npy_file']    = paths['SPC']/'{}.npy'.format(paths['fif_file'].stem)
        paths['SPC_output']  = paths['SPC']/'{}_CIRCUS_{}'.format(paths['case'], sens)       
        paths['SPC_params']  = paths['npy_file'].with_suffix('.params') 
        paths['SPC_results'] = paths['SPC_root'] / 'Results'
        if not paths['SPC'].is_dir():
             paths['SPC'].mkdir(exist_ok=True) 
        if not paths['SPC_output'].is_dir():
            paths['SPC_output'].mkdir(exist_ok=True)        
        if not paths['SPC_results'].is_dir():
            paths['SPC_results'].mkdir(exist_ok=True)  
    else: print("Error. No fif file")
    
    ### Folders for plots 
    paths['Templates']     = paths['SPC'] / 'Templates_{}'.format(sens)
    paths['temporal_svg']  = paths['SPC'] / 'temporal_svg'
    if not paths['Templates'].is_dir():
         paths['Templates'].mkdir(exist_ok=True)
    if not paths['temporal_svg'].is_dir():
         paths['temporal_svg'].mkdir(exist_ok=True) 
        
    return paths

def fif_to_npy(fif, npy):
    '''
    Convert and save fif-file to numpy file

    Parameters
    ----------
    fif : pathlib.PosixPath
        The path to fif-file
    npy : pathlib.PosixPath
        The path to numpy file

    '''
    import mne
    import numpy as np
    data = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
    data.pick_types(meg=True, eeg=False, stim=False, eog=False, ecg=False)
    np.save(npy, (data.get_data(reject_by_annotation='omit')).astype(float))
    del data