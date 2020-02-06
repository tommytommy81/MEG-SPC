#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:47:50 2020

@author: valery
"""
from pathlib import Path

def set_paths(fif, sens='grad'):
    from pkg_resources import resource_filename
    code_source = Path.cwd()
    case = case
    case_meg = Path(root) / case
        
    SPC_root = case_meg / 'CIRCUS'
    ### Package paths
    self.circus_clustering = Path(resource_filename('circus','clustering.py'))
    self.circus_pkg = self.circus_clustering.parent
    
    self.circus_updates =  self.code_source / 'circus_utilities' / 'spyking_circus_updates'
    
    self.fif_file = self.case_meg / 'art_corr' / fif
    self.SPC = self.SPC_root / self.fif_file.stem #folder
    self.npy_file = self.SPC / '{}_0'.format(self.fif_file.stem)
    self.npy_file = self.npy_file.with_suffix('.npy')
    self.SPC_output = self.SPC / '{}_CIRCUS_{}'.format(self.case, sens)       

    self.SPC_params = self.npy_file.parent / self.npy_file.stem #'%s%s.params'%(self.folder,self.filename[:-4])
    self.SPC_params = self.SPC_params.with_suffix('.params') 
    self.SPC_results = self.SPC_root / 'Results'
    if not self.SPC_results.is_dir():
        self.SPC_results.mkdir(exist_ok=True)    