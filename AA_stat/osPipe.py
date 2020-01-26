#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import os

"""
Created on Sun Jan 26 15:41:40 2020

@author: julia
"""
OS_PARAMS_DEF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'open_search.params')
import os

def run_os(files, msf_path, save_dir, parameters=None):
    if parameters:
        folder_name = 'custom_os'
    else:
        folder_name = 'default_os'
#    print(msf_path, parameters, *files)
    
#    subprocess.call(['java', '-jar', msf_path, parameters, *files])
    directory = os.path.dirname(files[0])
    os.makedirs(os.path.join(save_dir, folder_name), exist_ok=True)
    print(directory, files)
    for f in os.listdir(directory):
        if f.endswith('.pepXML'):
            print("========================.", f)
            os.rename(os.path.join(directory,f), os.path.join(save_dir, folder_name , f))
#            os.path.join(save_dir, folder_name, '*.pepXML'))
    return os.path.join(save_dir, folder_name)
#    l = ''.join('java -jar', msf_path, parameters, files)
    