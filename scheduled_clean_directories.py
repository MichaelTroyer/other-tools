# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 07:46:07 2017

@author: mtroyer
"""

import os
import time
import shutil
import logging
 
DELETE_AFTER  = 168 # Hours
DELETE_PATHS  = [r'C:\Users\mtroyer\Downloads',
                 r'U:\Troyer\z-GIS-Exchange',
                 r'T:\CO\GIS\gisuser\rgfo\mtroyer\z-GIS-Exchange']
DELETE_NEVER  = ['.doc', '.docx', '.xlsx', '.xlx', '.csv', '.txt']

 
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO,
                    filename=r'C:\Users\mtroyer\Logs\temp_dirs_cleanup.log')
 
def is_too_old(full_path):
    return (time.time() - os.path.getmtime(full_path) > (DELETE_AFTER * 3600))
 
def is_a_keeper(full_path):
    _, ext = os.path.splitext(full_path)
    return ext in DELETE_NEVER
 
for DELETE_PATH in DELETE_PATHS: 
    # files to evaluate
    contents = []

    # omit hidden files
    for f in os.listdir(DELETE_PATH):
        if not f.startswith('.'):
            contents.append(f)
     
    logging.info('Cleaning Directory: %s', DELETE_PATH)
     
    # evaluate all files
    for f in contents:
        fullpath = os.path.join(DELETE_PATH, f)
        if (is_too_old(fullpath) and not is_a_keeper(fullpath)):
            if os.path.isfile(fullpath):
                try:
                    os.remove(fullpath)
                    logging.info('Deleted file: %s', f)
                except:
                    logging.info('Failed to delete file: %s', f)
            elif os.path.isdir(fullpath):
                try:
                    shutil.rmtree(fullpath)
                    logging.info('Deleted directory: %s', f)
                except:
                    logging.info('Failed to delete directory: %s', f)