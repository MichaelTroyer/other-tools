# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 07:46:07 2017

@author: mtroyer
"""


import os
import sys
import time
import shutil
import logging
import datetime


TODAY      = datetime.date.today()
MONTH      = TODAY.strftime('%B')
LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'
LOG_PATH   = r'C:\Users\mtroyer\Logs\backup_prep.log'


logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format=LOG_FORMAT)

# Print and log: print_log! 
def print_log(msg): print msg; logging.info(msg)
   
# If mdate (last modified date) is less than 'days' (in seconds), return True
def update_needed(file_name, days=30):
     return (time.time() - os.path.getmtime(file_name)) <= (days * 24 * 60 * 60)
     
# Create the inital, complete backup
def create_full_backup(src_root, src_dirs, dst_root): 
    print_log('Creating full backup')
    # Skip a dir if it already exists, return if all exist
    targets = [d for d in src_dirs if not os.path.exists(os.path.join(dst_root, d))]
    if not targets: 
        print_log('Backup folders already exist')
        return
    
    # If the root does not already exist, build it and its branch
    if not os.path.exists(dst_root): rebuild_dir(dst_root)
    
    for target_dir in targets:
        try:
            src = os.path.join(src_root, target_dir)
            dst = os.path.join(dst_root, target_dir)
            shutil.copytree(src, dst)
            print_log('Copied directory: {}'.format(target_dir))
        except:
            print_log("### Unexpected error: {}".format(sys.exc_info()[1]))


# Split 'file_path' into current directory and parent directory.
# If parent directory does not exist, recurse up through 'file_path' until
# an existing branch is found, and create remaining 'file_path' branch.
def rebuild_dir(file_path):
    try:
        parent, child = os.path.split(file_path)
        if not os.path.exists(parent):
            rebuild_dir(parent)
        if not os.path.exists(os.path.join(parent, child)):
            os.mkdir(os.path.join(parent, child))
    except:
        print_log("Path error: {}".format(file_path))
        print_log(str((sys.exc_info()[1])).format(sys.exc_info()[1]))

# Replace the source root path with the destination root path,
# build destination path and copy src -> dst
def copy_to_backup(src_path, src_root, dst_root, src_type='file'):
    try:
        dst_path = src_path.replace(src_root, dst_root)
        dst_parent, dst_file = os.path.split(dst_path)
        rebuild_dir(dst_parent)
        if src_type == 'file':
            shutil.copy(src_path, dst_path)
        if src_type == 'gdb':
            gdb_ignore = shutil.ignore_patterns('*.lock')
            shutil.copytree(src_path, dst_path, ignore=gdb_ignore)
    except:
        print_log("### Unexpected error: {}".format(sys.exc_info()[1]))
    
# Main
def backup_dir(src_root, src_dirs, dst_root):
    for src_dir in src_dirs:
        # Recurse the tree - this can take a while.
        for root, dirs, files in os.walk(os.path.join(src_root, src_dir)):
  
            print_log('Searching: {}'.format(root))
            
            # Remove unwanted
            if 'Thumbs.db' in files: 
                files.remove('Thumbs.db')
            if '.git' in dirs: 
                dirs.remove('.git')
            if '.ipynb_checkpoints' in dirs:
                dirs.remove('.ipynb_checkpoints')
            if 'Env_Data' in dirs:
                dirs.remove('Env_Data')
                
            for d in dirs:
                # copy geodatabases whole and do not recurse into them.
                if d.endswith('.gdb'):
                    src_path = os.path.join(root, d)
                    if update_needed(src_path):
                        copy_to_backup(src_path, src_root, dst_root, 'gdb')
                    dirs.remove(d)
                # Remove unwanted
                if d.endswith('Env_Data'):
                    dirs.remove(d)
            
            for filename in files:
                src_path = os.path.join(root, filename)
                if update_needed(src_path):
                    copy_to_backup(src_path, src_root, dst_root)
                        
    
if __name__=='__main__':
    
    print_log('Looking for files to update..')
        
    # My U network drive
    u_root   = r'U:\Troyer'
    u_dirs   = ['a-Archive' 'a-Projects', 'a-Program-Work']
    u_target = r'C:\Users\mtroyer\Documents\{}_U_Backup'.format(MONTH)
    backup_dir(u_root, u_dirs, u_target)
    
    # My T network drive
    t_root   = r'T:\CO\GIS\gisuser\rgfo\mtroyer'
    t_dirs   = ['a-Projects', 'a-Program-Work']
    t_target = r'C:\Users\mtroyer\Documents\{}_T_Backup'.format(MONTH)
    backup_dir(t_root, t_dirs, t_target)
    
    print_log('Complete..')

