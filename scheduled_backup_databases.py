# -*- coding: utf-8 -*-
"""
Created on Mon August 7 2017

@author: mtroyer

This tool takes the current, uncompressed gdb backup in SHORT_BACKUP, compresses it, saves it to
LONG_ARCHIVE with the same name, and then removes the uncompressed version from SHORT_ARCHIVE.
The tool then copies and datestamps the current BLM_Cultural_Resources.gdb to SHORT_ARCHIVE.
"""


import os
import shutil
import logging
import datetime
import traceback

ROOT = r'T:\CO\GIS\gistools\tools\Cultural'

SHORT_BACKUP = os.path.join(ROOT, 'z_backups')
LONG_ARCHIVE = os.path.join(ROOT, 'z_backups', 'Archived')
LOGFILE_PATH = os.path.join(ROOT, 'z_logs', 'gdb_backup.log')
CRM_GDB_PATH = os.path.join(ROOT, 'BLM_Cultural_Resources', 'BLM_Cultural_Resources.gdb')

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filename=LOGFILE_PATH)

today = datetime.date.today()
date_stamp = '{:02}{:02}{:02}'.format(today.year, today.month, today.day)[2:]

try:
    # Get last short archive, compress, and move to long archive
    last_months_backup  = [d for d in os.listdir(SHORT_BACKUP) if '.gdb' in d][-1]
    last_months_path    = os.path.join(SHORT_BACKUP, last_months_backup)
    backup_to_archive   = os.path.join(LONG_ARCHIVE,  last_months_backup)
    this_months_backup  = os.path.join(SHORT_BACKUP, date_stamp + '_BLM_Cultural_Resources.gdb')
    
    shutil.make_archive(backup_to_archive, 'zip', last_months_path)
    logging.info("Created archive: {}".format(backup_to_archive))
    shutil.rmtree(last_months_path)
    logging.info("Removed backup: {}".format(last_months_path))
    
    # Get current GDB, copy and move to short archive as yymmdd_BLM_Cultural_Resources.gdb
    shutil.copytree(CRM_GDB_PATH, this_months_backup, ignore=shutil.ignore_patterns('*.lock'))
    logging.info("Copied BLM_Cultural_Resources.gdb")
    logging.info("Completed Successfully")

except:
   logging.info("Failure:\n{}".format(traceback.format_exc()))