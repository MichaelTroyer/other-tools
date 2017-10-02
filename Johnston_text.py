#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#
# FRONT MATTER ----------------------------------------------------------------
#
###############################################################################

"""
Author:
    Michael D. Troyer

Date:

Purpose:

Comments:

TODO:

    remove all formatting
    replace check marks with lower(x)
    export to ascii

"""

###############################################################################
#
# IMPORTS ---------------------------------------------------------------------
#
###############################################################################

from __future__ import division  # Integer division is lame - use // instead
from collections import defaultdict
import csv
import datetime
import getpass
import os
import re
import sys
import textwrap
import traceback

###############################################################################
#
# GLOBALS ---------------------------------------------------------------------
#
###############################################################################

###############################################################################
# Variables
###############################################################################

filename = os.path.basename(__file__)

start_time = datetime.datetime.now()

user = getpass.getuser()

working_dir = r'C:\Users\mtroyer\Documents\Johnston_Text'

###############################################################################
# Classes
###############################################################################

class py_log(object):
    """A custom logging class that simultaneously writes to the console,
       an optional logfile, and/or a production report. The methods provide
       three means of observing the tool behavior 1.) console progress updates
       during execution, 2.) tool metadata regarding date/user/inputs/outputs..
       and 3.) an optional logfile where the tool will print messages and
       unpack variables for further inspection"""

    def __init__(self, report_path, log_path, log_active=True, rep_active=True):
        self.report_path = report_path
        self.log_path = log_path
        self.log_active = log_active
        self.rep_active = rep_active

    def _write_arg(self, arg, path, starting_level=0):
        """Accepts a [path] txt from open(path)
           and unpacks that data like a baller!"""
        level = starting_level
        txtfile = open(path, 'a')
        if level == 0:
            txtfile.write(header)
        if type(arg) == dict:
            txtfile.write("\n"+(level*"\t")+(str(arg))+"\n")
            txtfile.write((level*"\t")+str(type(arg))+"\n")
            for k, v in arg.items():
                txtfile = open(path, 'a')
                txtfile.write('\n'+(level*"\t\t")+(str(k))+": "+(str(v))+"\n")
                if hasattr(v, '__iter__'):
                    txtfile.write((level*"\t\t")+"Values:"+"\n")
                    txtfile.close()
                    for val in v:
                        self._write_arg(val, path, starting_level=level+2)
        else:
            txtfile.write("\n"+(level*"\t")+(str(arg))+"\n")
            txtfile.write((level*"\t")+str(type(arg))+"\n")
            if hasattr(arg, '__iter__'):  # Does not include strings
                txtfile.write((level*"\t")+"Iterables:"+"\n")
                txtfile.close()
                for a in arg:
                    self._write_arg(a, path, starting_level=level+1)
        txtfile.close()

    def _writer(self, msg, path, *args):
        """A writer to write the msg, and unpacked variable"""
        with open(path, 'a') as txtfile:
            txtfile.write(msg+"\n")
            txtfile.close()
            if args:
                for arg in args:
                    self._write_arg(arg, path)

    def console(self, msg):
        """Print to console only - progress reports"""
        print(msg)  # Optionally - arcpy.AddMessage()

    def report(self, msg):
        """Write to report only - tool process metadata for the user"""
        if self.rep_active:
            path_rep = self.report_path
            self._writer(msg, path_rep)

    def logfile(self, msg, *args):
        """Write to logfile only - use for reporting debugging data
           With an optional shut-off"""
        if self.log_active:
            path_log = self.log_path
            self._writer(msg, path_log, *args)

    def logging(self, log_level, msg, *args):
        assert log_level in [1,2,3], "Incorrect log level"
        if log_level == 1: # Updates - Console, report, and logfile:
            self.console(msg)
            self.report(msg)
            self.logfile(msg, *args)
        if log_level == 2:  # Operational metadata - report and logfile
            self.report(msg)
            self.logfile(msg, *args)
        if log_level == 3:  # Debugging - logfile only
            self.logfile(msg, *args)

###############################################################################
# Functions
###############################################################################

def print_exception_full_stack(print_locals=True):
    """Print full stack in a more orderly way
       Optionally print the exception frame local variables"""
    exc = sys.exc_info()  # 3-tuple (type, value, traceback)
    if exc is None:
        return None

    tb_type, tb_value, tb_obj = exc[0], exc[1], exc[2]
    exc_type = str(tb_type).split(".")[1].replace("'>", '')
    lg.logging(1, '\n\n'+header+'\n'+header)
    lg.logging(1,'\nEXCEPTION:\n{}\n{}\n'.format(exc_type, tb_value))
    lg.logging(1, header+'\n'+header+'\n\n')
    lg.logging(1, 'Traceback (most recent call last):')

    # 4-tuple (filename, line no, func name, text)
    tb = traceback.extract_tb(exc[2])
    for tb_ in tb:
        lg.logging(1, "{}\n"
                   "Filename: {}\n"
                   "Line Number: {}\n"
                   "Function Name: {}\n"
                   "Text: {}\n"
                   "Exception: {}"
                   "".format(header, tb_[0], tb_[1], tb_[2],
                             textwrap.fill(tb_[3]), exc[1]))
    if print_locals:
        stack = []
        while tb_obj.tb_next:
            tb_obj = tb_obj.tb_next  # Make sure at end of stack
        f = tb_obj.tb_frame          # Get the frame object(s)

        while f:                     # Append and rewind, reverse order
            stack.append(f)
            f = f.f_back
        stack.reverse()

        lg.logging(3, '\n\nFrames and locals (innermost last):\n'+header)
        for frame in stack:
            if str(frame.f_code.co_filename).endswith(filename):
                lg.logging(3, "{}\n"
                           "FRAME {} IN:\n"
                           "{}\n"
                           "LINE: {}\n"
                           "".format(header,
                                     textwrap.fill(frame.f_code.co_name),
                                     textwrap.fill(frame.f_code.co_filename),
                                     frame.f_lineno))

                if not frame.f_locals.items():
                    lg.logging(3, "No locals\n")

                else:
                    lg.logging(3, "{} LOCALS:\n".format(frame.f_code.co_name))
                    for key, value in sorted(frame.f_locals.items()):
                        # Exclude private and the i/o and header parameters
                        if not str(key).startswith("_"):
                            if not str(key) in ['In', 'Out', 'header']:
                                lg.logging(3, (str(key)+":").strip())

                                try:
                                    lg.logging(3, str(value).strip()+'\n')
                                except:
                                    lg.logging(3, 'Error writing value')
    return

###############################################################################
# Settings
###############################################################################

# Text header settings for all the various print functions
header = ('='*100)

###############################################################################
#
# EXECUTION -------------------------------------------------------------------
#
###############################################################################

try:
    # Make a directory - or use woring_dir from above
    path = working_dir if working_dir else os.path.join(os.getcwd(), 'z_Test')
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    folder_path = os.path.join(dir_name, base_name)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    date_split = str(datetime.datetime.now()).split('.')[0]
    date_time_stamp = re.sub('[^0-9]', '', date_split)
    name_stamp = filename.split('.')[0]+"_"+date_time_stamp

    # Create the logger
    report_path = os.path.join(folder_path, name_stamp+"_Report.txt")
    logfile_path = os.path.join(folder_path, name_stamp+"_Logfile.txt")
    lg = py_log(report_path, logfile_path)

###
#    lg.log_active = False # Uncomment to disable logfile
    lg.rep_active = False # Uncomment to disable report
###

    # Start logging
    lg.logging(1, "\nExecuting: "+filename+' \nDate: '+date_split)
    lg.logging(2, header)
    lg.logging(2, "Running environment: Python - {}".format(sys.version))
    lg.logging(1, "User: "+user)
    if lg.log_active:
        lg.logging(1, "\nLogging to: "+folder_path)

###############################################################################
#
# MAIN PROGRAM ----------------------------------------------------------------
#
###############################################################################

    # Change this to actual full file path
    textfile = r'C:\Users\mtroyer\Documents\Johnston_Text\active_permits.txt'

    # Define the data dictionaries that will build the related tables
    company_data = defaultdict(lambda:   {'Permit_Notes'    : [],
                                          'Other_Notes'     : [],
                                          'Date'            : []})

    permit_data = defaultdict(lambda:    {'Company'         : [],
                                          'Type'            : [],
                                          'Reported'        : []})

    personnel_data = defaultdict(lambda: {'Company'         : [],
                                          'Vita_Date'       : [],
                                          'Updates_Needed'  : []})

    # These are boilerplate and do not contain useful info - skip em!
    omit_lines = ['Archaeology and Paleontology Permit Status',
                  'Received',
                  'Office of the State Archaeologist of Colorado',
                  'Permit number(s) Permit Type End-of-Year Report',
                  'Permit number(s) Permit Type End-of-Year Report',
                  'Update Due For 2016',
                  'Last date on vita and/or affidavit form for:',
                  'Last date of curriculum vita and/or affidavit form for:',
                  'Assistant State Archaeologist Date']

    # prep the lines for the search
    def prep_lines(line):
        line = re.sub("['?’']+", '', line)     # remove weird characters
        line = line.strip()                    # remove leading/trailing spaces
        return line

    # clean the lines after the search
    def finish_lines(line):
        line = re.sub("['\s\s']+", ' ', line)  # compress 2+ spaces to 1
        line = re.sub("['\t']+", '', line)     # remove tabs
        line = re.sub("['\n']+", '', line)     # remove newline characters
        line = re.sub("['__']+", '', line)     # remove 2+ underscores
        line = line.strip()                    # remove leading/trailing spaces
        return line

    # open the file and start reading line-by-line
    # everything is roughly organized in order of appearance in txt

    with open(textfile, 'rb') as txt:
        for line in txt.readlines():
            line = prep_lines(line)
            # filter out the omit lines
            if line and (finish_lines(line) not in omit_lines
                         and not line.startswith('( )')):
                print line
                # find the company name
                if line.startswith('Permit(s) issued to:'):
                    holder = finish_lines(line.split(':')[1])
                    company_data[holder]
                    continue
                # get the permit data
                if re.match('^\d\d\d\d-', line):
                    matches = [line_ for line_ in line.split('\t') if line_]
                    permit_ = finish_lines(matches[0])
                    permit_type_ = finish_lines(matches[1])
                    reported_ = finish_lines(matches[2])
                    permit_data[permit_]['Company'].append(finish_lines(holder))
                    permit_data[permit_]['Type'].append(finish_lines(permit_type_))
                    permit_data[permit_]['Reported'].append(finish_lines(reported_))
                    continue
                # get the permit comments
                if line.startswith('(x)'):
                    permit_notes = finish_lines(line).replace('(x) ', '')
                    company_data[holder]['Permit_Notes'].append(permit_notes)
                    continue
                # get the personnel data
                try:
                    re_no = "[a-zA-Z'-]+[\t\s]+[a-zA-Z'-]+[\t\s]+"\
                            "[0-9]+[\/-][0-9]+[\t\s]+[NnOo]+"

                    re_yes = "[a-zA-Z'-]+[\t\s]+[a-zA-Z'-]+[\t\s]+"\
                             "[0-9]+[\/-][0-9]+[\t\s]+"\
                             "[YyEeSs,]+[\t\s]+[a-zA-Z'-]+"

                    mo_y = re.findall(re_yes, finish_lines(line))
                    mo_n = re.findall(re_no, finish_lines(line))
                    if mo_y:
                        for mo in mo_y:
                            words = re.findall("[a-zA-Z0-9'\/-]+", mo)
                            name_ = words[0]+' '+words[1]
                            cv_date = words[2]
                            updates = words[3]+' '+words[4].title()
                            personnel_data[name_]['Company'].append(holder)
                            personnel_data[name_]['Vita_Date'].append(cv_date)
                            personnel_data[name_]['Updates_Needed'].append(updates)
                    if mo_n:
                        for mo in mo_n:
                            words = re.findall("[a-zA-Z0-9'\/-]+", mo)
                            name_ = words[0]+' '+words[1]
                            cv_date = words[2]
                            updates = words[3]
                            personnel_data[name_]['Company'].append(holder)
                            personnel_data[name_]['Vita_Date'].append(cv_date)
                            personnel_data[name_]['Updates_Needed'].append(updates)
                        continue
                except: pass
                # get the date - if not a date then must be 'Other_Notes'
                try:
                    date_line = line.replace(',', '')
                    date_ = datetime.datetime.strptime(finish_lines(date_line),
                                                       '%B %d %Y').date()
                    company_data[holder]['Date'].append(finish_lines(date_line))
                    continue
                except:
                    company_data[holder]['Other_Notes'].append(finish_lines(line))

    lg.logging(2, 'Company Data', company_data)
    lg.logging(2, 'Permit Data', permit_data)
    lg.logging(2, 'Presonnel Data', personnel_data)

#    print
#    print ' Company data '.center(80, '=')
#    print (' n = {} '.format(len(company_data.items()))).center(80, '=')
#    for k, v in company_data.items():
#        print k, v
#        print
#    print
#    print ' Permit data '.center(80, '=')
#    print (' n = {} '.format(len(permit_data.items()))).center(80, '=')
#    for k, v in permit_data.items():
#        print k, v
#        print
#    print ' Personnel data '.center(80, '=')
#    print (' n = {} '.format(len(personnel_data.items()))).center(80, '=')
#    for k, v in sorted(personnel_data.items(), key=(lambda v: v[1])):
#        print k, v

    permits_csv = os.path.join(working_dir, r'permits.csv')

    with open(permits_csv, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["DATE", "COMPANY", "REPORTED", "TYPE"])
        for permit in permit_data.items():
            csvwriter.writerow([permit[0], 
                                permit[1]['Company'], 
                                permit[1]['Reported'], 
                                permit[1]['Type']])

    person_csv = os.path.join(working_dir, r'personnel.csv')
    
    with open(person_csv, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["NAME", "COMPANY", "UPDATES_NEEDED", "VITA_DATE"])
        for person in personnel_data.items():
            csvwriter.writerow([person[0], 
                                person[1]['Company'], 
                                person[1]['Updates_Needed'],
                                person[1]['Vita_Date']])

    company_csv = os.path.join(working_dir, r'companies.csv')
    
    with open(company_csv, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["NAME", "DATE", "OTHER NOTES", "PERMIT NOTES"])
        for company in company_data.items():
            csvwriter.writerow([company[0], 
                                company[1]['Date'],
                                company[1]['Other_Notes'], 
                                company[1]['Permit_Notes']])
        
    lg.logging(1, '\n\n{} Sucessfully completed'.format(filename))
    
###############################################################################
#
# EXCEPTIONS ------------------------------------------------------------------
#
###############################################################################

except:
    print_exception_full_stack(print_locals=True)  # Or print_locals=False

    # Don't create exceptions in the except block!
    try:
        lg.logging(1, '\n\n{} did not sucessfully complete'.format(filename))
        lg.console('See logfile for details')

    except:
        pass

###############################################################################
#
# CLEAN-UP --------------------------------------------------------------------
#
###############################################################################

finally:
    end_time = datetime.datetime.now()

    try:
        lg.logging(1, "End Time: "+str(end_time))
        lg.logging(1, "Time Elapsed: {}".format(str(end_time - start_time)))

    except:
        pass

###############################################################################
# if this .py has been called by interpreter directly and not by another module
# __name__ == "__main__":  #will be True, else name of importing module
#if __name__ == "__main__":
