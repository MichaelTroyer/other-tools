#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Fri Feb 10 10:34:45 2017

@author: mtroyer

TODO: finish multiprocessor - can't pickle - need func wrapper
"""

def job_queue(toolbox, csv_path, start=0, stop=None, multiprocess=False,):
    """Read a csv for a list of job parameters, convert to dict, 
       and hand off to a python toolbox with optional multiprocessing"""

    import os
    import csv
#    import arcpy
    import multiprocessing

    # Make sure everything makes sense
    assert os.path.exists(toolbox), "Toolbox not found"
    assert os.path.exists(csv_path), "CSV file not found"
    assert type(start) == int, "Invalid start row (must be int)"
    if stop:
        assert type(stop) == int, "Invalid stop row (must be int)"
        assert stop > start, "Invalid start-stop sequence"
    assert type(multiprocess)  == bool, "Invalid multiprocessing option (must be boolean)"
    
    # Import the toolbox and get the alias
#    toolbox = arcpy.ImportToolbox(toolbox, "Toolbox")
#    alias = 
    
    # Open and read csv within start and stop - get header and zip row 
    # (tuple) values with header  and convert to dict as {header: value}
    # get a list of dicts representing the parameter inputs, i.e. jobs
    csv_reader = csv.reader(open(csv_path, 'r'))
    header = csv_reader.next()
    params = [zip(header, row) for row in csv_reader][start:stop]
    assert params, "Row sequence did not return any records"
    param_dicts = [dict(params[i]) for i, _ in enumerate(params)]
               
    if multiprocess:
        # If multiprocess map function to list of dicts
        pool = multiprocessing.Pool(processes=4)
        # Spread the work
        pool.map(toolbox.alias, param_dicts)
        # Sync for clean-up
        pool.close()
        pool.join()

    else:
        # Run consecutively
        for param_dict in param_dicts:
            try:
                toolbox.alias(**param_dict)
            except:               
                print "\n\nFAILURE {}\n\n{}".format(('#'*40), str(param_dict))
#                arcpy.AddMessage("\n\nFAILURE {}\n\n{}".format(('#'*40), str(param_dict)))
                continue
            
    return params, toolbox
    
if __name__ == '__main__':
    jobs = job_queue(toolbox=r'T:\CO\GIS\giswork\rgfo\projects\management_plans'\
                             r'\ECRMP\Draft_RMP_EIS\1_Analysis\ECRMP_Working'\
                             r'\_Development\Use_Restrictions.pyt',
                     csv_path=r'T:\CO\GIS\giswork\rgfo\projects\management_plans'\
                              r'\ECRMP\Draft_RMP_EIS\1_Analysis\ECRMP_Working'\
                              r'\_Development\Use_Restrictions_Params.csv',
                     start=0,
                     stop=None,
                     multiprocess=False)