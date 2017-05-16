# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 20:10:05 2017

@author: michael troyer

Take an input csv where record values are spread across multiple rows 
and compress to single record on primary key (pk) - assumed first value in row.
String out individual, unique non pk values if overlap.
"""

from collections import defaultdict
import csv

in_path  = r'C:\Users\mtroyer\Documents\row_compress\uncompressed.csv'
out_path = r'C:\Users\mtroyer\Documents\row_compress\compressed.csv'

# the primary data structure - a nested default dict/default dict/list
db = defaultdict(lambda: defaultdict(list))

# read the csv
with open(in_path, 'rb') as in_csv:
    csv_reader = csv.reader(in_csv)
    # secure the header row which starts at col index 1
    columns = csv_reader.next()[1:]
    for row in csv_reader:
        # pk is first value
        pk = row[0]
        # zip each col value with its header and pass to db w/ header as key
        for column, value in zip(columns, row[1:]):
            if value:
                db[pk][column].append(value)
 
# write the db content back to csv           
with open(out_path, 'wb') as out_csv:
    csv_writer = csv.writer(out_csv)
    # add an empty col since header starts at col index 1
    header = ['']
    header.extend(columns)
    csv_writer.writerow(header)
    for pk, column_dict in db.items():
        # pk is first entry
        print_row = [pk]
        # create a list in order as defined by columns sequence
        values = [column_dict[item] for item in columns]
        # set list to remove duplicates
        unique_values = [list(set(value)) for value in values]
        # compile multiple values (if multiple) as comma seperated string
        print_values = [', '.join([str(item) for item in value]) 
                           for value in unique_values]
        # add the values to the print row
        print_row.extend(print_values)
        # write it
        csv_writer.writerow(print_row)
        # profit