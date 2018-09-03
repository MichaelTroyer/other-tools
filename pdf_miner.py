# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 19:34:56 2017

@author: michael
"""

import csv
import PyPDF2
import re
from collections import defaultdict 


def search_pdf_text(in_pdf_path, search_text, out_csv_path):
    """Take an input pdf path, search term, and output csv path and extracts
    and searches the embedded pdf text (if any) for the search term. When a
    match is found, the line with the matching text, as well as the immediate 
    previous and following non-empty lines, are written to the csv along with
    the respective page and line numbers."""
    
    # Create a pdf reader object
    pdf_file = open(in_pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    
    # The main data structure nested defaultdict/defaultdict/string
    db = defaultdict(lambda: defaultdict(str))
    
    # for each page, collect all the text and split by newline character to
    # produce a list of individual lines. 
    for page in range(pdf_reader.numPages):
        # initialize previous line as None
        # Used to temporarily hold last line read in case following
        # line produces a match
        previous_line = None
         # initialize copy_next as False
         # used to flag the line following a match for writing to csv
        copy_next = False
        # get the text
        page_text = pdf_reader.getPage(page).extractText()
        # split to list of lines
        lines = page_text.split('\n')
        # number the lines
        for line_num, line in enumerate(lines):
            if line:
                # if previous line matched, copy and reset copy_next = false
                if copy_next:
                    db[page][line_num] = line
                    copy_next = False
                # the search
                if re.findall(search_text, line):
                    # if match - copy line. If previous line not already 
                    #copied, copy it too.
                    if not db[page][line_num]:
                        db[page][line_num] = line
                          
                    if not db[page][line_num-1]:
                        db[page][line_num-1] = previous_line
                    
                    copy_next = True  
                    
                previous_line = line
                
    # write to csv - sort by page number then line number           
    with open(out_csv_path, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Page", "Line", "Text"])
        
        for page, lines in sorted(db.items(), key=lambda x: int(x[0])):
            for line_num, line in sorted(lines.items(), key=lambda x: int(x[0])):
                try:
                    # try to write text but don't crash on something weird
                    print 'Page: {} \tLine: {}'.format(int(page)+1, int(line_num)+1)
                    print line
                    csvwriter.writerow([page, line_num, line])
                except: pass
    return db

if __name__ == "__main__":
    
    # change the paths, and the search term below
    in_path = r'C:\Users\michael\Documents\datasets\test_pdf.pdf'
    out_path =  r'C:\Users\michael\Documents\datasets\fuckin_pdfs.csv'
    search_term = "SVM"
    
    db = search_pdf_text(in_path, search_term, out_path)
    