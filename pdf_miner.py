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

    pdf_file = open(in_pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    
    db = defaultdict(lambda: defaultdict(str))
       
    for page in range(pdf_reader.numPages):
        previous_line = None
        copy_next = False
        page_text = pdf_reader.getPage(page).extractText()
        lines = page_text.split('\n')
        for line_num, line in enumerate(lines):
            if line:
                if copy_next:
                    db[page][line_num] = line
                    copy_next = False
                    
                if re.findall(search_text, line):
                    if not db[page][line_num]:
                        db[page][line_num] = line
                          
                    if not db[page][line_num-1]:
                        db[page][line_num-1] = previous_line
                    
                    copy_next = True  
                    
                previous_line = line
                
    with open(out_csv_path, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Page", "Line", "Text"])
        
        for page, lines in sorted(db.items(), key=lambda x: int(x[0])):
            for line_num, line in sorted(lines.items(), key=lambda x: int(x[0])):
                print 'Page: {} \tLine: {}'.format(int(page)+1, int(line_num)+1)
                print
                csvwriter.writerow([page, line_num, line])
            
    return db

if __name__ == "__main__":
    in_path = r'C:\Users\michael\Documents\datasets\test_pdf.pdf'
    out_path =  r'C:\Users\michael\Documents\datasets\test_pdf.csv'
    db = search_pdf_text(in_path, "SVM", out_path)
    