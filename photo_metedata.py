# -*- coding: utf-8 -*-
"""
Created on Sat Sep 09 21:50:03 2017

@author: michael

non-walk version:
images = [f for f in os.listdir(imgDir) if
          os.path.isfile(os.path.join(imgDir, f)) and
          os.path.splitext(f)[1] == '.jpg']
"""

import os
from PIL import Image
from PIL.ExifTags import TAGS

imgDir = r'C:\Users\michael\Pictures'

for root, dirs, files in os.walk(imgDir):
    images = [f for f in files if os.path.splitext(f)[1] == '.jpg']

    for image in images:
        try:
            imgFile = Image.open(os.path.join(root, image))
            info = imgFile._getexif()

            exifData = {}

            if info:
                for tag, val in info.items():
                    decoded = TAGS.get(tag, tag)
                    exifData[decoded] = val

            if exifData:
                print '[{}] has exif data'.format(image)
                for tag, val in exifData.items():
                    if not tag == 59932:
                        print '\t {}: {}'.format(tag, val)
                print
            else:
                print '[{}] has no exif data'.format(image)
                print

        except Exception as e:
            print e
