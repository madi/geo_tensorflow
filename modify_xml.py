# coding: utf-8

__author__ = "Margherita Di Leo"
__license__ = "GPL v.3"
__version__ = "0.1"
__email__ = "dileomargherita@gmail.com"

import xml.etree.ElementTree as ET
import os

#FILENAME = ".."

# Source folder
FILEPATH = '/run/media/madi/TOSHIBA EXT/DMC/jpg_subset/'
# Destination folder
DESTPATH = '/run/media/madi/TOSHIBA EXT/DMC/jpg_coarser/'
# Original resolution
RES_ORIGINAL = 0.15
# Current resolution
RES_CURRENT  = 0.60

#-----------------------------------------------------------------------

def CreateFileList(FILEPATH):
    for filename in os.listdir(str(FILEPATH)):
        if filename.endswith(".xml") and "aux" not in filename:
            fileList.append(filename)
    fileList.sort()
    print fileList
    return fileList
    
#-----------------------------------------------------------------------

def updateXml(FILENAME, FILEPATH, DESTPATH):
    tree = ET.parse(FILEPATH + FILENAME)
    root = tree.getroot()

    # Update size of current image
    orig_img_width = int(root.find("./size/width").text)
    curr_img_width = int(round(orig_img_width / factor))
    root.find("./size/width").text = str(curr_img_width)
    root.find("./size/width").set('updated', 'yes')

    orig_img_height = int(root.find("./size/height").text)
    curr_img_height = int(round(orig_img_height / factor))
    root.find("./size/height").text = str(curr_img_height)
    root.find("./size/height").set('updated', 'yes')

    # Update size of each object
    for obj in root.findall("./object/bndbox/xmin"):
        orig_xmin = int(obj.text)
        curr_xmin = int(round(orig_xmin / factor))
        obj.text = str(curr_xmin)
        obj.set('updated', 'yes')
    
    for obj in root.findall("./object/bndbox/ymin"):
        orig_ymin = int(obj.text)
        curr_ymin = int(round(orig_ymin / factor))
        obj.text = str(curr_ymin)
        obj.set('updated', 'yes')

    for obj in root.findall("./object/bndbox/xmax"):
        orig_xmax = int(obj.text)
        curr_xmax = int(round(orig_xmax / factor))
        obj.text = str(curr_xmax)
        obj.set('updated', 'yes')

    for obj in root.findall("./object/bndbox/ymax"):
        orig_ymax = int(obj.text)
        curr_ymax = int(round(orig_ymax / factor))
        obj.text = str(curr_ymax)
        obj.set('updated', 'yes')

    os.chdir(DESTPATH)
    tree.write(FILENAME)

#-----------------------------------------------------------------------

if __name__ == "__main__":
    
    factor = RES_CURRENT / RES_ORIGINAL
    os.chdir(FILEPATH)
    
    # Loop over files in source folders
    fileList = []
    
    fileList = CreateFileList(FILEPATH)
    for filename in fileList:
        updateXml(filename, FILEPATH, DESTPATH)
        
    print "Updated xml files saved in ", DESTPATH
    
    
    
    
    
    
    
