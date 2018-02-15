# coding: utf-8

import numpy as np
import os
import sys
import pandas as pd
from osgeo import gdal,ogr,osr

PATH_TO_TEST_IMAGES_DIR = '/home/madi/TensorFlow_utils/trees_recognition/images/pred_tile/'
# TEST_IMAGE_PATHS = os.path.join(PATH_TO_TEST_IMAGES_DIR, 'aaa_pt604000-4399000.jpg' )
# TEST_CSV_PATH = os.path.join(PATH_TO_TEST_IMAGES_DIR, 'aaa_pt604000-4399000.csv' )
# CSV_REPR_PATH = os.path.join(PATH_TO_TEST_IMAGES_DIR, 'aaa_pt604000-4399000_repr.csv' )

#-------------------------------------------------------------------------------

def GetExtent(gt, cols, rows):
    ext = []
    xarr = [0, cols]
    yarr = [0, rows]
    for px in xarr:
        for py in yarr:
            x = gt[0] + (px * gt[1]) + (py * gt[2])
            y = gt[3] + (px * gt[4]) + (py * gt[5])
            ext.append([x, y])
            print x, y
        yarr.reverse()
    return ext

#-------------------------------------------------------------------------------

def ReprojectCoords(coords, ext):
    '''From normalized coords to src
    '''
    xmin_img = ext[0][0]
    xmax_img = ext[2][0]
    ymin_img = ext[1][1]
    ymax_img = ext[0][1]
    trans_coords=[]
    ymax_t =  ymin_img + ((ymax_img - ymin_img) * coords[1][1]) #ymax
    xmin_t =  xmin_img + ((xmax_img - xmin_img) * coords[0][0]) #xmin
    ymin_t =  ymin_img + ((ymax_img - ymin_img) * coords[0][1]) #ymin
    xmax_t =  xmin_img + ((xmax_img - xmin_img) * coords[1][0]) #xmax
    trans_coords.append([ymax_t, xmin_t, ymin_t, xmax_t])
    return trans_coords

#-------------------------------------------------------------------------------

def writeShapefile(df, outShp, src):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outShp):
        driver.DeleteDataSource(outShp)
    ds = driver.CreateDataSource(outShp)
    layer = ds.CreateLayer('Boxes', src, ogr.wkbPolygon)
    # Add fields
    field_classe = ogr.FieldDefn("classe", ogr.OFTInteger)
    field_classe.SetWidth(3)
    layer.CreateField(field_classe)
    field_score = ogr.FieldDefn("score", ogr.OFTReal)
    field_score.SetWidth(20)
    layer.CreateField(field_score)
    # Process df and add the attributes and features to the shapefile
    for i in range(len(df)):
        # Create geometry of a box
        box = ogr.Geometry(ogr.wkbLinearRing)
        box.AddPoint(df.iloc[i].xmin, df.iloc[i].ymin) #LL
        box.AddPoint(df.iloc[i].xmin, df.iloc[i].ymax) #UL
        box.AddPoint(df.iloc[i].xmax, df.iloc[i].ymax) #UR
        box.AddPoint(df.iloc[i].xmax, df.iloc[i].ymin) #LR
        box.AddPoint(df.iloc[i].xmin, df.iloc[i].ymin) #close ring
        # Create polygon
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(box)
        #print 'Polygon area = ', poly.GetArea()
        #print poly.ExportToWkt()
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("classe", df.iloc[i].classe)
        feature.SetField("score", df.iloc[i].score)
        # Set the feature geometry using the box
        feature.SetGeometry(poly)
        # Create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # Flush memory
        feature.Destroy()
    # Deference all
    ds = layer = feature = poly = None

#-------------------------------------------------------------------------------

# Make the conversion
def Convert(TEST_IMAGE_PATHS, TEST_CSV_PATH, CSV_REPR_PATH, PATH_TO_TEST_IMAGES_DIR):
    # create dataframe
    df = pd.read_csv(TEST_IMAGE_PATHS.split(".")[0] + ".csv", \
                     header = 0, \
                     names = ['ID','ymax', 'xmin', 'ymin', 'xmax', 'classe', 'score'], \
                     index_col = 'ID', \
                     usecols = ['ID','ymax', 'xmin', 'ymin', 'xmax', 'classe', 'score'])

    # Remove records having score < 0.10
    df = df.drop(df[df.score < 0.10].index)

    # The y axis is reversed
    df['ymin'] = df['ymin'].apply(lambda x: 1 - x)
    df['ymax'] = df['ymax'].apply(lambda x: 1 - x)

    # Retrieve coordinates and src of image corners
    raster = TEST_IMAGE_PATHS
    ds = gdal.Open(raster)
    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    ext = GetExtent(gt,cols,rows)
    wkt = ds.GetProjection()
    src = osr.SpatialReference()
    src.ImportFromWkt(wkt)

    # write file
    fileCSV = open(CSV_REPR_PATH, "w")
    fileCSV.write("ID, ymax, xmin, ymin, xmax, classe, score" + "\n")
    coords = []
    reprojected = []
    for i in range(len(df)):
        coord = [[df.xmin[i],df.ymin[i]], [df.xmax[i],df.ymax[i]]]
        coords.append(coord)
        # Perform coordinate conversion of boxes
        repr = ReprojectCoords(coord, ext)
        rep = [i, repr, df.classe[i], df.score[i]]
        reprojected.append(rep)
        # write csv
        fileCSV.write("%s" % i)
        fileCSV.write(",")
        fileCSV.write("%s" % repr[0][0])
        fileCSV.write(",")
        fileCSV.write("%s" % repr[0][1])
        fileCSV.write(",")
        fileCSV.write("%s" % repr[0][2])
        fileCSV.write(",")
        fileCSV.write("%s" % repr[0][3])
        fileCSV.write(",")
        fileCSV.write("%s" % df.classe[i])
        fileCSV.write(",")
        fileCSV.write("%s" % df.score[i])
        fileCSV.write("\n")

    fileCSV.close()

    # Import clean dataframe
    # create dataframe
    df = pd.read_csv(CSV_REPR_PATH, \
                     header = 0, \
                     names = ['ID','ymax', 'xmin', 'ymin', 'xmax', 'classe', 'score'], \
                     index_col = 'ID', \
                     usecols = ['ID','ymax', 'xmin', 'ymin', 'xmax', 'classe', 'score'])

    # Create shapefile
    outShp = os.path.join(PATH_TO_TEST_IMAGES_DIR, TEST_IMAGE_PATHS.split(".")[0]) + ".shp"
    writeShapefile(df, outShp, src)
    print "Shapefile written in ", outShp

#-------------------------------------------------------------------------------

def CreateFileList(FILEPATH):
    for filename in os.listdir(str(FILEPATH)):
        if filename.endswith(".jpg") and "aux" not in filename:
            fileList.append(filename.split(".")[0])
    fileList.sort()
    print fileList
    return fileList
    
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    # loop over files in folder    
    os.chdir(PATH_TO_TEST_IMAGES_DIR)

    fileList = []
    
    fileList = CreateFileList(PATH_TO_TEST_IMAGES_DIR)
    for filename in fileList:
        TEST_IMAGE_PATHS = os.path.join(PATH_TO_TEST_IMAGES_DIR, filename + '.jpg' )
        TEST_CSV_PATH = os.path.join(PATH_TO_TEST_IMAGES_DIR, filename + '.csv' )
        CSV_REPR_PATH = os.path.join(PATH_TO_TEST_IMAGES_DIR, filename + '_repr.csv' )
        Convert(TEST_IMAGE_PATHS, TEST_CSV_PATH, CSV_REPR_PATH, PATH_TO_TEST_IMAGES_DIR)
        
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
