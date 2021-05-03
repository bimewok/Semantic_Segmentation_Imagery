# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 08:14:36 2021

@author: bimew
"""

def create_raster_like(original_path, array, output_path, num_bands=1):
    from osgeo import gdal
    import os
    
    if os.path.isfile(output_path) == True:
        os.remove(output_path)
    
    original_raster = gdal.Open(original_path)
    
    driver = gdal.GetDriverByName('GTiff')
    
    new_raster = driver.Create(
        output_path, 
        original_raster.RasterXSize, 
        original_raster.RasterYSize, 
        num_bands, 
        gdal.GDT_Byte
        )
    
    new_raster.SetGeoTransform(original_raster.GetGeoTransform())
    new_raster.SetProjection(original_raster.GetProjection())
    
    for i in range(num_bands):
        band = new_raster.GetRasterBand(i+1)
        band.WriteArray(array[...,i])
    
    original_raster = None
    new_raster = None
    band = None
    


def merge_rasters(folder):
    from osgeo import gdal
    import glob
    
    files_to_merge = []
    for file in glob.glob(folder+'\\'+'*.tif'):
        files_to_merge.append(file)
    
    merged_filepath = folder+'\\'+'merged.tif'
    
    merged = gdal.Warp(
    merged_filepath, 
    files_to_merge, 
    format='GTiff'
    )
    
    merged = None
    

def tile_raster(input_raster, output_dir, tile_size, gdal_retile_path):
    from osgeo import gdal
    import os
    import subprocess
    
    retile_string = '''
call activate {env}
python {gdal_retile_path} ^
-ps {tile_size} {tile_size} ^
-overlap 0 ^
-ot UInt16 ^
-levels 1 ^
-targetDir {tile_dir} ^
{merged_filepath}
'''.format(
    gdal_retile_path=gdal_retile_path,
    tile_size=tile_size,
    tile_dir=output_dir,
    merged_filepath=input_raster,
    env=os.environ['CONDA_DEFAULT_ENV']
    )

    text_file = open(output_dir+'\\'+'tile_raster.bat', "wt")
    n = text_file.write(retile_string)
    text_file.close()
    tile_raster = subprocess.call([output_dir+'\\'+'tile_raster.bat'])
    os.remove(output_dir+'\\'+'tile_raster.bat')
    
    
def load_rasters_to_array(directory, file_type, num_bands, tile_size):
    import glob
    import numpy as np
    from osgeo import gdal
    
    file_list = []
    data = []
    
    
    for file in glob.glob(directory+'\\'+'*.{}'.format(file_type)):
  
        img = gdal.Open(file)
        bands = []
        for i in range(1, num_bands+1):
            band = np.array(img.GetRasterBand(i).ReadAsArray())
            bands.append(band)
        arr = np.dstack(bands)
        if arr.shape == (tile_size, tile_size, num_bands):
            file_list.append(file)
            
        img = None
    
    file_list.sort()
    for file in file_list:
  
        img = gdal.Open(file)
        bands = []
        for i in range(1, num_bands+1):
            band = np.array(img.GetRasterBand(i).ReadAsArray())
            bands.append(band)
        arr = np.dstack(bands)

        data.append(arr)
        img = None
    
    return np.array(data, dtype=np.int32), file_list


def raster_to_poly(raster_path, shp_out_path, layer_name, field_name):
    from osgeo import gdal, ogr, osr
    
    raster = gdal.Open(raster_path)
    band = raster.GetRasterBand(1)
    driver = ogr.GetDriverByName('ESRI Shapefile')        
    poly = driver.CreateDataSource(shp_out_path)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(raster.GetProjectionRef())

    out_layer = poly.CreateLayer(layer_name, srs)        
    new_field = ogr.FieldDefn(field_name, ogr.OFTReal)
    
    out_layer.CreateField(new_field)        
    gdal.FPolygonize(band, band, out_layer, 0, [], callback=None)        
    poly.Destroy()
    source_raster = None