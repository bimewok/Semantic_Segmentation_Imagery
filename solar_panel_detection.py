# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 08:21:49 2021

@author: bimew
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow_examples.models.pix2pix import pix2pix
import glob, os, shutil, subprocess, time
import gdal
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
import geopandas as gpd
import pandas as pd


base_dir = os.path.realpath(__file__)
os.chdir(base_dir)

from bg_geo_tools.bg_geo_tools import *




train_dir = base_dir+'\\train'
test_dir = base_dir+'\\test'
labels_dir = base_dir+'\\train\\labels'
labels_tiles_dir = labels_dir+'\\'+'tiles'
test_tiles_dir = test_dir+'\\'+'tiles'
train_tiles_dir = train_dir+'\\'+'tiles'
gdal_retile_path = base_dir+'\\bg_geo_tools\\gdal_retile.py'
predictions_dir = base_dir+'\\predictions'
temp_dir = predictions_dir+'\\temp'
shapes_dir = base_dir+'\\shapes'

tile_size = 256 # images will be tiled to tile_size x tile_size pixels





#====================== Create blank label rasters ===========================
''' The labels to be used for the input of the model will be raster tiles
where each pixel is classified as a solar panel (1) or not (0). To create this,
build a polygon layer in a GIS with the same shape and CRS as your training 
area. Then, trace polygons representing the features of interest and set 
their attributes to be an integer code. Once all features in the training area
are traced, convert the polygon to a raster layer with the same specs as the
imagery. In this example, my traced polygons are in solar_panels_poly.py.
The following code converts this into a raster. In my examample, I chose 
to include two training areas that do not touch geographically. The below
rasterize tool will try to create empty pixels for the space inbetween, thus
you can save a lot of time and post-processing by doing the rasterize operation 
on each training area.
'''

label_rast_path = labels_dir+'\\labelsN.tif'
cell_size = 0.3
extent_coords = '565500.0 4189500.0 567000.0 4191000.0'
labels_poly_path = train_dir+'\\solar_panels_poly_north.shp'
labels_poly_name = 'solar_panels_poly_north'
class_label_field = 'class'

gdal_string = ('gdal_rasterize -l '
               '{labels_poly_name} -a {class_label_field} '
               '-tr {cell_size} {cell_size} '
               '-a_nodata 2.0 -te {extent_coords} '
               '-ot UInt16 -of GTiff {labels_poly_path} '
               '{label_rast_path}').format(labels_poly_name=labels_poly_name,
                                           class_label_field=class_label_field, 
                                           cell_size=cell_size,
                                           extent_coords=extent_coords,
                                           labels_poly_path=labels_poly_path,
                                           label_rast_path=label_rast_path) 

os.system(gdal_string)
time.sleep(5)
                                           
label_rast_path = labels_dir+'\\labelsS.tif'
extent_coords = '547500.0 4174500.0 549000.0 4176000.0'
labels_poly_path = train_dir+'\\solar_panels_poly_south.shp'
labels_poly_name = 'solar_panels_poly_south'


gdal_string = ('gdal_rasterize -l '
               '{labels_poly_name} -a {class_label_field} '
               '-tr {cell_size} {cell_size} '
               '-a_nodata 2.0 -te {extent_coords} '
               '-ot UInt16 -of GTiff {labels_poly_path} '
               '{label_rast_path}').format(labels_poly_name=labels_poly_name,
                                           class_label_field=class_label_field, 
                                           cell_size=cell_size,
                                           extent_coords=extent_coords,
                                           labels_poly_path=labels_poly_path,
                                           label_rast_path=label_rast_path)                                            
                                           
os.system(gdal_string)                                           
                                           
                                           
                                           
                                           
#====================== tile rasters =========================================
'''The original and labels rasters must be split into tiles of specified size
before being fed into the model. The following code deletes the tiles if they 
exisit in the tile folders and then tiles the original imagery and label rasters
'''                                           

for folder in [labels_tiles_dir, test_tiles_dir, train_tiles_dir]:
    for file in glob.glob(folder+'\\'+'*.*'):
        os.remove(file)
                                        
                                           

train_to_tile = []
test_to_tile = []
labels_to_tile = []

for file in glob.glob(labels_dir+'\\'+'*.tif'):
    labels_to_tile.append(file)
    tile_raster(file, labels_tiles_dir, tile_size, gdal_retile_path)
    
for file in glob.glob(test_dir+'\\'+'*.tif'):
    test_to_tile.append(file)
    tile_raster(file, test_tiles_dir, tile_size, gdal_retile_path)
    
for file in glob.glob(train_dir+'\\'+'*.tif'):
    train_to_tile.append(file)
    tile_raster(file, train_tiles_dir, tile_size, gdal_retile_path)


for folder in [labels_tiles_dir, test_tiles_dir, train_tiles_dir]:
    try:
        shutil.rmtree(folder+'\\1')
    except:
        pass

#====================== load tiles to numpy arrays ===========================

test, test_files = load_rasters_to_array(
    test_tiles_dir, 'tif', 3, tile_size
    )

train, train_files = load_rasters_to_array(
    train_tiles_dir, 'tif', 3, tile_size
    )

labels, label_files = load_rasters_to_array(
    labels_tiles_dir, 'tif', 1, tile_size
    )


print('test shape', test.shape)    
print('train shape', train.shape)   
print('labels shape', labels.shape)

#====================== augment images with positives ========================

new_train = []
new_labels = []

for i in range(len(train)):
    if 1 in labels[i].flatten():
        new_train.append(train[i])
        new_labels.append(labels[i])
        
        new_train.append(np.fliplr(train[i]))
        new_labels.append(np.fliplr(labels[i]))
        
        for rot in range(45, 316, 45):
        
            new_train.append(scipy.ndimage.rotate(train[i], rot, reshape=False))
            new_labels.append(scipy.ndimage.rotate(labels[i], rot, reshape=False))
            new_train.append(scipy.ndimage.rotate(np.fliplr(train[i]), rot, reshape=False))
            new_labels.append(scipy.ndimage.rotate(np.fliplr(labels[i]), rot, reshape=False))
        
    else:
        new_train.append(train[i])
        new_labels.append(labels[i])       

train = np.array(new_train)
labels = np.array(new_labels)   

print('augmented train shape', train.shape)   
print('augmented labels shape', labels.shape)

#====================== shuffle and scale color images =======================

randomize = np.arange(len(train))
np.random.shuffle(randomize)
train = train[randomize]
labels = labels[randomize]

labels = labels.reshape(len(labels), 256, 256)



train = train / 255
test = test / 255


#====================== build model ==========================================

TRAIN_LENGTH = len(train)
BATCH_SIZE = 8
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
OUTPUT_CHANNELS = 3
base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3], include_top=False)
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False


up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)



model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])




es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=4,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)


EPOCHS = 200

model_history = model.fit(train, labels, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          callbacks=es,
                          validation_split = 0.21)

keras.models.save_model(model, base_dir+'\\model')

#====================== get and show predictions =============================

pred = model.predict(test)

masks = []
for i in range(len(pred)):
    predict = pred[i]
    am = np.argmax(predict, axis=-1)
    masks.append(am)
    
masks = np.array(masks)
print('got masks')
rand_ind = np.random.choice(np.arange(len(masks)), size=20)

for i in rand_ind:
    f, (ax1, ax2) = plt.subplots(1,2)
    f.suptitle('Model Preciction Masks', y=0.89)
    #f.tight_layout()
    ax1.imshow(masks[i])
    ax1.set_title('Prediction')
    ax2.imshow(test[i])
    ax2.set_title('Actual')
    plt.show()
    
#====================== convert predictions to polygon =======================
''' The following loop writes each prediction array as a georeferenced tiff
with the same geometry as the original tile, converts it to a shapefile (poly),
and merges those shapes into a master shapefile of all positive predictions.
'''



all_positives = gpd.GeoDataFrame()
at = 1

for i in range(len(test_files)):
    arr = masks[i].reshape(tile_size, tile_size, 1)
    in_raster = test_files[i]
    out_raster = temp_dir+'\\'+in_raster.split('\\')[-1]

    create_raster_like(in_raster, arr, out_raster)
    shape_name = (out_raster.split('\\')[-1]).split('.')[0]
    raster_to_poly(out_raster, temp_dir, shape_name, 'class')
    shp = gpd.read_file(temp_dir+'\\'+shape_name+'.shp')
    if len(shp) > 0:
        crs = shp.crs
        all_positives = all_positives.append(shp, ignore_index=True)
    for file in glob.glob(temp_dir+'\\'+'*.*'):
        os.remove(file)
    if at % 100 == 0:
        print(len(test_files)-at,'to go')
    at += 1

all_positives.crs = crs


all_positives.to_file(predictions_dir+'\\all_positives.geojson', driver='GeoJSON')


#====================== remove positives that are not on buildings ===========
''' We can filter out some false positives by removing those that are not on
buildings. Luckily, OSM has decent data for this region, and we can use their 
footprints rather than manually having to trace buildings.
'''


buildings = gpd.read_file(shapes_dir+'\\osm_building_footprints.shp')

roof_top_panels = gpd.sjoin(all_positives, buildings, how='inner', op='within')

roof_top_panels = roof_top_panels[['class', 'geometry']].to_file(
    predictions_dir+'\\rooftop.geojson', driver='GeoJSON'
    )
 