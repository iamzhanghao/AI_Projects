import tensorflow as tf
import numpy as np

import math
from PIL import Image
from numpy import int32
import copy

import sys

from getimagenetclasses import *
from alexnet import * 


def preproc_py2(imname,shorterside):

  pilimg = Image.open(imname)
  w,h=pilimg.size
  
  print((w,h))
  
  if w > h:
    longerside= np.int32(math.floor(float(shorterside)/float(h)*w))
    neww=longerside
    newh=shorterside
  else:
    longerside= np.int32(math.floor(float(shorterside)/float(w)*h))
    newh=longerside
    neww=shorterside    
  resimg=pilimg.resize((neww,newh))
  
  
  im = np.array(resimg,dtype=np.float32)

  return im
  
def cropped_center(im,hsize,wsize):
  h=im.shape[0]
  w=im.shape[1]
  
  cim=im[(h-hsize)/2:(h-hsize)/2+hsize,(w-wsize)/2:(w-wsize)/2+wsize,:]
  
  return cim

def preproc(image):
  
  '''
  filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./images/*.jpg"))

  # Read an entire image file which is required since they're JPEGs, if the images
  # are too large they could be split in advance to smaller files or use the Fixed
  # reader to split up the file.
  image_reader = tf.WholeFileReader()
  
  # Read a whole file from the queue, the first returned value in the tuple is the
  # filename which we are ignoring.
  _, image_file = image_reader.read(filename_queue)
  
  '''
  
  height=tf.shape(image)[0]
  width=tf.shape(image)[1]
  new_shorter_edge=tf.cast(227, tf.int32)
  
  def _compute_longer_edge(height, width, new_shorter_edge):
    return tf.cast(width*new_shorter_edge/height, tf.int32)
  
  
  
  height_smaller_than_width = tf.less_equal(height, width)
  new_height_and_width = tf.cond(
    height_smaller_than_width,
    lambda: (new_shorter_edge, _compute_longer_edge(height, width, new_shorter_edge)),
    lambda: (_compute_longer_edge(width, height, new_shorter_edge), new_shorter_edge)
  )
  
  
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  
    
  resimg=tf.image.resize_images(
      image,
      new_height_and_width
      #tf.stack(new_height_and_width) #,
      #tf.concat(new_height_and_width)
      #method=ResizeMethod.BILINEAR,
      #align_corners=False
  )
  resimg = tf.expand_dims(resimg, 0)
  #image = tf.subtract(image, 110.0)

  return resimg

def getout():
  
  batchsize=1
  is_training=False
  num_classes=1000
  keep_prob=1.
  skip_layer=[]
  
  x = tf.placeholder(tf.float32, [batchsize, 227, 227, 3])  
  net=AlexNet( x, keep_prob, num_classes, skip_layer, is_training, weights_path = 'DEFAULT')
  
  out=net.fc8

  return out,x,net

def run2():

  cstepsize=20.0
  target_label=949
  imname = 'C:\\Users\H\PycharmProjects\AI_Projects\week6\mon\imgs\img0.png'

  imagenet_mean = np.array([104., 117., 123.], dtype=np.float32) 
  cls=get_classes()
  
  sess = tf.Session()

  out,x,net=getout()
  init = tf.global_variables_initializer()
  sess.run(init)
  net.load_initial_weights(sess)

  image=preproc_py2(imname,227)
  image=cropped_center(image,227,227)
  print(image.shape)
  print(imname)

  print(type(image))

  #convert grey to color    
  if(image.ndim<3):
    image=np.expand_dims(image,2)
    image=np.concatenate((image,image,image),2)
    
  # dump alpha channel if it exists    
  if(image.shape[2]>3):
    image=image[:,:,0:3]

  image=image[:,:,[2,1,0]] #RGB to BGR
  image=image-imagenet_mean
  image=np.expand_dims(image,0)


  #run initial classification
  predict_values=sess.run(out, feed_dict={x: image})
  

  original_label=np.argmax(predict_values)

  print('at start: classindex: ' + str(original_label)+ '\nclasslabel: '+ cls[np.argmax(predict_values)]+'\nscore:'+ str(np.max(predict_values)))
  print(predict_values[0,target_label],predict_values[0,original_label])
  print(len(predict_values))




  img = Image.fromarray(image)
  img.save('C:\\Users\H\PycharmProjects\AI_Projects\week6\mon\imgs\modified_img0.png')
  img.show()

if __name__=='__main__':
  run2()
