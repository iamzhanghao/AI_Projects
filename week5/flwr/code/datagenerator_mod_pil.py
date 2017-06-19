import numpy as np
import PIL.Image

import os

import copy

"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""

class ImageDataGenerator_r:
    def __init__(self,class_list,replacementpath, horizontal_flip=False, shuffle=False, 
                 mean = np.array([104., 117., 124.]), scale_size=(227, 227),
                 nb_classes = 2):
        
                
        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0
        
        self.read_class_list_mod(class_list,replacementpath)
        
        if self.shuffle:
            self.shuffle_data()

    
    def read_class_list_mod(self,class_list,replacementpath):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            for l in lines:
                items = l.strip().split()
                
                
                z=os.path.basename(items[0])
                z=os.path.join(replacementpath,z)
                
                self.images.append(z)
                self.labels.append(int(items[1]))
            
            #store total number of data
            self.data_size = len(self.labels)
            
    
    '''        
    def read_class_list_flowers102(self,imgpath,labelfile):
        """
        get the flowers labels
        """
        numsamples=8189
        
        self.labels = []
        labelarr=np.load(labelfile)
        
        for i in range(labelarr.shape[0]):
          self.labels.append(labelarr[i])
        
        if(numsamples!=len(self.labels)):
          print('ERR: numsamples!=len(self.labels) ',numsamples,len(self.labels))
          exit()
        
        self.images = []
        
        for i in range(numsamples):
            fname=os.path.join(imgpath,'image_0'+str(i+1)+'.jpg')
            if not os.path.isfile(fname):
                print('image does not exist:',fname)
                exit()
            
            self.images.append(fname)
        

            
            #store total number of data
        self.data_size = len(self.labels)        
    '''
        
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = copy.deepcopy(self.images)
        labels = copy.deepcopy(self.labels)
        self.images = []
        self.labels = []
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        paths = self.images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]
        
        #update pointer
        self.pointer += batch_size
        
        # Read images
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
        
            #print(paths[i])
        
            img0 = PIL.Image.open(paths[i])
            
            #flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.5:
                img0=img0.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            
            #rescale image
            img0=img0.resize( (self.scale_size[0], self.scale_size[1]),PIL.Image.LANCZOS) # PIL.Image.BICUBIC)
            img = np.array(img0,dtype=np.float32)
            if(img.ndim>3):
                img=img[:,:,0:3]
            img=img[:,:,[2,1,0]]
            
            #subtract mean
            img -= self.mean
                                                                 
            images[i] = img

        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        #return array of images and labels
        return images, one_hot_labels

