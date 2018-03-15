'''
made few changes according to my dataset

Also included code to prepare individual text files (for train, val and test) that includes the list of files names
and store it in 'ImageSets/....txt'

No need of separate dataset for 'val' as train-val split is implemented in the yolo_retrain.py
'''

import argparse
import os
import xml.etree.ElementTree as ElementTree

import h5py
import numpy as np

train_set = 'train'
#val_set = 'val'
test_set = 'test'

classes = ["lable-1", "lable-2"]

parser = argparse.ArgumentParser(
    description='Convert object detection phase-I dataset to HDF5.')
parser.add_argument(
    '-p',
    '--path_to_data',
    help='path to Images',
    default='data')


def get_boxes_for_id(data_path, dataset, image_id):
    """Get object bounding boxes annotations for given image.

    Parameters
    ----------
    data_path : str
        Path to data directory.
    dataset : str
        Folder name for train, test or val
    image_id : str
        File name for given image.

    Returns
    -------
    boxes : array of int
        bounding box annotations of class label, xmin, ymin, xmax, ymax as a
        5xN array.
    """
    fname = os.path.join(data_path, 'Annotations/{}/{}.txt'.format(dataset,image_id))
    with open(fname) as in_file:
        xml_tree = ElementTree.parse(in_file)
    root = xml_tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        label = obj.find('name').text
        xml_box = obj.find('bndbox')
        bbox = (classes.index(label), int(xml_box.find('xmin').text),
                int(xml_box.find('ymin').text), int(xml_box.find('xmax').text),
                int(xml_box.find('ymax').text))
        boxes.extend(bbox)
   
    return np.array(boxes)

def get_image_for_id(data_path, dataset, image_id):
    """Get image data as uint8 array for given image.

    Parameters
    ----------
    data_path : str
        Path to data directory.
    dataset : str
        Folder name for - train, test or val
    image_id : str
        File name for given image.

    Returns
    -------
    image_data : array of uint8
        Compressed PNG byte string represented as array of uint8.
    """
    fname = os.path.join(data_path, 'PNGImages/{}/{}.png'.format(dataset,image_id))
    with open(fname, 'rb') as in_file:
        data = in_file.read()
    # Use of encoding based on: https://github.com/h5py/h5py/issues/745
    return np.fromstring(data, dtype='uint8')

def get_ids(data_path, dataset):
    """Get image identifiers for corresponding list of dataset identifies.

    Parameters
    ----------
    data_path : str
        Path to data directory.
    dataset : train, test or val

    Returns
    -------
    ids : list of str
        List of all image identifiers for given datasets.
    """
    
    #writes file names in txt files
    for subdir,dirs,files in os.walk(data_path+os.sep+'PNGImages'+os.sep+dataset):
        for file in files:
            with open(data_path+os.sep+'ImageSets'+os.sep+dataset+'.txt', 'a') as f:
                f.write(os.path.splitext(file)[0]+'\n')
    #print('done')
    
    ids = []
    id_file = os.path.join(data_path, 'ImageSets/{}.txt'.format(dataset))
    print(id_file)
    with open(id_file, 'r') as image_ids:
        ids.extend(map(str.strip, image_ids.readlines()))
    return ids

def add_to_dataset(data_path, dataset, ids, images, boxes, start=0):
    """Process all given ids and adds them to given datasets."""
    for i, img_id in enumerate(ids):
        image_data = get_image_for_id(data_path, dataset, img_id)
        image_boxes = get_boxes_for_id(data_path, dataset, img_id)
        images[start + i] = image_data
        boxes[start + i] = image_boxes
    return i

def _main(args):
    data_path = os.path.expanduser(args.path_to_data)
    train_ids = get_ids(data_path, train_set)
    #val_ids = get_ids(data_path, val_set)
    test_ids = get_ids(data_path, test_set)
    
    #Create HDF5 dataset structure
    print('Creating HDF5 dataset structure...')
    fname = os.path.join(data_path, 'phaseI-dataset.hdf5')
    phaseI_h5file = h5py.File(fname, 'w')
    uint8_dt = h5py.special_dtype(vlen=np.dtype('uint8')) #variable length uint8
    uint16_dt = h5py.special_dtype(vlen=np.dtype('uint16')) # included uint16 as coordinates of bounding boxes are > 255 
    vlen_int_dt = h5py.special_dtype(vlen=np.dtype(int)) #variable lenght int
    train_group = phaseI_h5file.create_group('train')
    #val_group = phaseI_h5file.create_group('val')
    test_group = phaseI_h5file.create_group('test')
    
    #store class list for reference class ids as csv fixed-length numpy string
    phaseI_h5file.attrs['classes'] = np.string_(str.join(',', classes))
    
    #store images as variable length uint8 array
    train_images = train_group.create_dataset('images', shape=(len(train_ids), ), dtype=uint8_dt)
    #val_images = val_group.create_dataset('images', shape=(len(val_ids), ), dtype=uint8_dt)
    test_images = test_group.create_dataset('images', shape=(len(test_ids), ), dtype=uint8_dt)
    
    #store boxes as class_id, xmin, ymin, xmax, ymax
    train_boxes = train_group.create_dataset('boxes', shape=(len(train_ids), ), dtype=uint16_dt)
    #val_boxes = val_group.create_dataset('boxes', shape=(len(val_ids), ), dtype=uint16_dt)
    test_boxes = test_group.create_dataset('boxes', shape=(len(test_ids), ), dtype=uint16_dt)
    
    #process all ids and add to datasets
    print('Processing Phase I datasets for training set.')
    add_to_dataset(data_path, train_set, train_ids, train_images, train_boxes)
    #print('Processing Phase I datasets for val set.')
    #add_to_dataset(data_path, val_set, val_ids, val_images, val_boxes)
    print('Processing Phase I datasets for test set.')
    add_to_dataset(data_path, test_set, test_ids, test_images, test_boxes)
    
    print('Closing HDF5 file.')
    phaseI_h5file.close()
    print('Done!')
    
if __name__=='__main__':
    _main(parser.parse_args())
                              

