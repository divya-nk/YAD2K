#! /usr/bin/env python

import argparse
import io
import os
import matplotlib
matplotlib.use('agg')
import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

argparser = argparse.ArgumentParser(
    description='Train YOLO_v2 model to labelled dataset.')

argparser.add_argument(
    '-d',
    '--data_path',
    help='path to HDF5 file containing pascal voc dataset',
    default='data/phaseI-dataset.hdf5')

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/yolo_anchors.txt')

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to labels.txt',
    default='model_data/labels.txt')

def _main(args):
    data_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
    else:
        anchors = YOLO_ANCHORS
        
    data = h5py.File(data_path, 'r')

    #Pre-processing data
    boxes_list, image_data_list = get_preprocessed_data(data)
    detectors_mask, matching_true_boxes = get_detector_mask(boxes_list, anchors)
    
    
    #Create model
    model_body, model = create_model(anchors, class_names, load_pretrained=True, freeze_body=False)
    
    #train model
    train(model, class_names, anchors, image_data_list, boxes_list, detectors_mask, matching_true_boxes)
    
    draw(model_body, class_names, anchors, image_data_list, image_set='val', # assumes training/validation split is 0.9
        weights_name='trained_stage_3_best.h5',
        save_all=False)
    
def get_preprocessed_data(data):
    image_list = []
    boxes_list = []
    image_data_list = []
    processed_box_data = []
    
    # boxes processing
    box_dataset = data['train/boxes']
    processed_box_data = boxprocessing(box_dataset)
    processed_box_data = processed_box_data.reshape(len(box_dataset),4,5)
    
    for i in range(len(box_dataset)): 
        image = PIL.Image.open(io.BytesIO(data['train/images'][i]))
        orig_size = np.array([image.width, image.height])
        orig_size = np.expand_dims(orig_size, axis=0)
        
        #Image preprocessing
        image = image.resize((416,416), PIL.Image.BICUBIC)
        image_data = np.array(image, dtype=np.float)
        image_data /= 255.0
        image_data.resize((image_data.shape[0], image_data.shape[1], 1))
        image_data = np.repeat(image_data, 3, 2)
        image_list.append(image)
        image_data_list.append(image_data)
        
        #Box preprocessing
        boxes = processed_box_data[i]
        
        #Get box parameters as x_center, y_center, box_width, box_height, class
        boxes_xy = 0.5 * (boxes[:, 3:5] + boxes[:, 1:3])
        boxes_wh = boxes[:, 3:5] - boxes[:, 1:3]
        boxes_xy = boxes_xy / orig_size
        boxes_wh = boxes_wh / orig_size
        boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 0:1]), axis=1)
        boxes_list.append(boxes)
        
    boxes_list = np.array(boxes_list, float)
    image_data_list = np.array(image_data_list, dtype=np.float)
    
    return np.array(boxes_list, float), np.array(image_data_list, dtype=np.float)

def boxprocessing(box_data):
    processed_box_data = []
    processed_box_data = np.array(processed_box_data)
    
    for i in range(len(box_data)):
        z = np.zeros([1,20])
        y = np.append(box_data[i], z)
        y = y[0:20]
        processed_box_data = np.append(processed_box_data, y)
    return processed_box_data

def get_detector_mask(boxes_list, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes_list))]
    matching_true_boxes = [0 for i in range(len(boxes_list))]
    for i, box in enumerate(boxes_list):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)
  
def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
    
    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)
        
    #Create model input layers
    image_input = Input(shape=(416,416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)
                
    #Create model body
    yolo_model = yolo_body(image_input,len(anchors),len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    
    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)
    
    #model_body = Model(image_input, model_body.output)
        
    with tf.device('/cpu:0'):
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1,),
            name='yolo_loss',
            arguments={'anchors': anchors,'num_classes': len(class_names)})([
            model_body.output, boxes_input,
            detectors_mask_input, matching_boxes_input])
    
    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)
    
    model.summary()
    #stop

    return model_body, model

def train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, validation_split=0.1):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    print('content of boxes')
    #print(boxes[1])
    print(boxes.shape)
    
    
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
    
    '''
    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=32,
              epochs=5,
              callbacks=[logging])
    model.save_weights('trained_stage_1.h5')
    
    
    model_body, model = create_model(anchors, class_names, load_pretrained=True, freeze_body=True)

    #model.load_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    
    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=8,
              epochs=30,
              callbacks=[logging])

    model.save_weights('trained_stage_2.h5')
    '''
    
    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=32,
              epochs=500,
              callbacks=[logging, checkpoint])

    model.save_weights('trained_stage_3.h5')
    
def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[:int(len(image_data)*.9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[int(len(image_data)*.9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
        
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.40, iou_threshold=0.0)

    # Run prediction
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image {}.'.format(len(out_boxes), str(i)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()
    
if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)


