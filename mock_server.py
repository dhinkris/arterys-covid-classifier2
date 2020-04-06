"""
Demo script that starts a server which exposes liver segmentation.

Based off of https://github.com/morpheus-med/vision/blob/master/ml/experimental/research/prod/model_gateway/ucsd_server.py
"""

import functools
import logging
import logging.config
import os
import tempfile
import yaml
import json
import numpy
import pydicom

from utils.image_conversion import convert_to_nifti

from utils import tagged_logger
import tensorflow as tf
# ensure logging is configured before flask is initialized
print(tf.__version__)

with open('logging.yaml', 'r') as f:
    conf = yaml.safe_load(f.read())
    logging.config.dictConfig(conf)

logger = logging.getLogger('inference')

# pylint: disable=import-error,no-name-in-module
from gateway import Gateway
from flask import make_response
import cv2
import numpy as np

def handle_exception(e):
    logger.exception('internal server error %s', e)
    return 'internal server error', 500

def get_empty_response():
    response_json = {
        'protocol_version': '1.0',
        'parts': []
    }
    return response_json, []
def load_model():
    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph('./covidnet/output/model.meta_eval')
    saver.restore(sess, './covidnet/output/model-2069')

    graph = tf.get_default_graph()

    pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")

    return pred_tensor, sess, graph

def get_bounding_box_2d_response(json_input, dicom_instances):
    base_model, sess, graph = load_model()
    image_tensor = graph.get_tensor_by_name("input_1:0")

    height = 224
    width = 224
    channel = 3

    response_json = {
        'protocol_version': '1.0',
        'parts': [],
        'bounding_boxes_2d': []
    }
    for instances in dicom_instances:
        dcm = pydicom.read_file(instances)
        dataset = dcm.pixel_array
        img = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
        stacked = cv2.resize(np.array(np.stack((img,img,img), axis=-1)), (height,width))

        img = np.reshape(stacked, (1,height, width, channel))
        prediction = sess.run(base_model, feed_dict={image_tensor: img})

        if np.argmax(prediction, axis=1)==0:
            label='negative'
        elif np.argmax(prediction, axis=1)==1:
            label='positive'

        response_json['bounding_boxes_2d'].append(
            {
                'SOPInstanceUID': dcm.SOPInstanceUID,
                'top_left': [0, 0],
                'bottom_right': [dataset.shape[0], dataset.shape[1]],
                'label': label
            }
        )


    return response_json, []


def request_handler(json_input, dicom_instances, input_digest):
    """
    A mock inference model that returns a mask array of ones of size (height * depth, width)
    """
    transaction_logger = tagged_logger.TaggedLogger(logger)
    transaction_logger.add_tags({ 'input_hash': input_digest })
    transaction_logger.info('mock_model received json_input={}'.format(json_input))

    # If your model accepts Nifti files as input then uncomment the following lines:
    # convert_to_nifti(dicom_instances, 'nifti_output.nii')
    # print("Converted file to nifti 'nifti_output.nii'")
    
    if json_input['inference_command'] == 'get-bounding-box-2d':
        return get_bounding_box_2d_response(json_input, dicom_instances)
    else:
        return get_empty_response()


if __name__ == '__main__':
    app = Gateway(__name__)
    app.register_error_handler(Exception, handle_exception)
    app.add_inference_route('/', request_handler)

    app.run(host='0.0.0.0', port=8002   , debug=True, use_reloader=True)
