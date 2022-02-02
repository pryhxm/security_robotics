#!/usr/bin/env python3
import os

"""
Create the folder data and models 
"""
DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
for dir in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)

import tarfile
import urllib.request

"""
Download the model
"""
MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_resnet101_v1_fpn_640x640_coco17_tpu-8'
MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
if not os.path.exists(PATH_TO_CKPT):
    print('Downloading model. This may take a while... ', end='')
    urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
    tar_file = tarfile.open(PATH_TO_MODEL_TAR)
    tar_file.extractall(MODELS_DIR)
    tar_file.close()
    os.remove(PATH_TO_MODEL_TAR)
    print('Done')

"""
Download labels file
"""
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = \
    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
if not os.path.exists(PATH_TO_LABELS):
    print('Downloading label file... ', end='')
    urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
    print('Done')

"""
Load the model
"""
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    # print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

configs = config_util.get_configs_from_pipeline_file(
    "data/models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8/pipeline.config")
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)
# print(model_config)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore("data/models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0").expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(
    "data/models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8/mscoco_label_map.pbtxt",
    use_display_name=True)


"""
Detect objects in a frame
"""
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections, prediction_dict, tf.reshape(shapes, [-1])


import cv2
import numpy as np
import threading

cap = cv2.VideoCapture("target_video.mp4")
# cap = cv2.VideoCapture(0)

lock = threading.Lock()
frame_od = np.zeros(1)


def object_detect():
    global frame_od

    while True:
        # Read frame from camera
        ret, image_np = cap.read()
        # image_np_expanded = np.expand_dims(image_np, axis=0)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        detection_boxes = None
        detection_classes = None
        detection_scores = None

        label_id_offset = 1

        for i in range(100):
            # i = i+1
            if detections['detection_classes'][0][i] + label_id_offset == 1 or detections['detection_classes'][0][
                i] + label_id_offset == 3:

                if detection_boxes is None:
                    detection_boxes = tf.reshape(detections['detection_boxes'][0][i], [1, 4])
                    detection_classes = tf.reshape(detections['detection_classes'][0][i], [1, 1])
                    detection_scores = tf.reshape(detections['detection_scores'][0][i], [1, 1])
                else:
                    detection_boxes = tf.concat(
                        [detection_boxes, tf.reshape(detections['detection_boxes'][0][i], [1, 4])], 0)
                    detection_classes = tf.concat(
                        [detection_classes, tf.reshape(detections['detection_classes'][0][i], [1, 1])], 0)
                    detection_scores = tf.concat(
                        [detection_scores, tf.reshape(detections['detection_scores'][0][i], [1, 1])], 0)
        image_np_with_detections = image_np.copy()
        detection_classes = tf.reshape(detection_classes, [tf.shape(detection_classes)[0]])
        detection_scores = tf.reshape(detection_scores, [tf.shape(detection_scores)[0]])

        # Visualize the result with box and name
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detection_boxes.numpy(),
            (detection_classes.numpy() + label_id_offset).astype(int),
            detection_scores.numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)
        lock.acquire()
        try:
            frame_od = image_np_with_detections
        finally:
            lock.release()


t1 = threading.Thread(target=object_detect)
t1.start()

import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject


"""
Start the RTSP server
"""
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.cap = cv2.VideoCapture(0)
        self.number_frames = 0
        self.fps = 30
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width=1280,height=720,framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96'.format(self.fps)

    def on_need_data(self, src, lenght):
        global frame_od
        if self.cap.isOpened():
            # lock =
            ret = True
            # frame = frame_od
            # ret, frame = self.cap.read()
            if ret:
                try:
                    data = frame_od.tostring()
                    buf = Gst.Buffer.new_allocate(None, len(data), None)
                    buf.fill(0, data)
                    buf.duration = self.duration
                    timestamp = self.number_frames * self.duration
                    buf.pts = buf.dts = int(timestamp)
                    buf.offset = timestamp
                    self.number_frames += 1
                    retval = src.emit('push-buffer', buf)
                    # print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
                    #                                                                        self.duration,
                    #                                                                        self.duration / Gst.SECOND))
                    if retval != Gst.FlowReturn.OK:
                        print(retval)
                except:
                    pass

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/test", self.factory)
        self.attach(None)


GObject.threads_init()
Gst.init(None)

server = GstServer()

loop = GObject.MainLoop()
loop.run()

# while True:
#     print('Please choose which stream to be played:')
#     print('1. Pure RTSP')
#     print('2. RTSP with object detection')
#     print('3. Both streams')
#     choose_1 = input()
#
#     if choose_1 == "1":
#         break
#     elif choose_1 == "2":
#         break
#     elif choose_1 == "3":
#         break
#     else:
#         print("Please just type in 1, 2, or 3")


"""
based on the codes:
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/object_detection_camera.html#sphx-glr-auto-examples-object-detection-camera-py
https://github.com/davidvuong/gstreamer-test/blob/master/src/python/opencv-rtsp-server.py
"""
