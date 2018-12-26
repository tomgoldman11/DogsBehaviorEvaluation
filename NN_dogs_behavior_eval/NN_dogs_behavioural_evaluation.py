# Imports
import getopt

import numpy as np
import csv
import os
import sys
import tensorflow as tf
import cv2
import time

# Video Capture
cap = None

# Env setup
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# # Model preparation

# Variables
label_path = ''
video_path = ''
graph_path = ''
csv_path = ''

MODEL_NAME = 'dogs_toys_graph2'

# Path to graph
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

NUM_CLASSES = 2

# Label map
label_map = None
categories = None
category_index = None

# Load graph

detection_graph = tf.Graph()


def load_graph():
    global detection_graph
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


# Detection
PATH_TO_TEST_IMAGES_DIR = 'imagesPresentation'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(1, 29)]

start_time = time.time()


# CSV Helpers
def open_csv_file(csv_path):
    csv_file = open(csv_path, mode='w', newline='')
    return csv_file


def close_csv_file(csv_file):
    csv_file.close()


def create_csv_file_writer(csv_file):
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(['time', 'dog to toy distance [image units]', 'interaction'])
    return csv_file_writer


def evaluate_dog_behaviour(box, clas, score, csv_writer):
    dog_coord = None
    toy_coord = None

    dog_box = None
    toy_box = None

    for i, b in enumerate(box):
        if score[i] < 0.5:
            continue

        # dog box
        if clas[i] == 1:
            dog_box = b
            dog_mid_x = ((b[1] + b[3]) / 2)
            dog_mid_y = ((b[0] + b[2]) / 2)
            dog_coord = np.array((dog_mid_x, dog_mid_y))

        # toy box
        if clas[i] == 2:
            toy_box = b
            toy_mid_x = ((b[1] + b[3]) / 2)
            toy_mid_y = ((b[0] + b[2]) / 2)
            toy_coord = np.array((toy_mid_x, toy_mid_y))

    # calculate euclid distance between toy and dog
    if dog_coord is not None and toy_coord is not None:
        intersects = do_boxes_intersect(toy_box, dog_box)
        distance = np.linalg.norm(dog_coord - toy_coord)
        if distance < 0.3 or intersects:
            interaction = 'V'
        else:
            interaction = ' '
        # save into csv
        video_time = time.time() - start_time
        csv_writer.writerow([video_time, distance, interaction])


def do_boxes_intersect(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    xI1 = max(x11, x21)
    xI2 = min(x12, x22)

    yI1 = max(y11, y21)
    yI2 = min(y12, y22)

    inter_area = max((xI2 - xI1), 0) * max((yI1 - yI2), 0)

    if inter_area:
        return True
    else:
        return False


def run_detection():
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Create a new CSV file
            csv_file = open_csv_file(csv_path)
            # Create a writer for the file
            csv_file_writer = create_csv_file_writer(csv_file)

            while True:
                ret, image_np = cap.read()
                image_np_expanded = np.expand_dims(image_np, axis=0)

                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # OPTIONAL VISUALIZATION
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                evaluate_dog_behaviour(
                    boxes[0],
                    classes[0],
                    scores[0],
                    csv_file_writer)

                cv2.imshow('object_detection', image_np)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    close_csv_file(csv_file)
                    break


def main():
    global cap
    global label_path
    global video_path
    global graph_path
    global csv_path
    global label_map
    global categories
    global category_index

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hg:o:v:l:", [])
    except getopt.GetoptError:
        print('test.py -g <graph .pb path> -o <output csv file path> '
              '-l <label map path> -v <video evaluation path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -g <graph .pb path> -o <output csv file path> '
                  '-l <label map path> -v <video evaluation path>')
            sys.exit()
        elif opt == "-g":
            graph_path = arg
        elif opt == "-o":
            csv_path = arg
        elif opt == "-v":
            cap = cv2.VideoCapture(arg)
        elif opt == "-l":
            label_path = arg

    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    load_graph()

    run_detection()


if __name__ == '__main__':
    main()
