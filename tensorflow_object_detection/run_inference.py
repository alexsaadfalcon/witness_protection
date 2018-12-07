import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


flags = tf.app.flags
flags.DEFINE_string('checkpoint', './inference_graph/frozen_inference_graph.pb',
                    'Path to checkpoint.')
flags.DEFINE_string('label_map', './wider_label_map.pbtxt',
                    'Path to label map pbtxt.')
flags.DEFINE_integer('num_classes', '1',
                 'Number of classes in dataset.')
flags.DEFINE_string('image_path', '',
                    'Path to test image.')
FLAGS = flags.FLAGS


def main(_):
    label_map = label_map_util.load_labelmap(FLAGS.label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=FLAGS.num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image = cv2.imread(FLAGS.image_path)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.70)

    valid_scores = sum([1 for i in np.squeeze(scores) if i > .70])
    valid_boxes = np.squeeze(boxes)[0:7, :]
    print(valid_scores)
    print('Num detection: {}'.format(num))
    print('Scores: {}'.format(scores))
    print('Bounding boxes: {}'.format(valid_boxes))
    print('Bounding boxes shape: {}'.format(boxes.shape))
    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', image)

    # Press any key to close the image
    cv2.waitKey(0)

    # Clean up
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.run()
