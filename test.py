import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
from PIL import ImageFont
import time
from scipy.stats import norm
import matplotlib
import scipy

def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords

def draw_boxes(image, boxes, classes, scores, thickness=4):
    """Draw bounding boxes on the image"""
    
    # Note: Traffic light class should be in this order, 
    # because I defined them in this order before training my model. See the label_map.pbtxt.
    traffic_light = ['red','green','yellow']
    cmap = ImageColor.colormap
    COLOR_LIST = sorted([c for c in cmap.keys()])
    # Utilize PIL.ImageDraw.Draw module.
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        
        # Note: classes is a list starts from 1, not 0. See the label_map.pbtxt.
        class_id = int(classes[i])
        
        # Multiply the class_id by 10, to make the color more.
        color = COLOR_LIST[class_id*10] 
         
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)
        textsize = 20
        
        # Note: the path to *.tif should be specified in Linux system. Please modify the below line if you 
        # are not using Linux system.
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 28, encoding="unic")
        
        # Utilize Draw.text() func.
        draw.text((left, top), traffic_light[class_id-1]+'='+str(scores[i]*100)[0:4]+'%',font =font, fill='yellow')
    return draw

        
def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

def main():
    SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_TF1.4.0/frozen_inference_graph.pb'
    detection_graph = load_graph(SSD_GRAPH_FILE)


    # The input placeholder for the image.  
    # get_tensor_by_name` returns the Tensor with the associated name in the Graph.
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

    # The classification of the object (integer id).
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    image = Image.open('./traffic_light_img/sim_2.jpeg')
    image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

    with tf.Session(graph=detection_graph) as sess:                
        # Actual detection.
        (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes], 
                                        feed_dict={image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.8
        # Filter boxes with a confidence score less than `confidence_cutoff`, 
        boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = image.size
        box_coords = to_image_coords(boxes, height, width)

        # Each class with be represented by a differently colored box
        draw = draw_boxes(image, box_coords, classes,scores)
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        plt.imshow(image) 
        pix = np.array(image)
        scipy.misc.imsave('outfile.jpg', pix)

if __name__ == "__main__":
    main()
