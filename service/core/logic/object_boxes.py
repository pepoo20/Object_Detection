from service.core.logic.tf_lite_load import object_detection
import cv2 
import numpy as np
import tensorflow as tf

def load_and_prepare_image(image, H, W):
    img = cv2.resize(image, (H, W))
    image = tf.image.resize(image, [H, W])
    return img, image



def detect_objects(output, THRESH, classes):
    object_positions = tf.concat(
        [tf.where(output[..., 0] >= THRESH), tf.where(output[..., 5] >= THRESH)], axis=0)
    selected_output = tf.gather_nd(output, object_positions)
    return object_positions, selected_output


def calculate_bounding_boxes(object_positions, selected_output, H, W, classes,THRESH,B):
    final_boxes = []
    final_scores = []

    for i, pos in enumerate(object_positions):
        for j in range(B):
            if selected_output[i][j * 5] > THRESH:
                output_box = selected_output[i][j * 5 + 1: j * 5 + 5]

                x_centre = (tf.cast(pos[1], dtype=tf.float32) + output_box[0]) * 32
                y_centre = (tf.cast(pos[2], dtype=tf.float32) + output_box[1]) * 32
                x_width, y_height = tf.math.abs(H * output_box[2]), tf.math.abs(W * output_box[3])

                x_min, y_min = int(x_centre - (x_width / 2)), int(y_centre - (y_height / 2))
                x_max, y_max = int(x_centre + (x_width / 2)), int(y_centre + (y_height / 2))

                if x_min <= 0: x_min = 0
                if y_min <= 0: y_min = 0
                if x_max >= W: x_max = W
                if y_max >= H: y_max = H

                final_boxes.append(
                    [x_min, y_min, x_max, y_max, str(classes[tf.argmax(selected_output[..., 5*B:], axis=-1)[i]])])
                final_scores.append(selected_output[i][j * 5])


    return final_boxes, final_scores

def annotate_image(img, final_boxes, final_scores):
    final_boxes = np.array(final_boxes)

    object_classes = final_boxes[..., 4]
    nms_boxes = final_boxes[..., 0:4]

    nms_output = tf.image.non_max_suppression(
        nms_boxes, final_scores, max_output_size=100, iou_threshold=0.1,
        score_threshold=float('-inf')
    )
    labels = []
    for i in nms_output:
        label = []
        cv2.rectangle(
            img,
            (int(final_boxes[i][0]), int(final_boxes[i][1])),
            (int(final_boxes[i][2]), int(final_boxes[i][3])), (0, 0, 255), 1)
        cv2.putText(
            img,
            final_boxes[i][-1],
            (int(final_boxes[i][0]), int(final_boxes[i][1]) + 15),
            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (2, 225, 155), 1
        )
        labels.append(final_boxes[i,:])
    return img,np.array(labels)


# def model_predict(image, THRESH = 0.15, H=224, W=224):
#     classes=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',
#          'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
#     img, image = load_and_prepare_image(image,H,W)

#     output = object_detection(image)

#     object_positions, selected_output = detect_objects(output, THRESH, classes)

#     final_boxes, final_scores = calculate_bounding_boxes(object_positions, selected_output, H, W, classes,THRESH,B=2)

#     annotated_img,labels = annotate_image(img, final_boxes, final_scores)
#     return cv2.resize(annotated_img,(H,W)),labels

def model_predict(image, THRESH = 0.15, H=224, W=224):
    classes=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',
        'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    img, image = load_and_prepare_image(image,H,W)

    output =object_detection(image)
    object_positions, selected_output = detect_objects(output, THRESH, classes)

    final_boxes, final_scores = calculate_bounding_boxes(object_positions, selected_output, H, W, classes,THRESH,B=2)

    annotated_img,labels = annotate_image(img, final_boxes, final_scores)
    return annotated_img,labels
