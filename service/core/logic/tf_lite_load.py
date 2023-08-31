
import numpy as np
from tensorflow import lite as tf_lite
def object_detection(img_array):
    tflite_model_path = 'service\core\logic\pretrained_model.tflite'
    interpreter = tf_lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img_array,axis=0))
    interpreter.invoke()


    return interpreter.get_tensor(output_details[0]['index'])