import tensorflow as tf




def convert(model, conversion_type='float32', model_name=None):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    name_model_tflite = 'srgan.tflite'
    tflite_model_file = model_dir.joinpath(name_model_tflite)                          
    tflite_model_file.write_bytes(tflite_model)