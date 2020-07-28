#!/usr/bin/python3
import tensorflow as tf
print(tf.__version__)
import_dir = "./savedmodel"
export_dir = import_dir+"_signaturedefs"

def get_graph_def_from_saved_model(saved_model_dir):
    with tf.compat.v1.Session() as session:
        meta_graph_def = tf.compat.v1.saved_model.loader.load(
            session,
            tags=['serve'],
            export_dir=saved_model_dir
        )
    return meta_graph_def.graph_def

graph_def = get_graph_def_from_saved_model(import_dir)

input_nodes = ['sub_2']
output_nodes = ['float_segments' ] #, 'float_short_offsets']

with tf.compat.v1.Session(graph=tf.Graph()) as session:
    tf.compat.v1.import_graph_def(graph_def, name='')
    inputs = {input_node: session.graph.get_tensor_by_name(f'{input_node}:0') for input_node in input_nodes}
    outputs = {output_node: session.graph.get_tensor_by_name(f'{output_node}:0') for output_node in output_nodes}
    tf.compat.v1.saved_model.simple_save(
        session,
        export_dir,
        inputs=inputs,
        outputs=outputs
    )

model = tf.saved_model.load(export_dir)
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, 257, 257, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

with tf.io.gfile.GFile('bodypix.tflite', 'wb') as f:
  f.write(tflite_model)

