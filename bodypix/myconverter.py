#!/usr/bin/python3
import tensorflow as tf
import tfjs_graph_converter.api as tfjs

print(tf.__version__)
export_dir = "./savedmodel"

def get_graph_def_from_saved_model(saved_model_dir):
    with tf.compat.v1.Session() as session:
        meta_graph_def = tf.compat.v1.saved_model.loader.load(
            session,
            tags=['serve'],
            export_dir=saved_model_dir
        )
    return meta_graph_def.graph_def

graph_def = get_graph_def_from_saved_model(export_dir)

input_nodes = ['sub_2']
output_nodes = ['float_segments' ] #, 'float_short_offsets']

with tf.compat.v1.Session(graph=tf.Graph()) as session:
    tf.compat.v1.import_graph_def(graph_def, name='')
    inputs = {input_node: session.graph.get_tensor_by_name(f'{input_node}:0') for input_node in input_nodes}
    outputs = {output_node: session.graph.get_tensor_by_name(f'{output_node}:0') for output_node in output_nodes}
    tf.compat.v1.saved_model.simple_save(
        session,
        export_dir+'_signaturedefs',
        inputs=inputs,
        outputs=outputs
    )

