import tensorflow as tf

file_name = '/home/teddy/tensorflow_face/train_result/vgg/vgg_model.ckpt-25000.meta'
para_file = '/home/teddy/tensorflow_face/train_result/vgg/vgg_model.ckpt-25000'
with tf.Graph().as_default() as import_gragh:
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(file_name, clear_devices=True)
        saver.restore(sess, para_file)
        for n in tf.get_default_graph().as_graph_def().node:
            print n.name

