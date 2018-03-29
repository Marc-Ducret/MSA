import tensorflow as tf
from keras import backend as K

def use_gpu(flag):
    num_cores = 4
    num_CPU = 1
    num_GPU = 1 if flag else 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)
    return session
