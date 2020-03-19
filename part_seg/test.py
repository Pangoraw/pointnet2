import argparse
import importlib
import os
import sys
import tensorflow as tf
import numpy as np
from matplotlib import cm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
import part_dataset_all_normal as part_dataset
import show3d_balls
output_dir = os.path.join(BASE_DIR, './test_results')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--category', default='Airplane', help='Which single class to train on [default: Airplane]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: pointnet2_part_seg]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
FLAGS = parser.parse_args()


MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model) # import network module
NUM_CLASSES = 50
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
TEST_DATASET = part_dataset.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test')

def output_color_point_cloud(data, seg, out_file, color_map):
    with open(out_file, 'w') as f:
        for i in range(len(seg)):
            color = color_map(seg[i])
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def get_model(batch_size, num_point):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl)#, end_points)
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss}
        return sess, ops

def inference(sess, ops, pc, batch_size):
    ''' pc: BxNx3 array, return BxN pred '''
    assert pc.shape[0]%batch_size == 0
    num_batches = pc.shape[0]/batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    for i in range(num_batches):
        feed_dict = {ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
                     ops['is_training_pl']: False}
        batch_logits = sess.run(ops['pred'], feed_dict=feed_dict)
        logits[i*batch_size:(i+1)*batch_size,...] = batch_logits
    return np.argmax(logits, 2)

if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    total = 0
    accurate = 0
    color_map = cm.get_cmap('viridis', NUM_CLASSES)

    SIZE = len(TEST_DATASET)
    for i in range(len(TEST_DATASET)):
        print(">>>> running sample " + str(i) + "/" + str(SIZE))

        ps, normal, seg = TEST_DATASET[i]
        ps = np.hstack((ps, normal))
        sess, ops = get_model(batch_size=1, num_point=ps.shape[0])
        segp = inference(sess, ops, np.expand_dims(ps, 0), batch_size=1)
        segp = segp.squeeze()

        total += segp.shape[0]
        accurate += np.sum(seg == segp)

        output_color_point_cloud(ps, seg, './test_results/gt_%d.obj' % (i), color_map)
        output_color_point_cloud(ps, segp, './test_results/pred_%d.obj' % (i), color_map)
        output_color_point_cloud(ps, segp == seg, './test_results/diff_%d.obj' % (i), lambda eq: (0,1,0) if eq else (1, 0, 0))

    print("Accuracy: %f" % (float(accurate) / float(total)))
