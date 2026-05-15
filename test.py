# --- בתוך main() ---

from __future__ import division, print_function

import os
import time
import argparse
import shutil
import _pickle as cPickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

import data
import models
import metrics



def test(standardized_data_path,model_dir,log_dir):
    
    dataset = data.Dataset(standardized_data_path)
    # test_users = ['B','C','D','E','F','H','I']
    test_users = ['D']
    print(f'test users: {test_users}\n')

    _, test_raw_seqs = dataset.get_splits(test_users)
    user = 5
    test_raw_seqs = [test_raw_seqs[user-1]]

    test_triplets = [data.prepare_raw_seq(seq) for seq in test_raw_seqs]

    test_input_seqs, test_reset_seqs, test_label_seqs = zip(*test_triplets)

    model_type = "BidirectionalLSTM"
    hidden_layer_size = 512
    num_layers = 1
    Model = eval('models.' + model_type + 'Model')
    
    input_size = dataset.input_size
    target_size = dataset.num_classes

    # This is just to satisfy a low-CPU requirement on our cluster
    # when using GPUs.
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        config = tf.ConfigProto(intra_op_parallelism_threads=2,
                                inter_op_parallelism_threads=2)
    else:
        config = None

    with tf.Session(config=config) as sess:
        droput_keep_prob = 1.0
        init_scale = 0.1
        print('LOAD MODEL PAREMETERS:\n' )
        
        model = Model(input_size, target_size, num_layers,
                      hidden_layer_size, init_scale,
                      droput_keep_prob)

        saver = tf.train.Saver()

        ckpt_path = tf.train.latest_checkpoint(model_dir)
        if ckpt_path:
            print(f"LOAD WEIGHTS from: {ckpt_path}")
            saver.restore(sess, ckpt_path)
        else:
            print("No checkpoint found.")
            exit(0)

        # בתוך ה-Session, אחרי ה-saver.restore:
        print("Running prediction on test sequences...")
        test_prediction_seqs = models.predict(sess, model, test_input_seqs, test_reset_seqs)
        
        print("Computing metrics...")
        test_accuracy_, test_edit_dist_ = metrics.compute_metrics(test_prediction_seqs, test_label_seqs)

        print("-" * 30)
        print(f"Test Accuracy: {test_accuracy_:.6f}\n")
        print(f"Test Edit Distance: {test_edit_dist_:.3f}")
        print("-" * 30)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created directory: {log_dir}")

        save_results_file = os.path.join(log_dir,'results.txt')
        with open(save_results_file, "w", encoding="utf-8") as file:
            file.write(f"Test Accuracy: {test_accuracy_:.6f}")
            file.write(f"Test Edit Distance: {test_edit_dist_:.3f}")


if __name__ == '__main__':
    
    user = "H"
    model_type = 'BidirectionalLSTM_1_512_05_0.50_1.0000'
    needle_passing_dir = r'C:\Users\win10\desktop\data\JIGSAWS\Needle_Passing_A'
    suturing_logs_dir = r'C:\Users\win10\desktop\data\JIGSAWS\Suturing\logs'

    features = ['GH_kappa_tau_lambda']
    for ft in features: 
        tf.reset_default_graph()

        log_dir = os.path.join(needle_passing_dir, 'test', ft, model_type, 'LOUO_' + user)
        print(f'LOG DIR: {log_dir}\n')

        data_dir = os.path.join(needle_passing_dir, ft + '.pkl')
        print(f'DATA DIR: {data_dir}\n')

        model_dir = os.path.join(suturing_logs_dir, ft, model_type, user)
        print(f'MODEL DIR: {model_dir}\n')
        print('\n')

        test(data_dir, model_dir, log_dir)
