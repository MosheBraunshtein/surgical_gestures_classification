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







def define_and_process_args():
    """ Define and process command-line arguments.

    Returns:
        A Namespace with arguments as attributes.
    """

    description = main.__doc__
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=formatter_class)

    parser.add_argument('--data_dir', default=r'C:\Users\win10\desktop\data\JIGSAWS\Needle_Passing',
                        help='Data directory.')
    parser.add_argument('--model_dir', default=r'C:\Users\win10\desktop\data\JIGSAWS\Suturing\logs\kappa_tau_lambda_gripper\BidirectionalLSTM_1_512_05_0.50_1.0000',
                        help='Data directory.')
    parser.add_argument('--data_filename', default='standardized_data_kappa_tau_lambda_gripper.pkl',
                        help='''The name of the standardized-data pkl file that
                                resides in data_dir.''')
    parser.add_argument('--features', default='kappa_gripper',
                        help='''A string of the users that make up the test set,
                                with users separated by spaces.''')
    parser.add_argument('--louo_user', default='B',
                        help='''A string of the user how wasn't the train set.''')

    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    return args


def get_log_dir(args):
    """ Form a convenient log directory that summarizes arguments.

    Args:
        args: An object containing processed arguments as attributes.

    Returns:
        A string. The full path to log directory.
    """

    # test_users_str = '_'.join(args.test_users)
    test_users_str = 'all_users'

    return os.path.join(args.model_dir, args.louo_user, 'test_on_Needle_Passing', test_users_str)

def get_data_pkl(args):
    """ 
     Args:
        args: An object containing processed arguments as attributes.

    Returns:
        A string. The full path to data directory.
    """

    return os.path.join(args.data_dir, args.data_filename)

def get_model_dir(args):
    """ 
    Args:
        args: An object containing processed arguments as attributes.

    Returns:
        A string. The full path to model pkl directory.
    """

    return os.path.join(args.model_dir,args.louo_user)

def main():
    
    args = define_and_process_args()
    print('\n', 'ARGUMENTS', '\n\n', args, '\n')

    log_dir = get_log_dir(args)
    print('\n', 'LOG DIRECTORY', '\n\n', log_dir, '\n')

    data_dir = get_data_pkl(args)
    print('\n', 'DATA DIRECTORY', '\n\n', data_dir, '\n')

    model_dir = get_model_dir(args)
    print('\n', 'MODEL DIRECTORY', '\n\n', model_dir, '\n')
    
    print(f'LOAD {data_dir}\n')
    standardized_data_path = os.path.join(args.data_dir, args.data_filename)
    if not os.path.exists(standardized_data_path):
        message = '%s does not exist.' % standardized_data_path
        raise ValueError(message)
    

    dataset = data.Dataset(standardized_data_path)
    test_users = ['B','C','D','E','F','H','I']

    print(f'test users: {test_users}\n')

    _, test_raw_seqs = dataset.get_splits(test_users)
    test_triplets = [data.prepare_raw_seq(seq) for seq in test_raw_seqs]

    test_input_seqs, test_reset_seqs, test_label_seqs = zip(*test_triplets)

    print("LOAD MODEL STRACTURE: ", model_dir, '\n' )
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
        print(f"Test Accuracy: {test_accuracy_:.6f}")
        print(f"Test Edit Distance: {test_edit_dist_:.3f}")
        print("-" * 30)


if __name__ == '__main__':
    
    main()
