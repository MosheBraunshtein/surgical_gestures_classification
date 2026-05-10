from __future__ import division, print_function

import os
import argparse
import itertools

import numpy as np
import _pickle as cPickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data
from compute_geometric_features import compute_kappa, compute_tau, compute_lambda



# Standard JIGSAWS experimental setup. In particular, it's the only
# recognition setup that exists, corresponding to
# JIGSAWS/Experimental/Suturing/unBalanced/GestureRecognition/UserOut
# (User H's 2nd trial is left out because no video was available for labeling.)

def load_kinematics_and_new_labels_for_GH(data_dir, trial_name):
    """ Load kinematics data and labels.

    Args:
        data_dir: A string.
        trial_name: A string.

    Returns:
        A 2-D NumPy array with time on the first axis. Labels are appended
        as a new column to the raw kinematics data (and are therefore
        represented as floats).
    """
    print('process GH set ...')
    labels_dir = os.path.join(data_dir, 'transcriptions')
    labels_path = os.path.join(labels_dir, trial_name + '.txt')

    seq_len, (psm1_pos, psm1_vel, _, psm2_pos, psm2_vel, _, psm1_gripper, psm2_gripper) = load_kinematics(data_dir, trial_name)
    labels_data = load_labels(labels_path, seq_len)

    data = np.concatenate([psm1_pos, psm1_vel, psm2_pos, psm2_vel, psm1_gripper, psm2_gripper, labels_data], axis=1)

    labeled_data_only_mask = labels_data.flatten() != 0

    return data[labeled_data_only_mask, :]


def load_kinematics_and_new_labels_for_GH_kappa(data_dir, trial_name):
    print('process GH_kappa set ...')
    labels_dir = os.path.join(data_dir, 'transcriptions')
    labels_path = os.path.join(labels_dir, trial_name + '.txt')

    seq_len, (psm1_pos, psm1_vel, _, psm2_pos, psm2_vel, _, psm1_gripper, psm2_gripper) = load_kinematics(data_dir, trial_name)
    labels_data = load_labels(labels_path, seq_len)

    psm1_kappa = compute_kappa(psm1_pos, s=5)
    psm2_kappa = compute_kappa(psm2_pos, s=5)

    data = np.concatenate([psm1_pos, psm1_vel, psm1_kappa, psm2_pos, psm2_vel, psm2_kappa, psm1_gripper, psm2_gripper, labels_data], axis=1)

    labeled_data_only_mask = labels_data.flatten() != 0

    return data[labeled_data_only_mask, :]

def load_kinematics_and_new_labels_for_GH_kappa_tau(data_dir, trial_name):
    print('process GH_kappa_tau set ...')
    labels_dir = os.path.join(data_dir, 'transcriptions')
    labels_path = os.path.join(labels_dir, trial_name + '.txt')

    seq_len, (psm1_pos, psm1_vel, _, psm2_pos, psm2_vel, _, psm1_gripper, psm2_gripper) = load_kinematics(data_dir, trial_name)
    labels_data = load_labels(labels_path, seq_len)

    psm1_kappa, psm2_kappa = compute_kappa(psm1_pos, s=5), compute_kappa(psm2_pos, s=5)
    psm1_tau, psm2_tau = compute_tau(psm1_pos, s=5), compute_tau(psm2_pos, s=5)

    data = np.concatenate([psm1_pos, psm1_vel, psm1_kappa, psm1_tau, psm2_pos, psm2_vel, psm2_kappa, psm2_tau, psm1_gripper, psm2_gripper, labels_data], axis=1)

    labeled_data_only_mask = labels_data.flatten() != 0

    return data[labeled_data_only_mask, :]

def load_kinematics_and_new_labels_for_GH_kappa_lambda(data_dir, trial_name):
    print('process GH_kappa_lambda set ...')
    labels_dir = os.path.join(data_dir, 'transcriptions')
    labels_path = os.path.join(labels_dir, trial_name + '.txt')

    seq_len, (psm1_pos, psm1_vel, psm1_R, psm2_pos, psm2_vel, psm2_R, psm1_gripper, psm2_gripper) = load_kinematics(data_dir, trial_name)
    labels_data = load_labels(labels_path, seq_len)

    psm1_kappa, psm2_kappa = compute_kappa(psm1_pos, s=5)[1:-1,:], compute_kappa(psm2_pos, s=5)[1:-1,:]
    psm1_lambda, psm2_lambda = compute_lambda(psm1_R), compute_lambda(psm2_R)

    labels_data = labels_data[1:-1,:]
    psm1_pos = psm1_pos[1:-1,:]
    psm2_pos = psm2_pos[1:-1,:]
    psm1_vel = psm1_vel[1:-1,:]
    psm2_vel = psm2_vel[1:-1,:]
    psm1_gripper = psm1_gripper[1:-1,:]
    psm2_gripper = psm2_gripper[1:-1,:]

    data = np.concatenate([psm1_pos, psm1_vel, psm1_kappa, psm1_lambda, psm2_pos, psm2_vel, psm2_kappa, psm2_lambda, psm1_gripper, psm2_gripper, labels_data], axis=1)

    labeled_data_only_mask = labels_data.flatten() != 0

    return data[labeled_data_only_mask, :]
    
def load_kinematics_and_new_labels_for_GH_kappa_tau_lambda(data_dir, trial_name):
    print('process GH_kappa_tau_lambda set ...')
    labels_dir = os.path.join(data_dir, 'transcriptions')
    labels_path = os.path.join(labels_dir, trial_name + '.txt')

    seq_len, (psm1_pos, psm1_vel, psm1_R, psm2_pos, psm2_vel, psm2_R, psm1_gripper, psm2_gripper) = load_kinematics(data_dir, trial_name)
    labels_data = load_labels(labels_path, seq_len)

    psm1_kappa, psm2_kappa = compute_kappa(psm1_pos, s=5)[1:-1,:], compute_kappa(psm2_pos, s=5)[1:-1,:]
    psm1_tau, psm2_tau = compute_tau(psm1_pos, s=5)[1:-1,:], compute_tau(psm2_pos, s=5)[1:-1,:]

    psm1_lambda, psm2_lambda = compute_lambda(psm1_R), compute_lambda(psm2_R)

    labels_data = labels_data[1:-1,:]
    psm1_pos = psm1_pos[1:-1,:]
    psm2_pos = psm2_pos[1:-1,:]
    psm1_vel = psm1_vel[1:-1,:]
    psm2_vel = psm2_vel[1:-1,:]
    psm1_gripper = psm1_gripper[1:-1,:]
    psm2_gripper = psm2_gripper[1:-1,:]

    data = np.concatenate([psm1_pos, psm1_vel, psm1_kappa, psm1_tau, psm1_lambda, psm2_pos, psm2_vel, psm2_kappa, psm2_tau, psm2_lambda, psm1_gripper, psm2_gripper, labels_data], axis=1)

    labeled_data_only_mask = labels_data.flatten() != 0
    
    return data[labeled_data_only_mask, :]

def load_kinematics_and_new_labels_for_kappa_gripper(data_dir, trial_name):
    print('process kappa_gripper set ...')
    labels_dir = os.path.join(data_dir, 'transcriptions')
    labels_path = os.path.join(labels_dir, trial_name + '.txt')

    seq_len, (psm1_pos, _, _, psm2_pos, _, _, psm1_gripper, psm2_gripper) = load_kinematics(data_dir, trial_name)
    labels_data = load_labels(labels_path, seq_len)

    psm1_kappa, psm2_kappa = compute_kappa(psm1_pos, s=5), compute_kappa(psm2_pos, s=5)

    data = np.concatenate([psm1_kappa, psm2_kappa, psm1_gripper, psm2_gripper, labels_data], axis=1)

    labeled_data_only_mask = labels_data.flatten() != 0
    
    return data[labeled_data_only_mask, :]

def load_kinematics_and_new_labels_for_kappa_tau_gripper(data_dir, trial_name):
    print('process kappa_tau_gripper set ...')
    labels_dir = os.path.join(data_dir, 'transcriptions')
    labels_path = os.path.join(labels_dir, trial_name + '.txt')

    seq_len, (psm1_pos, _, _, psm2_pos, _, _, psm1_gripper, psm2_gripper) = load_kinematics(data_dir, trial_name)
    labels_data = load_labels(labels_path, seq_len)

    psm1_kappa, psm2_kappa = compute_kappa(psm1_pos, s=5), compute_kappa(psm2_pos, s=5)
    psm1_tau, psm2_tau = compute_tau(psm1_pos, s=5), compute_tau(psm2_pos, s=5)

    data = np.concatenate([psm1_kappa, psm1_tau, psm2_kappa, psm2_tau, psm1_gripper, psm2_gripper, labels_data], axis=1)

    labeled_data_only_mask = labels_data.flatten() != 0
    
    return data[labeled_data_only_mask, :]

def load_kinematics_and_new_labels_for_kappa_lambda_gripper(data_dir, trial_name):
    print('process kappa_lambda_gripper set ...')
    labels_dir = os.path.join(data_dir, 'transcriptions')
    labels_path = os.path.join(labels_dir, trial_name + '.txt')

    seq_len, (psm1_pos, psm1_vel, psm1_R, psm2_pos, psm2_vel, psm2_R, psm1_gripper, psm2_gripper) = load_kinematics(data_dir, trial_name)
    labels_data = load_labels(labels_path, seq_len)

    psm1_kappa, psm2_kappa = compute_kappa(psm1_pos, s=5)[1:-1,:], compute_kappa(psm2_pos, s=5)[1:-1,:]

    psm1_lambda, psm2_lambda = compute_lambda(psm1_R), compute_lambda(psm2_R)

    labels_data = labels_data[1:-1,:]

    data = np.concatenate([psm1_kappa, psm1_lambda, psm2_kappa, psm2_lambda, psm1_gripper, psm2_gripper, labels_data], axis=1)

    labeled_data_only_mask = labels_data.flatten() != 0
    
    return data[labeled_data_only_mask, :]

def load_kinematics_and_new_labels_for_kappa_tau_lambda_gripper(data_dir, trial_name):
    print('process kappa_tau_lambda_gripper set ...')
    labels_dir = os.path.join(data_dir, 'transcriptions')
    labels_path = os.path.join(labels_dir, trial_name + '.txt')

    seq_len, (psm1_pos, _, psm1_R, psm2_pos, _, psm2_R, psm1_gripper, psm2_gripper) = load_kinematics(data_dir, trial_name)
    labels_data = load_labels(labels_path, seq_len)

    psm1_kappa, psm2_kappa = compute_kappa(psm1_pos, s=5)[1:-1,:], compute_kappa(psm2_pos, s=5)[1:-1,:]
    psm1_tau, psm2_tau = compute_tau(psm1_pos, s=5)[1:-1,:], compute_tau(psm2_pos, s=5)[1:-1,:]
    psm1_lambda, psm2_lambda = compute_lambda(psm1_R), compute_lambda(psm2_R)

    labels_data = labels_data[1:-1,:]

    data = np.concatenate([psm1_kappa, psm1_lambda, psm1_tau, psm2_kappa, psm2_lambda, psm2_tau, psm1_gripper, psm2_gripper, labels_data], axis=1)

    labeled_data_only_mask = labels_data.flatten() != 0
    
    return data[labeled_data_only_mask, :]

features = {
    "standardized_data_GH": (['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'gripper']*2, load_kinematics_and_new_labels_for_GH),
    "standardized_data_GH_kappa":(['pos_x', 'pos_y', 'pos_z', 'kappa', 'vel_x','vel_y', 'vel_z', 'gripper']*2, load_kinematics_and_new_labels_for_GH_kappa),
    "standardized_data_GH_kappa_tau":(['pos_x', 'pos_y', 'pos_z', 'kappa', 'tau', 'vel_x', 'vel_y', 'vel_z', 'gripper']*2, load_kinematics_and_new_labels_for_GH_kappa_tau),
    "standardized_data_GH_kappa_lambda":(['pos_x', 'pos_y', 'pos_z', 'kappa', 'lambda', 'vel_x','vel_y', 'vel_z', 'gripper']*2, load_kinematics_and_new_labels_for_GH_kappa_lambda),
    "standardized_data_GH_kappa_tau_lambda":(['pos_x', 'pos_y', 'pos_z', 'kappa', 'tau', 'lambda', 'vel_x','vel_y', 'vel_z', 'gripper']*2, load_kinematics_and_new_labels_for_GH_kappa_tau_lambda),
    "standardized_data_kappa_gripper":(['kappa', 'gripper',]*2, load_kinematics_and_new_labels_for_kappa_gripper),
    "standardized_data_kappa_tau_gripper":(['kappa', 'tau', 'gripper',]*2, load_kinematics_and_new_labels_for_kappa_tau_gripper),
    "standardized_data_kappa_lambda_gripper":(['kappa', 'lambda', 'gripper',]*2, load_kinematics_and_new_labels_for_kappa_lambda_gripper),
    "standardized_data_kappa_tau_lambda_gripper":(['kappa', 'tau', 'lambda', 'gripper']*2, load_kinematics_and_new_labels_for_kappa_tau_lambda_gripper)
    }

USER_TO_TRIALS = {
    'Suturing': {
        'B': [1, 2, 3, 4, 5],
        'C': [1, 2, 3, 4, 5],
        'D': [1, 2, 3, 4, 5],
        'E': [1, 2, 3, 4, 5],
        'F': [1, 2, 3, 4, 5],
        'G': [1, 2, 3, 4, 5],
        'H': [1,    3, 4, 5],
        'I': [1, 2, 3, 4, 5]
    },
    'Needle_Passing': {
        'B': [1, 2, 3, 4   ],
        'C': [1, 2, 3, 4, 5],
        'D': [1, 2, 3, 4, 5],
        'E': [1,    3, 4, 5],
        'F': [1,    3, 4   ],
        'H': [   2,    4, 5],
        'I': [   2, 3, 4, 5]
    }
} 

FEATURES_SET = "standardized_data_GH_kappa_tau_lambda"

DATASET_NAME = 'JIGSAWS'
TASK = 'Needle_Passing' # Suturing / Needle_Passing
ORIG_CLASS_IDS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11] # suturing # needle passing * JUST FEW G9,G10
NEW_CLASS_IDS = range(len(ORIG_CLASS_IDS))
CLASSES = ['G%d' % id for id in ORIG_CLASS_IDS]
NUM_CLASSES = len(CLASSES)

ALL_USERS = sorted(USER_TO_TRIALS[TASK].keys())

KINEMATICS_COL_NAMES, load_kinematics_and_labels = features[FEATURES_SET]

KINEMATICS_USECOLS = [c-1 for c in [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 57,
                                            58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 76]]

LABELS_USECOLS = [0, 1, 2]
LABELS_COL_NAMES = ['start_frame', 'end_frame', 'string_label']
LABELS_CONVERTERS = {2: lambda x: int(x.decode().replace('G', '')) if isinstance(x, bytes) else int(x.replace('G', ''))}
STANDARDIZED_COL_NAMES = KINEMATICS_COL_NAMES + ['label']

data_dir = fr'C:\Users\win10\desktop\data\JIGSAWS\{TASK}'
data_filename = FEATURES_SET + '.pkl'

def get_trial_name(user, trial):
    """ Form a trial name that matches standard JIGSAWS filenames.

    Args:
        user: A string.
        trial: An integer.

    Returns:
        A string.
    """
    return '%s_%s%03d' % (TASK, user, trial)


def load_kinematics(data_dir, trial_name):
    """ Load kinematics data.

    Args:
        data_dir: A string.
        trial_name: A string.

    Returns:
        A integer

        A 2-D NumPy array with time on the first axis: 
        psm1_pos, psm1_vel, psm2_pos, psm2_vel, psm1_gripper, psm2_gripper
    """

    kinematics_dir = os.path.join(data_dir, 'kinematics', 'AllGestures')
    kinematics_path = os.path.join(kinematics_dir, trial_name + ".txt")
    data = np.loadtxt(kinematics_path, dtype=np.float,
                      usecols=KINEMATICS_USECOLS)
    seq_len = data.shape[0]

    # PSM1
    psm1_pos = data[:, 0:3]
    psm1_R = data[:, 3:12]
    psm1_vel = data[:, 12:15]
    psm1_gripper = data[:, 15].reshape(-1, 1)

    # PSM2
    psm2_pos = data[:, 16:19]
    psm2_R = data[:, 19:28]
    psm2_vel = data[:, 28:31]
    psm2_gripper = data[:, 31].reshape(-1, 1)

    return seq_len, (psm1_pos, psm1_vel, psm1_R, psm2_pos, psm2_vel, psm2_R, psm1_gripper, psm2_gripper)


def load_labels(labels_path,time_duration):
    """ Load labels.

    Args:
        labels_path: A string.
        time_duration: An integer.


    Returns:
        A 2-D NumPy array with time on the first axis.
    """
    raw_labels_data = np.genfromtxt(labels_path, dtype=np.int,
                                    converters=LABELS_CONVERTERS,
                                    usecols=LABELS_USECOLS)
    frames = np.arange(1, time_duration+1, dtype=np.int)
    labels = np.zeros(frames.shape, dtype=np.int)
    for start, end, label in raw_labels_data:
        mask = (frames >= start) & (frames <= end)
        labels[mask] = label
    labels_data = labels.reshape(-1, 1)

    return labels_data

def load_kinematics_and_new_labels(data_dir, trial_name):
    """ Load kinematics data and standardized labels.

    Args:
        data_dir: A string.
        trial_name: A string.

    Returns:
        A 2-D NumPy array with time on the first axis. Labels are appended as
        a new column and are converted from arbitrary labels (e.g., 1, 3, 5)
        to ordered, nonnegative integers (e.g., 0, 1, 2).
    """
    data = load_kinematics_and_labels(data_dir, trial_name)
    for orig, new in zip(ORIG_CLASS_IDS, NEW_CLASS_IDS):
        mask = data[:, -1] == orig
        data[mask, -1] = new
    return data


def downsample(data, factor=6):
    """ Downsample a data matrix.

    Args:
        data: A 2-D NumPy array with time on the first axis.
        factor: The factor by which we'll downsample.

    Returns:
        A 2-D NumPy array.
    """
    return data[::factor, :]


if __name__ == '__main__':
    """ Create a standardized data file from raw data. """

    print(f'Standardizing JIGSAWS:{TASK}..\n')
    print()

    print('%d classes:' % NUM_CLASSES)
    print(CLASSES)
    print()

    user_to_trial_names = {}
    for user, trials in USER_TO_TRIALS[TASK].items():
        user_to_trial_names[user] = [get_trial_name(user, trial)
                                     for trial in trials]
    print('Users and corresponding trial names:')
    for user in ALL_USERS:
        print(user, '   ', user_to_trial_names[user])
    print()


    all_trial_names = sorted(list(
        itertools.chain(*user_to_trial_names.values())
    ))
    print('All trial names, sorted:')
    print(all_trial_names)
    print()

    # Original data is at 30 Hz.
    all_data = {trial_name: downsample(
                    load_kinematics_and_new_labels(data_dir, trial_name),
                    factor=6)
                for trial_name in all_trial_names}
    print('Downsampled to 5 Hz.')
    print()

    fig, ax_list = plt.subplots(nrows=len(all_data), ncols=1,
                                sharex=True, figsize=(15, 50))
    for ax, (trial_name, data_mat) in zip(ax_list, sorted(all_data.items())):
        plt.sca(ax)
        data.plot_label_seq(data_mat[:, -1:], NUM_CLASSES, 0)
        plt.title(trial_name)
    plt.tight_layout()
    vis_path = os.path.join(data_dir, 'standardized_data_labels.png')
    plt.savefig(vis_path)
    plt.close(fig)
    print('Saved label visualization to %s.' % vis_path)
    print()

    export_dict = dict(
        dataset_name=DATASET_NAME, classes=CLASSES, num_classes=NUM_CLASSES,
        col_names=STANDARDIZED_COL_NAMES, all_users=ALL_USERS,
        user_to_trial_names=user_to_trial_names,
        all_trial_names=all_trial_names, all_data=all_data)
    standardized_data_path = os.path.join(data_dir, data_filename)
    with open(standardized_data_path, 'wb') as f:
        cPickle.dump(export_dict, f)
    print('Saved standardized data file %s.' % standardized_data_path)
    print()