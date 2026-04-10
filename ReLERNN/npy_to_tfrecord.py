#!/usr/bin/env python
"""
Converts ReLERNN simulation .npy files to TFRecord format for efficient training.
Run once after ReLERNN_SIMULATE.py.

Usage:
    pixi run python npy_to_tfrecord.py -d /path/to/projectDir
"""

from ReLERNN.imports import *
import tensorflow as tf


def convert_directory(treesDirectory, normalizedTargets):
    """Convert all .npy pairs in a directory to a single TFRecord file.
    
    Stores pre-processed data (normalized targets, raw haps/pos before
    padding) so that padding can still be applied dynamically in
    SequenceBatchGenerator based on maxLen, frameWidth, etc.
    """
    infoDir = pickle.load(open(os.path.join(treesDirectory, "info.p"), "rb"))
    numReps = infoDir["numReps"]
    outFile = os.path.join(treesDirectory, "data.tfrecord")

    print(f"Converting {numReps} records in {treesDirectory}...")
    with tf.io.TFRecordWriter(outFile) as writer:
        for i in range(numReps):
            haps = np.load(os.path.join(treesDirectory, f"{i}_haps.npy"))
            pos  = np.load(os.path.join(treesDirectory, f"{i}_pos.npy"))
            rho  = normalizedTargets[i]

            feature = {
                # Flatten arrays; store shapes to reconstruct
                'haps':       tf.train.Feature(float_list=tf.train.FloatList(value=haps.flatten().astype(np.float32))),
                'haps_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=haps.shape)),
                'pos':        tf.train.Feature(float_list=tf.train.FloatList(value=pos.flatten().astype(np.float32))),
                'pos_len':    tf.train.Feature(int64_list=tf.train.Int64List(value=[len(pos)])),
                'rho':        tf.train.Feature(float_list=tf.train.FloatList(value=[float(rho)])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    print(f"  Written to {outFile}")


def normalizeTargets(infoDir, norm='zscore'):
    """Replicate SequenceBatchGenerator.normalizeTargets() for standalone use."""
    nTargets = copy.deepcopy(infoDir['rho'])
    if norm == 'zscore':
        tar_mean = np.mean(nTargets, axis=0)
        tar_sd   = np.std(nTargets, axis=0)
        nTargets -= tar_mean
        nTargets  = np.divide(nTargets, tar_sd,
                              out=np.zeros_like(nTargets), where=tar_sd != 0)
    elif norm == 'divstd':
        tar_sd   = np.std(nTargets, axis=0)
        nTargets  = np.divide(nTargets, tar_sd,
                              out=np.zeros_like(nTargets), where=tar_sd != 0)
    return nTargets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--projectDir', dest='outDir', required=True,
                        help='Project directory (same as used for ReLERNN_SIMULATE)')
    parser.add_argument('--norm', dest='norm', default='zscore',
                        choices=['zscore', 'divstd'],
                        help='Target normalization method (default: zscore)')
    args = parser.parse_args()

    projectDir = args.outDir
    for split in ['train', 'vali', 'test']:
        treesDir = os.path.join(projectDir, split)
        infoDir  = pickle.load(open(os.path.join(treesDir, "info.p"), "rb"))
        normalizedTargets = normalizeTargets(infoDir, norm=args.norm)
        convert_directory(treesDir, normalizedTargets)

    print("\n***npy_to_tfrecord.py FINISHED!***\n")


if __name__ == "__main__":
    main()
