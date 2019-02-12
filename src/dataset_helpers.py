import numpy as np
import SimpleITK as sitk
import os
import glob
import pandas as pd
import tensorflow as tf

datasets_path_pattern = {
    'brats2015-Test-all': '../../datasets/BRATS2015/Testing/*/*/*/*.mha',
    'brats2015-Train-all': '../../datasets/BRATS2015/BRATS2015_Training/*/*/*/*.mha',
    'BD2Decide-T1T2': '../../datasets/BD2Decide/T1T2/*/*.mha'
}
output_stat_path = '../datasets/meta/'

# Description on the features contained in the .tfrecord dataset
feature_description = {
                        'mri_raw': tf.FixedLenFeature([], tf.string),
                        'seg_raw': tf.FixedLenFeature([], tf.string),
                        'z_slice': tf.FixedLenFeature([], tf.int64),
                        'mri_type': tf.FixedLenFeature([], tf.string),
                        'mri_z_dimension' : tf.FixedLenFeature([], tf.int64),
                        'mri_x_dimension' : tf.FixedLenFeature([], tf.int64),
                        'mri_y_dimension' : tf.FixedLenFeature([], tf.int64),
                        'mri_z_origin' : tf.FixedLenFeature([], tf.int64),
                        'mri_x_origin' : tf.FixedLenFeature([], tf.int64),
                        'mri_y_origin' : tf.FixedLenFeature([], tf.int64),
                        'mri_z_spacing' : tf.FixedLenFeature([], tf.float32),
                        'mri_x_spacing' : tf.FixedLenFeature([], tf.float32),
                        'mri_y_spacing' : tf.FixedLenFeature([], tf.float32),
                        'path' : tf.FixedLenFeature([], tf.string),
                        'patient_grade' : tf.FixedLenFeature([], tf.string),
                        'location' : tf.FixedLenFeature([], tf.string),
                        'patient' : tf.FixedLenFeature([], tf.string),
                        'patient_mri_seq': tf.FixedLenFeature([], tf.string),
                        'sample_number' : tf.FixedLenFeature([], tf.string)
                      }

def parse_meta(file_path):
    '''
    Return a dict of metadata containing the following indices (if available):
    'patient_grade': grade of patient condition
    'dataset_version': version of the dataset the sample first appeared
    'patient_number': code of the patient in the current dataset
    'patient_mri_seq': sequence number of the sample for the current patient. It's 1 if this is the first sample for the patient.
    'location': Area that the sample represents
    'mri_type': Sequencing type of the MRI
    'dataset_split': Wether if the sample belong to training or testing dataset. Unknown if the dataset hasn't been splitted already
    :param file_path: Path of a single .mha file. has to respect dataset_path_pattern
    :return:
    '''
    meta = dict()
    # Parsing meta from paths
    if 'BRATS2015' in file_path:
        splitted = file_path.split('/')
        meta['dataset_name'] = 'BRATS2015'
        meta['patient_grade'] = 'HIGH' if splitted[-4]=='HGG' else 'LOW' if splitted[-4]=='LGG' else 'unknown'
        _, meta['dataset_version'], meta['patient'], meta['patient_mri_seq'] = splitted[-3].split("_")
        _, meta['location'], _, _, meta['mri_type'], meta['sample_number'], ext = splitted[-1].split(".")
        meta['dataset_split'] = 'training' if 'Training' in file_path else 'testing' if 'Testing' in file_path else 'unknown'
        meta['path'] = file_path
    elif 'BD2Decide' in file_path:
        splitted = file_path.split('/')
        meta['dataset_name'] = 'BD2Decide'
        meta['patient_grade'] = 'unknown'
        meta['dataset_version'] = 'BD2Decide'
        meta['patient'] = splitted[-2]
        meta['patient_mri_seq'] = splitted[-2].split("_")[-1]
        meta['location'] = "Head-Neck"
        meta['sample_number'] = 'unknown'
        meta['mri_type'] = splitted[-1].split('.')[0]
        meta['dataset_split'] = 'training' if 'Training' in file_path else 'testing' if 'Testing' in file_path else 'unknown'
        meta['path'] = file_path
    else:
        raise NotImplementedError("Unknown dataset. Please implement how to extract information from the file_path")
    # Parsing meta from .mha
    image, origin, spacing = load_itk(file_path)
    meta['z_dimension'] = image.shape[0]
    meta['x_dimension'] = image.shape[1]
    meta['y_dimension'] = image.shape[2]
    meta['z_origin'] = origin[0]
    meta['x_origin'] = origin[1]
    meta['y_origin'] = origin[2]
    meta['z_spacing'] = spacing[0]
    meta['x_spacing'] = spacing[1]
    meta['y_spacing'] = spacing[2]

    return meta, image, origin, spacing


def load_itk(filename):
    ''' Read an .mha image and returns a numpy array, the origin coordinates and the voxel sizes in mm '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,x,y
    image = sitk.GetArrayFromImage(itkimage)
    # Here we have (z, y, x).
    image = image.transpose((0, 2, 1))

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(itkimage.GetOrigin())[[2,0,1]]

    # Read the spacing along each dimension
    spacing = np.array(itkimage.GetSpacing())[[2,0,1]]

    return image, origin, spacing

def compute_dataset_statistics(path_pattern, name):
    file_paths = sorted(glob.glob(path_pattern))
    meta = [parse_meta(path)[0] for path in file_paths]
    os.makedirs(output_stat_path, exist_ok=True)
    pd.DataFrame(meta).to_csv(output_stat_path+name+'.csv')

    pass

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if type(value) == str:
      value = value.encode('utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value, flatten=False):
  """Returns an int64_list from a bool / enum / int / uint."""
  if flatten:
      value = value.flatten()
  else:
      value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def pack_dataset(dataset_name):
    # Code adapted from https://www.tensorflow.org/tutorials/load_data/tf_records


    file_paths = sorted(glob.glob(datasets_path_pattern[dataset_name]))
    def get_one_sample():
        for f, fp in enumerate(file_paths):
            print("Parsing sample {} of {}".format(f, len(file_paths)))
            if 'brats2015' in dataset_name.lower():
                # Finding the target given a filename
                meta, mri, origin, spacing = parse_meta(fp)
                if meta['mri_type'] != "OT": # if not ground truth...
                    # ...search ground truth
                    gt_path = glob.glob(os.path.dirname(os.path.dirname(fp)) + '/*/*OT*.mha')[0]
                    label_mri, label_origin, label_spacing = load_itk(gt_path)
                    # Checking that the MRI properties are compatible to those of the segmentation
                    #assert all(label_origin == origin),   "MRI {} and Label {} has different origin: {} vs {}".format(fp, gt_path, origin, label_origin)
                    assert all(label_spacing == spacing), "MRI {} and Label {} has different spacing: {} vs {}".format(fp, gt_path, spacing, label_spacing)
                    assert label_mri.shape == mri.shape,  "MRI {} and Label {} has different size: {} vs {}".format(fp, gt_path, mri.shape, label_mri.shape)
                    # Iterating over the z dimension of the MRI/Segmentation to produce <mri_z_dimension> couples (mri, gt)
                    for z in range(meta['z_dimension']):
                        # Serializing data
                        features = {
                            'mri_raw': _bytes_feature(mri[z, ...].astype(np.uint16).tobytes()),
                            'seg_raw': _bytes_feature(label_mri[z, ...].astype(np.uint16).tobytes()),
                            'z_slice': _int64_feature(z),
                            'mri_type': _bytes_feature(meta['mri_type']),
                            'mri_z_dimension': _int64_feature(int(meta['z_dimension'])),
                            'mri_x_dimension': _int64_feature(int(meta['x_dimension'])),
                            'mri_y_dimension': _int64_feature(int(meta['y_dimension'])),
                            'mri_z_origin': _int64_feature(int(meta['z_origin'])),
                            'mri_x_origin': _int64_feature(int(meta['x_origin'])),
                            'mri_y_origin': _int64_feature(int(meta['y_origin'])),
                            'mri_z_spacing': _float_feature(meta['z_spacing']),
                            'mri_x_spacing': _float_feature(meta['x_spacing']),
                            'mri_y_spacing': _float_feature(meta['y_spacing']),
                            'path': _bytes_feature(fp),
                            'patient_grade': _bytes_feature(meta['patient_grade']),
                            'location': _bytes_feature(meta['location']),
                            'patient': _bytes_feature(meta['patient']),
                            'patient_mri_seq': _bytes_feature(meta['patient_mri_seq']),
                            'sample_number': _bytes_feature(meta['sample_number'])
                        }
                        yield tf.train.Example(features=tf.train.Features(feature=features))
    record_generator = get_one_sample()
    outpath = '../datasets/{}.tfrecords'.format(dataset_name)

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(outpath, options=options) as writer:
        print("Writing samples to {}...".format(outpath))
        for sample in record_generator:
            writer.write(sample.SerializeToString())
        print("Samples written to {}")


def load_dataset(name, mri_type=None, batch_size=32, cache=True, buffer_size=2048, interleave=1, cast_to=tf.float32, only_nonempty_labels=True, clip_labels_to=0.0, take_only=None, shuffle=True):
    '''
    Load a tensorflow dataset <name> (see definition in dataset_helpers).
    :param name: Name of the dataset
    :param mri_type: MRI sequencing to include in the dataset
    :param batch_size: batch size of the returned tensors
    :param cache: [True] wether to cache data in main memory
    :param buffer_size: Buffer size used to prefetch data
    :param interleave: If true, subsequent calls to the iterator will generate <interleave> equal batches. Eg. if 3, batch returned will be [A A A B B B ....]
    :param cast_to: [tf.float32] cast image data to this dtype
    :param only_nonempty_labels: [True] If true, filters all the samples that have a completely black (0.0) label map (segmentation)
    :param clip_labels_to: [0.0] If > 0, clips all the segmentation labels to the provided value. eg. providing a 1 yould produce a segmentation with only 0 and 1 values
    :param take_only: [None] If > 0, only returns <take_only> samples from the given dataset before starting a new iteration.
    :return:
    '''
    path = '../datasets/{}.tfrecords'.format(name)
    dataset = tf.data.TFRecordDataset(path, compression_type='GZIP')

    def parse_sample(sample_proto):
        parsed = tf.parse_single_example(sample_proto, feature_description)
        # Decoding image arrays
        shape = [parsed['mri_x_dimension'], parsed['mri_y_dimension'], 1]
        parsed['mri'] = tf.cast(tf.reshape(tf.io.decode_raw(parsed['mri_raw'], tf.uint16), shape=shape), dtype=cast_to)
        parsed['seg'] = tf.cast(tf.reshape(tf.io.decode_raw(parsed['seg_raw'], tf.uint16), shape=shape), dtype=cast_to)
        parsed['seg'] = tf.clip_by_value(parsed['seg'], 0.0, clip_labels_to) if clip_labels_to else parsed['seg']
        return parsed

    parsed_dataset = dataset.map(parse_sample, num_parallel_calls=os.cpu_count())
    parsed_dataset = parsed_dataset.filter(lambda x: tf.reduce_any(tf.greater(x['seg'], 0.0))) if only_nonempty_labels else parsed_dataset
    parsed_dataset = parsed_dataset.filter(lambda x: tf.equal(x['mri_type'], mri_type)) if mri_type is not None else parsed_dataset
    parsed_dataset = parsed_dataset.take(take_only) if take_only is not None else parsed_dataset
    parsed_dataset = parsed_dataset.cache() if cache else parsed_dataset
    parsed_dataset = parsed_dataset.prefetch(buffer_size)
    #parsed_dataset = parsed_dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size))
    parsed_dataset = parsed_dataset.shuffle(buffer_size, reshuffle_each_iteration=True) if shuffle else parsed_dataset
    parsed_dataset = parsed_dataset.batch(batch_size)

    if interleave > 1:
        parsed_dataset = parsed_dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(interleave), cycle_length=os.cpu_count(), block_length=interleave, num_parallel_calls=os.cpu_count())
    return parsed_dataset


if __name__ == '__main__':
    name = 'brats2015-Train-all'
    # name = 'brats2015-Test-all'
    # name = 'BD2Decide-T1T2'
    #pack_dataset(name)
    #load_dataset(name)
    #compute_dataset_statistics(datasets_path_pattern[name], name=name)
    #testshuffle()