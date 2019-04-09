import tensorflow as tf
import argparse

def default_parse(serialized):

    features = \
        {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.float64)
    
    # Get the label associated with the image.
    label = parsed_example['label']

    # The image and label are now correct TensorFlow types.
    return image, label

def get_tfrecord_dataset(input_file, parse_fn=None):
    """read a tfrecord and return a dataset.

    Keyword arguments:
    input_file -- the path to tfrecord file
    parse_fn -- the function to parse tfrecords
    """
    dataset = tf.data.TFRecordDataset(input_file)
    if parse_fn:
        dataset = dataset.map(parse_fn)
    else:
        dataset = dataset.map(default_parse) 
    
    
    return dataset    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='read TFrecord file.')
    parser.add_argument('-i', '--input-file', metavar='', default='dataset.tfr', help='the TFRecord file')

    args = parser.parse_args()
    dataset = get_tfrecord_dataset(args.input_file)
    iter = dataset.make_initializable_iterator()
    data_iter = iter.get_next()

    with tf.Session() as sess:
        sess.run(iter.initializer)
        preview_data = sess.run(data_iter)


    print(f'the batch images shape is :{preview_data[0].shape}')
    print(f'the batch labels shape is :{preview_data[1].shape}')
    


