import os

import numpy as np
import skimage
import tensorflow as tf

from . import core

tf.enable_eager_execution()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_example(image):
    image_filepath = image.file_attributes['image']
    # raw_filepath = image.file_attributes['raw']
    image_string = tf.io.read_file(image_filepath)
    image_shape = tf.image.decode_image(image_string).shape
    mask = _generate_mask(regions=image.regions, height=image_shape[0], width=image_shape[1])
    mask_string = tf.io.serialize_tensor(mask)

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'file_attributes.image': _bytes_feature(image.file_attributes.get('image', '').encode()),
        'file_attributes.raw': _bytes_feature(image.file_attributes.get('raw', '').encode()),
        'image': _bytes_feature(image_string),
        'mask_raw': _bytes_feature(mask_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def _tfrecord_decoder(filename):
    raw_image_dataset = tf.data.TFRecordDataset(filename)

    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'file_attributes.image': tf.io.FixedLenFeature([], tf.string),
        'file_attributes.raw': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    for image_features in parsed_image_dataset:
        yield \
            {'image': image_features['file_attributes.image'].numpy().decode(), 'raw': image_features['file_attributes.raw'].numpy().decode()}, \
            tf.image.decode_image(image_features['image']).numpy(), \
            tf.io.parse_tensor(image_features['mask_raw'], tf.bool).numpy()


def _generate_mask(regions, height, width):
    masks = [_create_mask(height, width, region.shape_attributes) for region in regions]

    if len(masks) > 0:
        masks = np.stack(masks, axis=2)
        return np.any(masks, axis=2)

    else:
        return np.zeros((height, width), dtype='bool')


def _create_mask(height, width, shape_attributes):
    mask = np.zeros((height, width), dtype='bool')

    if shape_attributes['name'] == 'circle':
        rr, cc = skimage.draw.circle(shape_attributes['cy'], shape_attributes['cx'], shape_attributes['r'])

    elif shape_attributes['name'] == 'ellipse':
        rr, cc = skimage.draw.ellipse(shape_attributes['cy'], shape_attributes['cx'], shape_attributes['ry'], shape_attributes['rx'])

    elif shape_attributes['name'] == 'polygon':
        rr, cc = skimage.draw.polygon(shape_attributes['all_points_y'], shape_attributes['all_points_x'])

    elif shape_attributes['name'] == 'rect':
        rr, cc = skimage.draw.rectangle(start=(shape_attributes['y'], shape_attributes['x']),
                           extent=(shape_attributes['height'], shape_attributes['width']),
                           shape=mask.shape)

    else:
        raise NotImplementedError("Region type '{region_type}' not implemented.".format(region_type=shape_attributes['name']))

    mask[rr, cc] = True

    return mask


def _extract_polygons(mask, cutoff=100):
    contours = skimage.measure.find_contours(mask, .5)
    regions = []

    for contour in contours:
        # Extract polygons
        contour = skimage.measure.subdivide_polygon(contour)
        coordinates = skimage.measure.approximate_polygon(contour, tolerance=1)

        # Check polygon area
        relative_coordinates = coordinates - coordinates.min(axis=0)
        maximum_relative_coordinates = relative_coordinates.max(axis=0).astype(int) + 1
        region_mask = np.zeros(maximum_relative_coordinates, dtype='bool')
        rr, cc = skimage.draw.polygon(relative_coordinates[:, 0], relative_coordinates[:, 1])
        region_mask[rr, cc] = True

        # Append big regions
        if region_mask.sum() >= cutoff:
            region = {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": coordinates[:, 1].tolist(),
                    "all_points_y": coordinates[:, 0].tolist()},
                "region_attributes": {
                    "generated": "true"
                }
            }
            regions.append(region)

    return regions


class TensorflowDatasetExporter():
    """
    """

    def export(self, dataset_path, output_shape=None):
        os.makedirs(os.path.dirname(dataset_path), 0o755, exist_ok=True)

        with tf.io.TFRecordWriter(dataset_path) as writer:
            for _, image in self._image_dict.items():
                if output_shape:
                    image = skimage.transform.resize(image, output_shape=output_shape)
                tf_example = _image_example(image)
                writer.write(tf_example.SerializeToString())


class TensorflowDatasetImporter():
    """
    """

    def import_labels(self, tfrecords_path):
        image_set = core.ImageSet()

        for file_attributes, image, mask in _tfrecord_decoder(tfrecords_path):
            filename = os.path.basename(file_attributes['image'])
            regions = _extract_polygons(mask, 100)

            image = core.Image(filename=filename, size='-1', regions=regions, file_attributes=file_attributes)

            image_set._image_dict[filename + '-1'] = image

        return image_set
