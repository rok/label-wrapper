import json
import label_wrapper
import os


def test_add_images(tmp_path):
    image_filepath = 'data/example.jpg'
    image_set_filepath = 'data/image_set_1.json'
    out_json_path = os.path.join(tmp_path, "image_set_1.json")
    out_tfrecords_path = os.path.join(tmp_path, 'test.tfrecords')

    image_set = label_wrapper.ImageSet().add_images([image_filepath])
    image_set.save_json(out_json_path)

    with open(out_json_path, 'r') as f:
        out_json = json.load(f)

    with open(image_set_filepath, 'r') as f:
        expected_json = json.load(f)

    assert out_json == expected_json

    image_set.export(out_tfrecords_path)
    assert os.path.exists(out_tfrecords_path)

    image_set_from_tfrecords = label_wrapper.ImageSet().import_labels(out_tfrecords_path)
    assert image_set_from_tfrecords.to_dict == image_set.to_dict

def test_load_json(tmp_path):
    image_set_filepath = 'data/image_set_2.json'
    out_path = os.path.join(tmp_path, "image_set_2.html")
    out_tfrecords_path = os.path.join(tmp_path, 'test.tfrecords')

    image_set = label_wrapper.ImageSet().load_json(image_set_filepath, 'data')

    image_set.create_via(out_path)
    assert os.path.exists(out_path)

    image_set.export(out_tfrecords_path)
    assert os.path.exists(out_tfrecords_path)

    image_set_from_tfrecords = label_wrapper.ImageSet().import_labels(out_tfrecords_path)
    assert image_set_from_tfrecords.to_dict == image_set.to_dict