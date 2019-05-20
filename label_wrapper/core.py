import base64
from collections import OrderedDict
from itertools import islice
import json
import os
import random
import re

import attr
import jinja2

from .io import TensorflowDatasetExporter, TensorflowDatasetImporter

dir_path = os.path.dirname(os.path.realpath(__file__))

TEMPLATE_FILE = 'via-2.0.6.html'
ANNOTATION_LOADING_TEMPLATE = \
    """
    function _via_load_submodules() {
      _via_load_img();
      _via_define_attributes();
      _via_define_annotations();

      toggle_attributes_editor();
      update_attributes_update_panel();

      annotation_editor_show();
    }

    function _via_load_img() {
      var i, n;
      var file_count = 0;
      n = _via_img.length;
      for ( i = 0; i < n; ++i ) {
        project_file_add_base64( _via_img_filename[i], _via_img[i] );
        file_count += 1;
      }

      _via_show_img(0);
      update_img_fn_list();
    }
    
    function _via_define_attributes() {
      project_import_attributes_from_json('{{ attributes }}');
    }

    var _via_img = [];
    var _via_img_filename = [{{ filenames }}];
    
    function _via_define_annotations() {
      import_annotations_from_json(JSON.stringify({{labels_json}}));
    }
    
    {% for image in images %}
        _via_img.push('data:image/{{ image.format }};base64,{{ image.base64 }}');
    {% endfor %}
    """

SCRIPT_LOADING_AREA_PATTERN = '//<!--AUTO_INSERT_VIA_TEST_JS_HERE-->'

with open(os.path.join(dir_path, TEMPLATE_FILE), 'r') as f:
    via_html = f.read()

template_html = re.sub(SCRIPT_LOADING_AREA_PATTERN, ANNOTATION_LOADING_TEMPLATE, via_html)
template = jinja2.Template(template_html)

@attr.s
class Region(object):
    region_attributes = attr.ib(type=dict)
    shape_attributes = attr.ib(type=dict)

@attr.s(repr=False)
class Image(object):
    filename = attr.ib(type=str)
    size = attr.ib(default='-1', type=int)
    regions = attr.ib(default=[], converter=lambda x: [Region(**y) for y in x], type=dict)
    base64_img_data = attr.ib(default='', type=str)
    file_attributes = attr.ib(default={}, type=dict)

    def __repr__(self):
        return json.dumps(attr.asdict(self), indent=2)

    @property
    def to_dict(self):
        return attr.asdict(self)

@attr.s(repr=False)
class ImageSet(object):
    _image_dict = attr.ib(default={}, converter=lambda x: {k: Image(**v) for k, v in x.items()}, type=dict)
    exporters = [TensorflowDatasetExporter]
    importers = [TensorflowDatasetImporter]

    def export(self, dataset_path):
        return self.exporters[0].export(self, dataset_path)

    def import_labels(self, tfrecords_path):
        return self.importers[0].import_labels(self, tfrecords_path)

    def load_json(self, input_file_path, input_folder_path, overwrite=False):
        with open(input_file_path, 'r') as f:
            labels = json.load(f)

        filedicts = {k: Image(**v) for k, v in labels.items()}

        for x in filedicts.values():
            path = os.path.join(input_folder_path, x.filename)
            if os.path.exists(path):
                # x.fileref = path
                x.file_attributes = {"image": path}
            else:
                raise FileNotFoundError("{path} not found.".format(path=path))

        if overwrite:
            self._image_dict = {**self._image_dict, **filedicts}
        else:
            self._image_dict = {**filedicts, **self._image_dict}
        return self

    def add_images(self, image_filepaths, raw_filepaths=None, overwrite=False):

        filedicts = {}

        if raw_filepaths:
            assert len(image_filepaths) == len(raw_filepaths)
        else:
            raw_filepaths = image_filepaths

        for image_filepath, raw_filepath in zip(image_filepaths, raw_filepaths):
            filename = os.path.basename(image_filepath)
            filedicts[filename + '-1'] = \
                Image(filename=filename, file_attributes={'image': image_filepath, 'raw': raw_filepath})

        if overwrite:
            self._image_dict = {**self._image_dict, **filedicts}
        else:
            self._image_dict = {**filedicts, **self._image_dict}
        return self

    @property
    def to_dict(self):
        return {k: v.to_dict for k, v in self._image_dict.items()}

    @property
    def as_json(self):
        return json.dumps(attr.asdict(self)['_image_dict'], indent=2)

    def save_json(self, output_file_path):
        with open(output_file_path, 'w') as f:
            f.write(self.as_json)
        return output_file_path

    def split(self, how, chunk_size=200):
        d = self.order_images(how=how)._image_dict
        image_sets = []

        for i in range(0, len(d), chunk_size):
            image_set = ImageSet()
            image_set._image_dict = OrderedDict(islice(d.items(), i, i + chunk_size))
            image_sets.append(image_set)

        return image_sets

    def order_images(self, how="regions"):
        if how == "regions":
            d = sorted(self._image_dict.items(), key=lambda x: len(self._image_dict[x[0]].regions), reverse=True)

        elif how == "name":
            d = sorted(self._image_dict.items(), key=lambda x: x[0])

        elif how == "random":
            keys = list(self._image_dict.keys())
            random.shuffle(keys)
            d = {k: self._image_dict[k] for k in keys}

        self._image_dict = OrderedDict(d)

        return self

    def create_via(self, output_path):

        dataset_dict = self.to_dict

        filepaths = [image['file_attributes']['image'] for image in dataset_dict.values()]
        filenames = ",".join(["'{}'".format(os.path.basename(filepath)) for filepath in filepaths])

        images = [{
            'base64': base64.b64encode(open(filepath, 'rb').read()).decode(),
            'format': filepath.split('.').pop()
        } for filepath in filepaths]

        labels_json = json.dumps(dataset_dict, indent=2)

        all_attributes = [attribute for image in dataset_dict.values() for attribute in image['file_attributes'].keys()]
        attributes = json.dumps({"file": {x: {"type": "text", "default_value": ""} for x in set(all_attributes)}})

        output = template.render(labels_json=labels_json, attributes=attributes, images=images, filenames=filenames)

        with open(output_path, 'w') as f:
            f.write(output)

        return output_path

    def __repr__(self):
        sample_items = list(self._image_dict.keys())[:5]
        sample = json.dumps([{x: self._image_dict[x].to_dict} for x in sample_items])

        return "ImageSet containing {size} images.\n{sample} ...".format(
            size=len(self._image_dict.keys()),
            sample=sample,
        )

    def __add__(self, other):
        # TODO: Check if raw data sources match
        image_set = ImageSet()
        image_set._image_dict = {**self._image_dict, **other._image_dict}
        return image_set

    def __sub__(self, other):
        image_set = ImageSet()
        image_set._image_dict = {k: v for k, v in self._image_dict.items() if k not in other._image_dict.keys()}
        return image_set
