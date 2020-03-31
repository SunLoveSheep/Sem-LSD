from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.ctdet_line import CTDetLineDataset
from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.coco_hp import COCOHP
from .dataset.semantic_line_kaist import SemanticLineKAIST

dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'coco_hp': COCOHP,
  'semantic_line_kaist': SemanticLineKAIST,
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset,
  'ctdet_line': CTDetLineDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
