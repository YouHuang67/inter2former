from pathlib import Path
from scipy.io import loadmat

import numpy as np
import mmengine
from mmengine.dist import get_dist_info
from mmseg.registry import DATASETS

from ..test.sbd import SBDDataset


@DATASETS.register_module()
class SBDTrainDataset(SBDDataset):

    default_meta_root = 'data/meta-info/sbd-train.json'

    def load_data_list(self):
        data_root = Path(self.data_root)
        meta_root = Path(self.meta_root)
        if meta_root.is_file():
            data_list = mmengine.load(meta_root)['data_list']
        else:
            data_list = []
            with open(data_root / 'train.txt', 'r') as file:
                val_set = set(line.strip() for line in file.readlines())
            img_files = {p.stem: p for p in
                         data_root.rglob(f'*{self.img_suffix}')
                         if p.stem in val_set}
            ann_files = {p.stem: p for p in
                         data_root.rglob(f'*inst/*{self.ann_suffix}')
                         if p.stem in val_set}
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            for prefix in sorted(list(prefixes)):
                img_file = str(img_files[prefix])
                ann_file = str(ann_files[prefix])
                mask = loadmat(ann_file)['GTinst'][0][0][0]
                for idx in np.unique(mask):
                    if idx == 0:
                        continue
                    data_list.append(
                        dict(img_path=img_file, seg_map_path=ann_file,
                             seg_fields=[], segments_info=[dict(id=idx)],
                             reduce_zero_label=False)
                    )
            if get_dist_info()[0] == 0:
                mmengine.dump(dict(data_list=data_list), meta_root, indent=4)
        return data_list
