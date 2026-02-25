import os

from .base import BaseFrameDataset


class ScanNet(BaseFrameDataset):

    @property
    def _cache_path(self):
        return self.data_root / 'scannet.pkl'

    def _load_scenes(self):
        scenes = []
        for scene_id in os.listdir(self.data_root):
            if not scene_id.startswith('scene'):
                continue
            image_files = os.listdir(self.data_root / scene_id / 'color')
            image_files.sort()
            scene_info = {
                'scene_id': scene_id,
                'image_files': image_files,
                'num_images': len(image_files)
            }
            scenes.append(scene_info)
        return scenes

    def _get_image_path(self, scene_info, image_file):
        return self.data_root / scene_info['scene_id'] / 'color' / image_file
