import tempfile
from os import path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmdet.datasets import DATASETS

from ..core.bbox import get_box_type
from .pipelines import Compose
from .utils import extract_result_dict


@DATASETS.register_module()
class Custom3DDataset(Dataset):
    CLASSES = ()  # <== 이 줄을 추가하세요
    """Customized 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.

    Args:
        dataset_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(
        self,
        dataset_root,
        ann_file,
        pipeline=None,
        classes=None,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        self.epoch = -1
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self, "pipeline"):
            for transform in self.pipeline.transforms:
                if hasattr(transform, "set_epoch"):
                    transform.set_epoch(epoch)
        
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # return mmcv.load(ann_file)
        data = mmcv.load(ann_file)
        if isinstance(data, dict) and 'infos' in data:
            return data['infos']
        return data

    @staticmethod
    def RT_to_matrix(rotation, translation):
        """Convert rotation (quaternion or matrix) and translation to 4x4 matrix."""
        if isinstance(rotation, list):
            rotation = np.array(rotation)
        if rotation.shape == (4,):  # quaternion
            from pyquaternion import Quaternion
            R = Quaternion(rotation).rotation_matrix
        else:
            R = rotation
        T = np.array(translation).reshape(3, 1)

        RT = np.eye(4)
        RT[:3, :3] = R
        RT[:3, 3:] = T
        return RT

    def get_data_info(self, index):

        info = self.data_infos[index]
        sample_idx = info["token"]

        # Conditionally set lidar_path and lidar2ego based on modality
        if self.modality and not self.modality.get('use_lidar', True):
            lidar_path = None
            lidar2ego_matrix = None # Set to None if not used
        else:
            lidar_path = osp.join(self.dataset_root, info["lidar_path"])
            lidar2ego_matrix = np.eye(4, dtype=np.float32) # Default if used

        input_dict = dict(
            sample_idx=sample_idx,
            lidar_path=lidar_path,
            file_name=lidar_path, # file_name is often lidar_path, so keep it consistent
        )

        # camera 관련 정보 수집
        cams = info['cams']
        images = []
        camera_intrinsics = []
        camera2ego = []
        camera2lidar = []
        lidar2camera = []
        lidar2image = []
        img_aug_matrix = []
        lidar_aug_matrix = []

        for cam_name in sorted(cams.keys()):
            cam = cams[cam_name]

            # 이미지 경로
            images.append(osp.join(self.dataset_root, cam['data_path']))

            # intrinsic matrix
            intrinsic = np.array(cam['cam_intrinsic'])
            camera_intrinsics.append(intrinsic)

            # sensor2ego
            sensor2ego = self.RT_to_matrix(cam['sensor2ego_rotation'], cam['sensor2ego_translation'])
            camera2ego.append(sensor2ego)

            # sensor2lidar
            sensor2lidar = self.RT_to_matrix(cam['sensor2lidar_rotation'], cam['sensor2lidar_translation'])
            camera2lidar.append(sensor2lidar)

            # lidar2camera = inverse(sensor2lidar)
            lidar2cam = np.linalg.inv(sensor2lidar)
            lidar2camera.append(lidar2cam)

            # lidar2image = intrinsic @ lidar2camera[:3, :]
            lidar2img = intrinsic @ lidar2cam[:3, :]
            lidar2image.append(lidar2img)

        input_dict.update({
            "img" : images,
            # "img_filename": images,
            "camera_intrinsics": np.array(camera_intrinsics),
            "camera2ego": np.array(camera2ego),
            "camera2lidar": np.array(camera2lidar),
            "lidar2camera": np.array(lidar2camera),
            "lidar2image": np.array(lidar2image),
            "lidar2ego": lidar2ego_matrix,  # lidar 중심 좌표계
            "img_aug_matrix": np.array(img_aug_matrix),
            "lidar_aug_matrix": np.array(lidar_aug_matrix),
            "metas": info
        })
        print("images:", images)
        print("type(images):", type(images))
        print("len(images):", len(images))
        print("image[0]:", images[0])
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos
            if self.filter_empty_gt and ~(annos["gt_labels_3d"] != -1).any():
                return None
        print("[DEBUG] get_data_info keys:", input_dict.keys())
        return input_dict

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results["img_fields"] = ["img"]
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and (
            example is None or ~(example["gt_labels_3d"]._data != -1).any()
        ):
            return None
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # print("==== input_dict ====")
        # for k, v in input_dict.items():
        #     print(f"{k}: {type(v)} {getattr(v, 'shape', '')}")
        #     print(f"[DEBUG] input_dict keys: {input_dict.keys()}")
        try:
            example = self.pipeline(input_dict)
            return example
        except Exception as e:
            print("=== Pipeline Error ===")
            import traceback
            traceback.print_exc()
            print("Input dict keys:", input_dict.keys())
            raise e 

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        return class_names

    def format_results(self, outputs, pklfile_prefix=None, submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results, \
                tmp_dir is the temporal directory created for saving json \
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, "results")
            out = f"{pklfile_prefix}.pkl"
        mmcv.dump(outputs, out)
        return outputs, tmp_dir

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, "data loading pipeline is not provided"
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
