import json
import os.path
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from ..filesystem import FileSystem
from ..transforms import ImageTransforms, MinPointAndSizeBBox


class SkyFusionDataset(Dataset):
    directories_mapping = {"train": "train", "val": "valid", "test": "test"}
    annotation_json = "_annotations.coco.json"
    default_dataset_path = FileSystem.DATASETS_DIR / "SkyFusion"
    default_preprocessed_dir = "preprocessed"
    data_yaml_file = "data.yaml"
    categories_mapping = {
        1: {"internal_id": 0, "name": "aircraft"},
        2: {"internal_id": 1, "name": "ship"},
        3: {"internal_id": 2, "name": "vehicle"},
    }
    category_names = {
        category["internal_id"]: category["name"]
        for category in categories_mapping.values()
    }

    def __init__(
        self,
        dataset_path: Path | None = None,
        split: str | None = "train",
    ):
        if dataset_path is None:
            dataset_path = self.default_dataset_path

        if not os.path.exists(
            processed_dataset_path := dataset_path / self.default_preprocessed_dir
        ):
            data_yaml_path = self.export_to_yolo(dataset_path)
        else:
            data_yaml_path = processed_dataset_path / self.data_yaml_file

        with open(data_yaml_path, "r") as f:
            data_yaml = yaml.safe_load(f)

        if split not in self.directories_mapping.keys():
            raise ValueError(f"Invalid split: {split}")

        images_path = processed_dataset_path / data_yaml[split]
        labels_path = images_path.parent / "labels"

        self.images_labels_paths = [
            (
                images_path / image_filename,
                labels_path / self._label_filename(image_filename),
            )
            for image_filename in os.listdir(images_path)
        ]

    def __len__(self):
        return len(self.images_labels_paths)

    def __getitem__(self, idx: int):
        image_path, label_path = self.images_labels_paths[idx]
        image = Image.open(image_path)
        image = ImageTransforms.preprocess(image)

        image_annotations = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                annotation = list(map(float, line.strip().split(" ")))
                image_annotations.append(torch.asarray(annotation, dtype=torch.float32))
        image_annotations = torch.stack(image_annotations, dim=0)

        return image, image_annotations

    @classmethod
    def export_to_yolo(
        cls, dataset_path: Path | None = None, force: bool = False
    ) -> Path:
        if dataset_path is None:
            dataset_path = cls.default_dataset_path
        destination_dir = dataset_path / cls.default_preprocessed_dir

        if (
            os.path.exists(data_path := destination_dir / cls.data_yaml_file)
            and not force
        ):
            return data_path

        data_yaml = {"path": os.path.relpath(destination_dir, FileSystem.DATASETS_DIR)}

        for key, directory in cls.directories_mapping.items():
            images_dir = cls._convert_coco_to_yolo(
                dataset_path / directory, destination_dir / directory, force=force
            )

            data_yaml[key] = os.path.relpath(images_dir, destination_dir)

        data_yaml["nc"] = 3
        data_yaml["names"] = cls.category_names
        with open(data_path, "w") as f:
            yaml.dump(data_yaml, f)

        return data_path

    @classmethod
    def _load_annotations_json(cls, json_path: Path):
        with open(json_path) as f:
            raw_json = json.load(f)

        categories = pd.DataFrame(raw_json["categories"])[["id", "name"]]
        images = pd.DataFrame(raw_json["images"])[
            ["id", "file_name", "height", "width"]
        ]
        annotations = pd.DataFrame(raw_json["annotations"])[
            ["id", "image_id", "category_id", "bbox", "area"]
        ]

        return categories, images, annotations

    @classmethod
    def _label_filename(cls, image_filename: str):
        return os.path.splitext(image_filename)[0] + ".txt"

    @classmethod
    def _convert_coco_to_yolo(
        cls, source_dir: Path, destination_dir: Path, force: bool = False
    ):
        coco_json_path = source_dir / cls.annotation_json
        categories, images, annotations = cls._load_annotations_json(coco_json_path)

        images_dir = destination_dir / "images"
        labels_dir = destination_dir / "labels"

        if os.path.exists(images_dir) and os.path.exists(labels_dir) and not force:
            return images_dir

        for directory in [images_dir, labels_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

        for idx, image in images.iterrows():
            image_id = image["id"]

            image_annotations = [
                (
                    cls.categories_mapping[annotation["category_id"]]["internal_id"],
                    *MinPointAndSizeBBox.preprocess(
                        annotation["bbox"], image["width"], image["height"]
                    ),
                )
                for _, annotation in annotations[
                    annotations["image_id"] == image_id
                ].iterrows()
            ]

            image_old_filename = image["file_name"]
            image_new_filename = (
                f"img_{idx:05d}{os.path.splitext(image_old_filename)[1]}"
            )
            shutil.copy(
                source_dir / image_old_filename, images_dir / image_new_filename
            )

            label_filename = cls._label_filename(image_new_filename)
            with open(labels_dir / label_filename, "w") as label_file:
                label_file.write(
                    "\n".join(
                        [
                            " ".join(map(str, image_annotation))
                            for image_annotation in image_annotations
                        ]
                    )
                )

        return images_dir
