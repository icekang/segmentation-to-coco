from pathlib import Path
from typing import List, Tuple, TypedDict
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image
import json


class ImageInfo(TypedDict):
    id: int
    file_name: str
    height: int
    width: int


class AnnotationInfo(TypedDict):
    segmentation: List[List[float]]
    area: float
    iscrowd: int
    image_id: int
    bbox: List[float]
    category_id: int


class CategoryInfo(TypedDict):
    id: int
    name: str
    supercategory: str


class DataLoader():
    def __init__(self, image_dir=Path('./dataset/images'), mask_dir=Path('./dataset/masks'), image_types=set('.jpg'), mask_types=set('.jpg')) -> None:
        """Set image directory and image types

        Args:
            image_dir (pathlib.Path, optional): path to the directory of the images. Defaults to Path('./dataset/images').
            mask_dir (pathlib.Path, optional): path to the directory of the masks. Defaults to Path('./dataset/masks').
            image_types (set, optional): image extentions in the images/. Defaults to ('.jpg').
            mask_types (set, optional): image extentions in the mask/. Defaults to ('.jpg').
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_types = image_types
        self.mask_types = mask_types
        self.image_name_to_id = {}


    def make_coco_format(self, image_mask_pairs: List[Tuple[Path, Path]], output_path: Path) -> None:
        """Make coco format json file

        Args:
            image_mask_pairs (List[Tuple[Path, Path]]): list of image and mask pairs
            output_path (Path): path to the output json file
        """

        info = {
            'description': 'Solar Panel Segmentation Dataset', 
            'url': 'home page of the project or data set', 
            'version': 'dataset version, eg. 1.0', 
            'year': 2020, 
            'contributor': 'your name or organization', 
            'date_created': 'date in 2020/04/19 format'
        }
        
        image_mask_pairs = self.make_image_mask_pairs()
        images_list, annotations_list = self.make_coco_format_images_and_masks(image_mask_pairs)

        categories_list = [
            {
                "id": 0,
                "name": "background",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "solar panel",
                "supercategory": "none"
            }
        ]
        

        dataset = {
            'type': 'segmentation',
            'info': info,
            'categories': categories_list,
            'images': images_list,
            'annotations': annotations_list
        }

        with open(output_path, 'w') as f:
            json.dump(dataset, f)


    def make_coco_format_images_and_masks(self, image_mask_pairs: List[Tuple[Path, Path]]) -> Tuple[List[ImageInfo], List[AnnotationInfo]]:
        images_list = List[ImageInfo]()
        annotations_list = List[AnnotationInfo]()
        for image_path, mask_path in image_mask_pairs:
            image_info = self.process_image(image_path)
            annotation_infos = self.process_mask(mask_path)

            images_list.append(image_info)
            annotations_list.extend(annotation_infos)

        return images_list, annotations_list


    def process_image(self, image_path: Path) -> ImageInfo:
        image_info = ImageInfo()

        # Check if the image file name is already in the dictionary
        if image_path.stem not in self.image_name_to_id:
            self.image_name_to_id[image_path.stem] = len(self.image_name_to_id)
        image_id = self.image_name_to_id[image_path.stem]
        image_info['id'] = image_id

        # Image file name
        image_info['file_name'] = image_path.name

        # Image shape information
        image = Image.open(image_path)
        image_info['height'] = image.height
        image_info['width'] = image.width

        return image_info
    

    def process_mask(self, mask_path: Path) -> List[AnnotationInfo]:
        annotation_infos = List[AnnotationInfo]()

        mask = self.get_segmentation_mask(mask_path)
        categories = np.unique(mask)
        for category_id in categories:
            annotation_info = AnnotationInfo()
            if category_id == 0:
                continue
            annotation_info['category_id'] = category_id
            encoded_mask = mask_utils.encode(np.asfortranarray((mask == category_id).astype(np.uint8)))
            annotation_info['segmentation'] = list(encoded_mask['counts'])
            annotation_info['area'] = mask_utils.area(encoded_mask)
            annotation_info['bbox'] = mask_utils.toBbox(encoded_mask).tolist()
            annotation_info['iscrowd'] = 0
            annotation_infos.append(annotation_info)

        return annotation_infos


    def get_segmentation_mask(self, mask_path: Path) -> np.ndarray:
        """Get segmentation mask from mask file

        Args:
            mask_path (Path): path to the mask file

        Returns:
            np.ndarray: 2D numpy array of the segmentation mask with value 0 for background and >0 for objects
        """

        # Check if the mask file is a numpy array
        if mask_path.suffix == '.npy':
            segmentation_mask = np.load(mask_path)
        else:
            # If the mask file is not a numpy array, load it as an image
            # and convert it to a numpy array
            segmentation_mask = np.array(Image.open(mask_path).convert('L'))
        
        return segmentation_mask


    def make_image_mask_pairs(self) -> List[Tuple[Path, Path]]:
        """Make a list of image and mask pairs
        
        This method creates pairs of image and mask file paths by matching
        image files with corresponding mask files based on their names.
        
        Returns:
            List[Tuple[Path, Path]]: A list of image and mask pairs, where each
            pair is represented as a tuple of two `Path` objects.
        """

        # Create a dictionary to store the mapping of mask file names (without extension)
        # to their corresponding mask file paths
        name_to_mask_path = {}

        # Iterate through mask files in the mask directory
        for mask_path in self.mask_dir.iterdir():
            # Check if the file's extension (suffix) is in the list of valid mask types
            if mask_path.suffix in self.mask_types:
                # Add an entry to the dictionary where the key is the file name without extension
                # and the value is the full path to the mask file
                name_to_mask_path[mask_path.stem] = mask_path

        # Create a list to store image-mask pairs
        image_mask_pairs = []

        # Iterate through image files in the image directory
        for image_path in self.image_dir.iterdir():
            # Check if the file's extension (suffix) is in the list of valid image types
            if image_path.suffix in self.image_types:
                # Check if there is a corresponding mask for the current image
                if image_path.stem not in name_to_mask_path:
                    # If no mask is found, print a message and continue to the next image
                    print(f'No mask for {image_path.stem}')
                    continue

                # If a corresponding mask is found, get the path to the mask
                mask_path = name_to_mask_path[image_path.stem]

                # Append the image-mask pair as a tuple to the list
                image_mask_pairs.append((image_path, mask_path))

        # Return the list of image-mask pairs
        return image_mask_pairs


if __name__ == '__main__':
    print('Run as main')