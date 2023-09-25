# Segmentation-to-COCO
converting segmentation mask to COCO dataset format

# Expected directory structure for segmentation dataset
```
dataset
|
|---images
|   |---img1.jpg
|   |---img2.jpg
|   |---img3.jpg
|   ...
|
└---masks
    |---img1.png
    |---img2.png
    |---img3.png
    ...
```

# Final directory structure for COCO dataset
```
dataset
|
|---images
|   |---img1.jpg
|   |---img2.jpg
|   |---img3.jpg
|   ...
|
└---annotations
    |---segmentation_train.json
    |---segmentation_val.json
    |---segmentation_test.json
```