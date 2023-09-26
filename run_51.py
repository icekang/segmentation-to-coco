import fiftyone as fo

dataset = fo.Dataset.from_dir(
    name=f"small_ds_test",
    overwrite=True, # change here if you already run this cell
    persistent=True, # across notebook restarts
    dataset_type=fo.types.COCODetectionDataset, # <-- all kinds of detection (bbox, instance, segm)
    data_path='dataset/images',  # <-- images
    labels_path='dataset/annotations/solar_panel_segmentation.json', # <-- COCO style json annotations
    label_types=['segmentations'], # change to detections for your use case
    use_polylines=False, # in case of segmentation, this avoids crashing for large images
    include_id=True,
)
# classes = dataset.default_classes
# fo.utils.coco.add_coco_labels(dataset, 'predictions', 'output/small_ds_training/eval_no_threshold/coco_instances_results.json', classes=classes, coco_id_field='coco_id')
session = fo.launch_app(dataset)
session.wait()