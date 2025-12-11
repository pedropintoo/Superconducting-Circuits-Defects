import sahi
import glob
import json
import torch
from sahi import AutoDetectionModel
from sahi.predict import predict

from mmdetection.tools.analysis_tools.confusion_matrix import calculate_confusion_matrix
from mmengine import Config
from mmdet.utils import replace_cfg_vals, update_data_root
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS
from mmdetection.tools.analysis_tools.confusion_matrix import plot_confusion_matrix

# Locate the trained, tiled, MMDetection model
# model_name = 'BENTHIC_PATCHES_OFFLINE_250_250_05_OVERLAP_FASTER_RCNN_25_CLASS_SAHI_TEST'
# config_src = './mmdetection/configs/faster_rcnn/' + model_name + '.py'
# checkpoints_src = './work_dirs/' + model_name
# best_checkpoint = glob.glob(checkpoints_src + '/best_coco_bbox_mAP_epoch*.pth')[0]

best_checkpoint = "chip_defect_detection/sliced_640/weights/best.pt"
checkpoints_src = "."
# Locate the non-tiled test images JSON
img_src = './whole_images/'
whole_image_test_dataset_json_path = img_src + '/annotations/dataset_test.json'

# Required options
bbox_conf_threshold = 0.2
patch_size = (256, 256)
overlap = 0.2

# Load the MMDetection model
model = AutoDetectionModel.from_pretrained(model_type = 'ultralytics',
                           model_path = best_checkpoint,
                           # config_path = config_src,
                           confidence_threshold = bbox_conf_threshold,
                           device = 'cuda')

# Run prediction on the non-tiled images, save out in JSON format
output_name = 'sahi_eval_results' # output_dir/output_name

predict(
    model_type = 'ultralytics',
    model_path = best_checkpoint,
    model_device = 'cuda',
    model_confidence_threshold = bbox_conf_threshold,
    source = img_src,
    slice_height = patch_size[0],
    slice_width = patch_size[1],
    overlap_height_ratio = overlap,
    overlap_width_ratio = overlap,
    postprocess_type = 'NMM',
    postprocess_match_metric = 'IOU',
    dataset_json_path = whole_image_test_dataset_json_path,
    visual_bbox_thickness = 2,
    project = checkpoints_src,
    name = output_name,
    novisual = True
)

# Load in the predicted results
results_json = checkpoints_src + '/' + output_name + '/result.json'
results_json = json.load(open(results_json))

# Group the detections by image id
def group_detections_by_image_id(json):
    grouped = {}
    for i, item in enumerate(json):
        image_id = item['image_id']
        if image_id not in grouped:
            grouped[image_id] = []
        grouped[image_id].append(item)        
    return grouped

results_grouped = group_detections_by_image_id(results_json)

### Get into format for MMDetection eval scripts
results_grouped_mmdet = []
for per_img_res in results_grouped:
    per_img_res_mmdet = {}
    per_img_res_mmdet['image_id'] = per_img_res
    per_img_res_mmdet['pred_instances'] = {}
    
    # SAHI outputs results in form [x_min, y_min, w, h] but Ground Truth expects [x_min, y_min, x_max, y_max]
    new_bbox = []
    for item in results_grouped[per_img_res]:
        new_bbox.append([item['bbox'][0], item['bbox'][1], item['bbox'][0] + item['bbox'][2], item['bbox'][1] + item['bbox'][3]])
    per_img_res_mmdet['pred_instances']['bboxes'] = torch.tensor(new_bbox)
    
    per_img_res_mmdet['pred_instances']['scores'] = torch.tensor([item['score'] for item in results_grouped[per_img_res]])
    per_img_res_mmdet['pred_instances']['labels'] = torch.tensor([item['category_id'] for item in results_grouped[per_img_res]])
    
    results_grouped_mmdet.append(per_img_res_mmdet)

### Calculate conf matrix adapting from MMDetection analysis_tools/confusion_matrix.py

def build_dataset(cfg_src):
    cfg = Config.fromfile(cfg_src)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    return dataset


def conf_mat(dataset, results, score_thr = 0, tp_iou_thr = 0.5):
    
    # Get the confusion matrix
    confusion_matrix = calculate_confusion_matrix(
        dataset, results, score_thr= score_thr, tp_iou_thr= tp_iou_thr
    )
    return confusion_matrix        

dataset = build_dataset(config_src)

conf = conf_mat(dataset,
                results_grouped_mmdet,
                score_thr = bbox_conf_threshold,
                tp_iou_thr= 0.5)

labels_dataset = dataset.metainfo['classes'] + ['background']
plot_confusion_matrix(conf, labels_dataset, show = True)