import os
from sahi.slicing import slice_coco

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))


# Train!
coco_annotation_file_path = os.path.join(script_dir, "new_coco_dataset/train/instances.json")
image_dir = os.path.join(script_dir, "new_coco_dataset/train")

output_dir = os.path.join(script_dir, "new_128_coco_sliced/train/")

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path=coco_annotation_file_path,
    image_dir=image_dir,
    output_dir=output_dir,
    output_coco_annotation_file_name="instances",
    slice_height=128, # 256,
    slice_width=128, # 256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

print(f"Sliced COCO annotations saved to: {coco_path}")
print(f"Number of images in sliced dataset: {len(coco_dict['images'])}")
print(f"Fist image info: {coco_dict['images'][0]}")


# Val!
coco_annotation_file_path = os.path.join(script_dir, "new_coco_dataset/val/instances.json")
image_dir = os.path.join(script_dir, "new_coco_dataset/val")

output_dir = os.path.join(script_dir, "new_128_coco_sliced/val/")

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path=coco_annotation_file_path,
    image_dir=image_dir,
    output_dir=output_dir,
    output_coco_annotation_file_name="instances",
    slice_height=128, # 256,
    slice_width=128, # 256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

print(f"Sliced COCO annotations saved to: {coco_path}")
print(f"Number of images in sliced dataset: {len(coco_dict['images'])}")
print(f"Fist image info: {coco_dict['images'][0]}")