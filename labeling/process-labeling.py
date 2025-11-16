import os
import cv2
import random

RELATIVE_PATH = "../"

class YoloFormat:
    def __init__(self, yolo_path: str, images_path: str) -> None:
        self.yolo_path = yolo_path
        self.images_path = images_path  
        self.labels_path = os.path.join(RELATIVE_PATH, yolo_path, "labels")
        self.images_path_yolo = os.path.join(RELATIVE_PATH, yolo_path, "images")
        
    def fill_images(self, already_splitted: bool = False):
        if not already_splitted:
            if os.listdir(self.images_path_yolo):
                print("Yolo images folder is not empty, aborting...")
                return
            
            for label_file in os.listdir(self.labels_path):
                image_file = os.path.join(RELATIVE_PATH, self.images_path, label_file.replace(".txt", ".jpg"))
                img = cv2.imread(image_file)
                output_path = os.path.join(self.images_path_yolo, label_file.replace(".txt", ".jpg"))
                cv2.imwrite(output_path, img)
        # Just need to create images for /train and /val folders
        else:
            if not os.path.exists(self.images_path_yolo):
                os.makedirs(self.images_path_yolo, exist_ok=True)
            
            for split in ["train", "val"]:
                split_images_path = os.path.join(self.images_path_yolo, split)
                if not os.path.exists(split_images_path):
                    os.makedirs(split_images_path, exist_ok=True)
                
                split_labels_path = os.path.join(self.labels_path, split)
                for label_file in os.listdir(split_labels_path):
                    image_file = os.path.join(RELATIVE_PATH, self.images_path, label_file.replace(".txt", ".jpg"))
                    img = cv2.imread(image_file)
                    output_path = os.path.join(split_images_path, label_file.replace(".txt", ".jpg"))
                    cv2.imwrite(output_path, img)
    
    def split_train_val_test(self, validation_rate: float, seed: int):
        if os.path.exists(os.path.join(self.images_path_yolo, "train")):
            print("Train/Val folders already exist, aborting...")
            return
        
        image_files = [f for f in os.listdir(self.images_path_yolo) if not os.path.isdir(os.path.join(self.images_path_yolo, f))]
        num_samples = len(image_files)
        num_val = int(num_samples * validation_rate)
        num_train = num_samples - num_val
        
        print(f"Total: {num_samples}, Train: {num_train}, Validation: {num_val}")
        
        random.seed(seed)
        random.shuffle(image_files)
        val_files = image_files[:num_val]
        train_files = image_files[num_val:]
        
        self._create_split_folders(train_files, val_files)
    
    def _create_split_folders(self, train_files, val_files):
        train_images_path = os.path.join(self.images_path_yolo, "train")
        val_images_path = os.path.join(self.images_path_yolo, "val")
        train_labels_path = os.path.join(self.labels_path, "train")
        val_labels_path = os.path.join(self.labels_path, "val")
        
        os.makedirs(train_images_path, exist_ok=True)
        os.makedirs(val_images_path, exist_ok=True)
        os.makedirs(train_labels_path, exist_ok=True)
        os.makedirs(val_labels_path, exist_ok=True)
        
        for file_name in train_files:
            self._move_file(file_name, train_images_path, train_labels_path)
        for file_name in val_files:
            self._move_file(file_name, val_images_path, val_labels_path)
    
    def _move_file(self, file_name, images_dest, labels_dest):
        src_image = os.path.join(self.images_path_yolo, file_name)
        src_label = os.path.join(self.labels_path, file_name.replace(".jpg", ".txt"))
        dest_image = os.path.join(images_dest, file_name)
        dest_label = os.path.join(labels_dest, file_name.replace(".jpg", ".txt"))
        
        os.rename(src_image, dest_image)
        os.rename(src_label, dest_label)


if __name__ == "__main__":
    # To download the images from a already splitted YOLO format:
    # e.g: After clonning the repository you need to run this!
    yolo_format = YoloFormat("datasets/chip_defects", "datasets/RQ3_TWPA_V2_W2/251023_Junctions/dark")
    yolo_format.fill_images(already_splitted=True)

    # To generate from a simple YOLO format to a simpled one: (e.g: After downloading from Label Studio in YOLO format)
    # yolo_format.fill_images(already_splitted=False)
    # yolo_format.split_train_val_test(validation_rate=0.2, seed=4)
