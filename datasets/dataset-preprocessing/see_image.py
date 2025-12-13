import cv2

image = cv2.imread("new_128_train_val_dataset_sliced_balanced_upsampled_bg20/images/train/RQ1_Qubit_W1-251022_Bridges-dark-000047_0_1044_116_1172_244.png")
cv2.imshow('PCB Defect View', image)
cv2.waitKey(0)
cv2.destroyAllWindows()