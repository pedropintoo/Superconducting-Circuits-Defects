import cv2

IMAGE_PATH = "../train_val_dataset/images/train/Second_Batch-PM251015p1-251022_post_LO_mark-dark-000116.jpg"
window_name = "PCB Defect View"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise SystemExit(f"Could not read image at {IMAGE_PATH}")

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, image)

# Keep processing GUI events until ESC is pressed or the window is closed.
while True:
    key = cv2.waitKey(50) & 0xFF
    if key == 27:
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()