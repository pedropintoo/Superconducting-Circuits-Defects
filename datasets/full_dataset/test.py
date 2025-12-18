import cv2

image = cv2.imread("RQ1_Qubit_W1-251022_Bridges-dark-000047.jpg")
cv2.imshow('PCB Defect View', image)
cv2.waitKey(0)
cv2.destroyAllWindows()