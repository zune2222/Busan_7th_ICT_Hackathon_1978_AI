import cv2
import numpy as np
import os

if __name__ == "__main__":
    yolo_path = "./yolo/"
else:
    yolo_path = "./human_detection/yolo/"
yolo_weights = yolo_path+"yolov3-spp.weights"
yolo_cfg = yolo_path+"yolov3-spp.cfg"

YOLO_net = cv2.dnn.readNet(yolo_weights,yolo_cfg)

classes = ["person"]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

yolo_img_small = (320,320)
yolo_img_middle = (416,416)
yolo_img_large = (608,608)

def detect(frame):
    height, width, channels = frame.shape

    # YOLO 입력
    blob = cv2.dnn.blobFromImage(image=frame, scalefactor=0.00092, size=yolo_img_large, mean=(0, 0, 0, 0),
                                 swapRB=True, crop=False)
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    centers = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                centers.append((center_x, center_y))
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    results = []
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            if len(classes) > class_ids[i]:
                label = str(classes[class_ids[i]])
                results.append((centers[i], label))
            else:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, (0, 0, 255), 3)

    return (frame,results)

def detectImageAndShow(path, output_path):
    img = cv2.imread(path)

    result_image, _ = detect(img)
    cv2.imshow("result", result_image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

async def detectImage(path):
    img = cv2.imread(path)

    _, results = detect(img)

    return results

if __name__ == "__main__":

    sample_input_path = "./sample/input/"
    sample_output_path = "./sample/output/"

    yolo_path = "./yolo"

    for name in os.listdir(sample_input_path):
        if name == ".gitkeep": continue
        detectImageAndShow(sample_input_path+name,sample_output_path+name)

