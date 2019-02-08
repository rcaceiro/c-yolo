#include "yolo.h"

int main()
{
 yolo_object *yolo=NULL;
 yolo_detection_image *detection_image=NULL;
 yolo_init(&yolo, "./darknet", "./cfg/coco.data", "./cfg/yolov3.cfg", "../weights/yolov3.weights");
 yolo_detect_image(yolo, &detection_image, "./darknet/data/kite.jpg", 0.5);
 yolo_detection_image_free(&detection_image);
 yolo_cleanup(yolo);
}