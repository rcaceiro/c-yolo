#include "yolo.h"

int main()
{
 void *yolo=NULL;
 image_detection *detection_image=NULL;
 video_detection *detection_video=NULL;
 yolo_init(&yolo, "./darknet", "./cfg/coco.data", "./cfg/yolov3.cfg", "../weights/yolov3.weights");
 yolo_detect_image(yolo, &detection_image, "./darknet/data/kite.jpg", 0.5);
 yolo_detect_video(yolo, &detection_video, "./data/test.mp4", 0.5, 1);
 image_detection_free(&detection_image);
 video_detection_free(&detection_video);
 yolo_free(yolo);
}