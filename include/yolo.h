#ifndef YOLO_H
#define YOLO_H

#include <stddef.h>
#include "yolo_error.h"

#define __cplusplus 2018703L
#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
 float x;
 float y;
 float width;
 float height;
}detection_box;

typedef struct
{
 char *class_name;
 float probability;
 detection_box box;
}object_detection;

typedef struct
{
 object_detection *object_detections;
 size_t num_detections;
 float time_spent_for_classification;
}image_detection;

typedef struct
{
 image_detection image_detection;
 double millisecond;
 long frame;
}frame_detection;

typedef struct
{
 frame_detection *frame_detections;
 size_t num_frames;
}video_detection;

yolo_status yolo_init(void **yolo_obj, char *workingDir, char *datacfg, char *cfgfile, char *weightfile);
yolo_status yolo_detect_image(void *yolo, image_detection **detect, char *filename, float thresh);
yolo_status yolo_detect_video(void *yolo, video_detection **detect, char *filename, float thresh, double fraction_frames_to_process);
void image_detection_free(image_detection**p_image_detection);
void video_detection_free(video_detection **p_video_detection);
void yolo_free(void *yolo);

#ifdef __cplusplus
};

class Yolo
{

};
#endif
#endif // YOLO_H