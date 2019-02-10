#include "yolo.h"
#include "private_structs.h"
#include "darknet.h"

#include <pthread.h>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>

void fill_object_detection(yolo_object *yolo, detection *network_detection, int network_detection_index, object_detection *yolo_detect)
{
 size_t strlength=strlen(yolo->names[network_detection_index]);
 yolo->names[network_detection_index][strlength]='\0';
 yolo_detect->class_name=(char *)calloc(strlength+1, sizeof(char));
 strcpy(yolo_detect->class_name, yolo->names[network_detection_index]);
 yolo_detect->probability=network_detection->prob[network_detection_index]*100;
 yolo_detect->box.x=network_detection->bbox.x-(network_detection->bbox.w/2);
 yolo_detect->box.y=network_detection->bbox.y-(network_detection->bbox.y/2);
 yolo_detect->box.width=network_detection->bbox.w;
 yolo_detect->box.height=network_detection->bbox.h;

 if(yolo_detect->box.x<0)
 {
  yolo_detect->box.x=0;
 }
 if(yolo_detect->box.y<0)
 {
  yolo_detect->box.y=0;
 }
}

yolo_status fill_image_detection(yolo_object *yolo, detection *dets, image_detection *yolo_detect, float time_spent_for_classification, int nboxes, float thresh)
{
 yolo_detect->time_spent_for_classification=time_spent_for_classification;
 yolo_detect->num_detections=0;
 int class_index;
 detection *det;

 for(int i=0; i<nboxes; ++i)
 {
  class_index=-1;
  det=nullptr;
  for(int j=0; j<dets[i].classes; ++j)
  {
   if(dets[i].prob[j]>=thresh)
   {
    if(det == nullptr || (dets[i].prob[j]>det->prob[class_index]))
    {
     class_index=j;
     det=dets+i;
    }
   }
  }
  if(class_index>-1 && det != nullptr)
  {
   void *temp_pointer=realloc(yolo_detect->object_detections, sizeof(object_detection)*(yolo_detect->num_detections+1));
   if(temp_pointer == nullptr)
   {
    return yolo_cannot_realloc_detect;
   }
   yolo_detect->object_detections=(object_detection *)temp_pointer;
   fill_object_detection(yolo, det, class_index, yolo_detect->object_detections+yolo_detect->num_detections);
   yolo_detect->num_detections++;
  }
 }

 return yolo_ok;
}

yolo_status parse_detections_image(yolo_object *yolo, detection *dets, image_detection **yolo_detect, float time_spent_for_classification, int nboxes, float thresh)
{
 if((*yolo_detect) == nullptr)
 {
  (*yolo_detect)=(image_detection *)calloc(1, sizeof(image_detection));
  if((*yolo_detect) == nullptr)
  {
   return yolo_cannot_alloc_yolo_detection;
  }
 }
 return fill_image_detection(yolo, dets, (*yolo_detect), time_spent_for_classification, nboxes, thresh);
}

yolo_status parse_detections_video(yolo_object *yolo, detection *dets, video_detection **yolo_detect, float time_spent_for_classification, long frame_id, double millisecond, int nboxes, float thresh)
{
 if((*yolo_detect) == nullptr)
 {
  (*yolo_detect)=(video_detection *)calloc(1, sizeof(video_detection *));
  if((*yolo_detect) == nullptr)
  {
   return yolo_cannot_alloc_yolo_detection;
  }
 }
 video_detection *video_detection=*yolo_detect;
 auto *temp=(frame_detection *)realloc(video_detection->frame_detections, sizeof(frame_detection)*(video_detection->num_frames+1));
 if(temp == nullptr)
 {
  return yolo_cannot_alloc_yolo_detection;
 }
 memset(temp+video_detection->num_frames, 0, sizeof(frame_detection));
 video_detection->frame_detections=temp;
 video_detection->frame_detections[video_detection->num_frames].frame=frame_id;
 video_detection->frame_detections[video_detection->num_frames].millisecond=millisecond;
 yolo_status yolo_stats=fill_image_detection(yolo, dets, &video_detection->frame_detections[video_detection->num_frames].image_detection, time_spent_for_classification, nboxes, thresh);
 ++video_detection->num_frames;
 return yolo_stats;
}

image libyolo_ipl_to_image(IplImage *src)
{
 int h=src->height;
 int w=src->width;
 int c=src->nChannels;
 image im=make_image(w, h, c);
 auto *data=(unsigned char *)src->imageData;
 int step=src->widthStep;
 int i, j, k;

 for(i=0; i<h; ++i)
 {
  for(k=0; k<c; ++k)
  {
   for(j=0; j<w; ++j)
   {
    im.data[k*w*h+i*w+j]=data[i*step+j*c+k]/255.;
   }
  }
 }
 return im;
}

image libyolo_mat_to_image(cv::Mat &m)
{
 IplImage ipl=m;
 image im=libyolo_ipl_to_image(&ipl);
 rgbgr_image(im);
 return im;
}

unsigned long long unixTimeMillis()
{
 struct timeval tv{};

 gettimeofday(&tv, nullptr);

 return (unsigned long long)(tv.tv_sec)*1000+(unsigned long long)(tv.tv_usec)/1000;
}

void *thread_capture(void *data)
{
 if(data == nullptr)
 {
  return nullptr;
 }
 auto *thread_data=(thread_get_frame_t *)data;
 cv::Mat mat;
 bool skip;
 queue_image_t queue_image;
 while(true)
 {
  skip=false;

  if(pthread_mutex_lock(&thread_data->mutex))
  {
   continue;
  }

  if(!thread_data->video->isOpened())
  {
   thread_data->image_queue->common->end=true;
   pthread_mutex_unlock(&thread_data->mutex);
   break;
  }

  if(!thread_data->video->grab())
  {
   thread_data->image_queue->common->end=true;
   pthread_mutex_unlock(&thread_data->mutex);
   break;
  }

  if(!thread_data->number_frames_to_drop)
  {
   if(!thread_data->video->retrieve(mat))
   {
    thread_data->image_queue->common->end=true;
    pthread_mutex_unlock(&thread_data->mutex);
    break;
   }
   thread_data->number_frames_to_drop=thread_data->number_frames_to_process_simultaneously;
   queue_image.milisecond=thread_data->video->get(CV_CAP_PROP_POS_MSEC);
   queue_image.frame_number=(long)thread_data->video->get(CV_CAP_PROP_POS_FRAMES);
  }
  else
  {
   skip=true;
   --thread_data->number_frames_to_drop;
  }
  pthread_mutex_unlock(&thread_data->mutex);

  if(skip)
  {
   continue;
  }

  if(mat.empty())
  {
   thread_data->image_queue->common->end=true;
   break;
  }
  image yolo_image=libyolo_mat_to_image(mat);
  mat.release();

  queue_image.frame=yolo_image;

  sem_wait(thread_data->image_queue->empty);

  if(pthread_mutex_lock(&thread_data->image_queue->mutex))
  {
   sem_post(thread_data->image_queue->empty);
   continue;
  }
  thread_data->image_queue->queue.push_back(queue_image);
  pthread_mutex_unlock(&thread_data->image_queue->mutex);
  sem_post(thread_data->image_queue->full);
 }
 return nullptr;
}

void *thread_detect(void *data)
{
 if(data == nullptr)
 {
  return nullptr;
 }
 auto *th_data=(thread_processing_image_t *)data;
 while(true)
 {
  if(sem_trywait(th_data->image_queue->full))
  {
   if(th_data->image_queue->common->end)
   {
    break;
   }
   continue;
  }

  queue_image_t queue_image;
  bool im_got_sucessfull;
  if(pthread_mutex_lock(&th_data->image_queue->mutex))
  {
   continue;
  }
  im_got_sucessfull=!th_data->image_queue->queue.empty();
  if(im_got_sucessfull)
  {
   queue_image=th_data->image_queue->queue.front();
   th_data->image_queue->queue.pop_front();
  }
  pthread_mutex_unlock(&th_data->image_queue->mutex);
  sem_post(th_data->image_queue->empty);

  if(!im_got_sucessfull)
  {
   continue;
  }

  layer l=th_data->yolo->net->layers[th_data->yolo->net->n-1];
  unsigned long long time;
  float nms=0.45;

  image sized=resize_image(queue_image.frame, th_data->yolo->net->w, th_data->yolo->net->h);
  float *X=sized.data;
  time=unixTimeMillis();
  network_predict(th_data->yolo->net, X);

  int nboxes=0;
  detection *dets=get_network_boxes(th_data->yolo->net, queue_image.frame.w, queue_image.frame.h, th_data->thresh, 0, nullptr, 0, &nboxes);
  if(nms>0)
  {
   do_nms_sort(dets, l.side*l.side*l.n, l.classes, nms);
  }

  parse_detections_video(th_data->yolo, dets, th_data->yolo_detect, (unixTimeMillis()-time), queue_image.frame_number, queue_image.milisecond, nboxes, th_data->thresh);
  free_detections(dets, nboxes);

  free_image(queue_image.frame);
  free_image(sized);
 }
 return nullptr;
}

yolo_status yolo_check_before_process_filename(yolo_object *yolo, char *filename)
{
 if(yolo == nullptr)
 {
  return yolo_object_is_not_initialized;
 }

 if(access(filename, F_OK) == -1)
 {
  fprintf(stderr, "error yolo_detect: %s\n", strerror(errno));
  return yolo_file_is_not_exists;
 }

 if(access(filename, R_OK) == -1)
 {
  fprintf(stderr, "error yolo_detect: %s\n", strerror(errno));
  return yolo_file_is_not_readable;
 }
 return yolo_ok;
}

void yolo_free(void *yolo)
{
 if(yolo == nullptr)
 {
  return;
 }
 yolo_object *yolo_instance=static_cast<yolo_object *>(yolo);
 if(yolo_instance->net != nullptr)
 {
  free_network(yolo_instance->net);
 }

 if(yolo_instance->names != nullptr)
 {
  for(int i=0; i<yolo_instance->class_number; i++)
  {
   if(yolo_instance->names[i] != nullptr)
   {
    free(yolo_instance->names[i]);
   }
  }
  free(yolo_instance->names);
 }
 free(yolo_instance);
 yolo_object **ptr_yolo=&yolo_instance;
 (*ptr_yolo)=nullptr;
}

yolo_status yolo_init(void **yolo_obj, char *workingDir, char *datacfg, char *cfgfile, char *weightfile)
{
 clock_t time=clock();
 if(yolo_obj == nullptr)
 {
  return yolo_object_is_not_initialized;
 }
 yolo_object **yolo_instance=(yolo_object **)yolo_obj;
 yolo_free((*yolo_instance));

 (*yolo_obj)=(yolo_object *)malloc(sizeof(yolo_object));

 yolo_object *yolo=(*yolo_instance);
 if(!yolo)
 {
  return yolo_cannot_alloc_node_yolo_object;
 }
 memset(yolo, 0, sizeof(yolo_object));

 if(access(workingDir, F_OK) == -1)
 {
  fprintf(stderr, "error yolo_init: %s\n", strerror(errno));
  return yolo_working_dir_is_not_exists;
 }

 if(access(workingDir, R_OK) == -1)
 {
  fprintf(stderr, "error yolo_init: %s\n", strerror(errno));
  return yolo_working_dir_is_not_readable;
 }
 char cur_dir[1024];
 getcwd(cur_dir, sizeof(cur_dir));
 if(chdir(workingDir) == -1)
 {
  fprintf(stderr, "%s\n", strerror(errno));
  return yolo_cannot_change_to_working_dir;
 }

 if(access(cfgfile, F_OK) == -1)
 {
  fprintf(stderr, "error yolo_init: %s\n", strerror(errno));
  return yolo_cfgfile_is_not_exists;
 }
 if(access(cfgfile, R_OK) == -1)
 {
  fprintf(stderr, "error yolo_init: %s\n", strerror(errno));
  return yolo_cfgfile_is_not_readable;
 }
 if(access(weightfile, F_OK) == -1)
 {
  fprintf(stderr, "error yolo_init: %s\n", strerror(errno));
  return yolo_weight_file_is_not_exists;
 }
 if(access(weightfile, R_OK) == -1)
 {
  fprintf(stderr, "error yolo_init: %s\n", strerror(errno));
  return yolo_weight_file_is_not_readable;
 }
 yolo->net=load_network(cfgfile, weightfile, 0);

 if(access(datacfg, F_OK) == -1)
 {
  fprintf(stderr, "error yolo_init: %s\n", strerror(errno));
  return yolo_datacfg_is_not_exists;
 }

 if(access(datacfg, R_OK) == -1)
 {
  fprintf(stderr, "error yolo_init: %s\n", strerror(errno));
  return yolo_datacfg_is_not_readable;
 }

 list *options=read_data_cfg(datacfg);
 char *name_list=option_find_str(options, (char *)"names", (char *)"data/names.list");

 if(access(name_list, F_OK) == -1)
 {
  fprintf(stderr, "error yolo_init: %s\n", strerror(errno));
  return yolo_names_file_is_not_exists;
 }
 if(access(name_list, R_OK) == -1)
 {
  fprintf(stderr, "error yolo_init: %s\n", strerror(errno));
  return yolo_names_file_is_not_readable;
 }
 yolo->names=get_labels(name_list);
 char *classes=option_find_str(options, (char *)"classes", (char *)"data/names.list");
 char *bad_ptr=nullptr;
 long value=strtol(classes, &bad_ptr, 10);
 if(value<INT_MAX)
 {
  yolo->class_number=(int)value;
 }

 set_batch_network(yolo->net, 1);
 srand(2222222);

 printf("Network configured and loaded in %f seconds\n", sec(clock()-time));
 chdir(cur_dir);
 return yolo_ok;
}

yolo_status yolo_detect_image(void *yolo, image_detection **detect, char *filename, float thresh)
{
 if(yolo == nullptr)
 {
  return yolo_object_is_not_initialized;
 }
 yolo_object *yolo_instance=static_cast<yolo_object *>(yolo);
 yolo_status status=yolo_check_before_process_filename(yolo_instance, filename);
 if(status != yolo_ok)
 {
  return status;
 }

 cv::Mat mat=cv::imread(filename, CV_LOAD_IMAGE_COLOR);
 if(mat.empty())
 {
  fprintf(stderr, "error yolo_detect: %s\n", strerror(errno));
  return yolo_file_is_corrupted;
 }

 layer l=yolo_instance->net->layers[yolo_instance->net->n-1];
 unsigned long long time;
 float nms=0.45;

 image im=libyolo_mat_to_image(mat);
 mat.release();

 image sized=resize_image(im, yolo_instance->net->w, yolo_instance->net->h);
 float *X=sized.data;
 time=unixTimeMillis();
 network_predict(yolo_instance->net, X);

 int nboxes=0;
 detection *dets=get_network_boxes(yolo_instance->net, im.w, im.h, thresh, 0.5, nullptr, 0, &nboxes);
 if(nms>0)
 {
  do_nms_sort(dets, l.side*l.side*l.n, l.classes, nms);
 }

 status=parse_detections_image(yolo_instance, dets, detect, unixTimeMillis()-time, nboxes, thresh);
 if(status != yolo_ok)
 {
  return status;
 }

 free_detections(dets, nboxes);
 free_image(im);
 free_image(sized);

 return yolo_ok;
}

yolo_status yolo_detect_video(void *yolo, video_detection **detect, char *filename, float thresh, double fraction_frames_to_process)
{
 if(yolo == nullptr)
 {
  return yolo_object_is_not_initialized;
 }
 yolo_object *yolo_instance=static_cast<yolo_object *>(yolo);
 yolo_status status=yolo_check_before_process_filename(yolo_instance, filename);
 if(status != yolo_ok)
 {
  return status;
 }
 const size_t num_capture_image_threads=2;
 pthread_t *capture_image_thread;
 pthread_t process_image_thread;
 cv::VideoCapture *capture;

 thread_image_queue_t image_queue;
 thread_common_t data_image_common;
 thread_get_frame_t data_get_image;
 thread_processing_image_t data_process_image;

 data_image_common.end=false;
 image_queue.queue=std::deque<queue_image_t>();
 image_queue.common=&data_image_common;

 data_get_image.image_queue=data_process_image.image_queue=&image_queue;

 data_process_image.yolo=yolo_instance;
 data_process_image.thresh=thresh;
 data_process_image.yolo_detect=detect;

 data_get_image.number_frames_to_process_simultaneously=data_get_image.number_frames_to_drop=(unsigned int)floor((1/fraction_frames_to_process)-1);

 if(pthread_mutex_init(&data_get_image.mutex, nullptr))
 {
  return yolo_video_cannot_alloc_base_structure;
 }

 if(pthread_mutex_init(&image_queue.mutex, nullptr))
 {
  return yolo_video_cannot_alloc_base_structure;
 }

 image_queue.empty=sem_open("/image_empty", O_CREAT, 0644, 20);
 if(image_queue.empty == SEM_FAILED)
 {
  return yolo_video_cannot_alloc_base_structure;
 }
 image_queue.full=sem_open("/image_full", O_CREAT, 0644, 0);
 if(image_queue.full == SEM_FAILED)
 {
  return yolo_video_cannot_alloc_base_structure;
 }

 capture=new cv::VideoCapture(filename);
 if(!capture->isOpened())
 {
  return yolo_cannot_open_video_stream;
 }
 data_get_image.video=capture;

 capture_image_thread=(pthread_t *)calloc(num_capture_image_threads, sizeof(pthread_t));
 if(capture_image_thread == nullptr)
 {
  return yolo_video_cannot_alloc_base_structure;
 }

 for(size_t i=0; i<num_capture_image_threads; ++i)
 {
  pthread_create(capture_image_thread+i, nullptr, thread_capture, &data_get_image);
 }
 pthread_create(&process_image_thread, nullptr, thread_detect, &data_process_image);

 for(size_t i=0; i<num_capture_image_threads; ++i)
 {
  pthread_join(capture_image_thread[i], nullptr);
 }
 capture->release();
 delete capture;
 free(capture_image_thread);

 pthread_join(process_image_thread, nullptr);

 sem_close(image_queue.full);
 sem_close(image_queue.empty);
 pthread_mutex_destroy(&image_queue.mutex);
 image_queue.queue.clear();

 pthread_mutex_destroy(&data_get_image.mutex);
 return yolo_ok;
}

void yolo_detect_free(image_detection *yolo_det)
{
 for(size_t i=0; i<yolo_det->num_detections; i++)
 {
  free(yolo_det->object_detections[i].class_name);
 }
 free(yolo_det->object_detections);
}

void image_detection_free(image_detection **p_image_detection)
{
 if((*p_image_detection) == nullptr)
 {
  return;
 }
 yolo_detect_free(*p_image_detection);
 free(*p_image_detection);
 (*p_image_detection)=nullptr;
}

void video_detection_free(video_detection **p_video_detection)
{
 video_detection *video_detection=*p_video_detection;
 if(video_detection == nullptr)
 {
  return;
 }
 for(size_t i=0; i<video_detection->num_frames; ++i)
 {
  yolo_detect_free(&video_detection->frame_detections[i].image_detection);
 }
 free(video_detection->frame_detections);
 free(video_detection);
 (*p_video_detection)=nullptr;
}

#ifdef __cplusplus

#include <stdexcept>

Yolo::Yolo(std::string workingDir, std::string datacfg, std::string cfgfile, std::string weightfile)
{
 this->yolo=nullptr;
 yolo_status status=yolo_init(&this->yolo, (char *)workingDir.c_str(), (char *)datacfg.c_str(), (char *)cfgfile.c_str(), (char *)weightfile.c_str());
 if(status != yolo_ok)
 {
  //TODO Complete this with some custom error
  throw std::invalid_argument("...");
 }
}

Yolo::~Yolo()
{
 yolo_free(this->yolo);
 this->yolo=nullptr;
}

image_detection *Yolo::detectImage(std::string filename, float thresh)
{

}

yolo_status Yolo::detectVideo(video_detection **detect, std::string filename, float thresh, double fraction_frames_to_process)
{

}

void Yolo::image_detection_free(image_detection **p_image_detection)
{
}

void Yolo::video_detection_free(video_detection **p_video_detection)
{
}

#endif