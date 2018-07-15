#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;
static CvVideoWriter * writer = 0;
static CvVideoWriter * stream_writer = 0;
static CvVideoWriter * vpre_writer = 0;
static IplImage *disp = 0;
static IplImage *disp_vpre = 0;


typedef struct thread_args {
    int enable_mqtt;
    char *topic;
    char *vpre;
    int fps;
} thread_args;


static float get_pixel(image m, int x, int y, int c) {
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
    ///return get_pixel(m, x, y, c);
}


detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);

int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}

void remember_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for(j = 0; j < demo_frame; ++j){
        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}


void do_write_vpre(image im, const char * vpre, int fps)
{

    int x,y,k;
    //static int frame_counter = 1;

    if(!fps) {
      fps = 25;
    }
 
    if(!disp_vpre) {
      disp_vpre = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    }

    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    
    
    int step = disp_vpre->widthStep;
    
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(k= 0; k < im.c; ++k){
                disp_vpre->imageData[y*step + x*im.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
            }
        }
    }
    free_image(copy);


    
    CvSize size;
    size.width = disp_vpre->width;
    size.height = disp_vpre->height;

    
    if (!vpre_writer)
    {
         // printf("\n SRC output_video = %p \n", writer);
      vpre_writer = cvCreateVideoWriter(vpre, CV_FOURCC('D', 'I', 'V', 'X'), fps, size, 1);
         // printf("\n cvCreateVideoWriter, DST output_viwriterdeo = %p  \n", writer);
          
    }

    cvWriteFrame(vpre_writer, disp_vpre);
      
}


void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;
    int enable_mqtt = 0;
    char *topic = NULL;
    char *vpre = NULL;
    int fps = 25;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);

    thread_args *detect_thread_args = (thread_args*)ptr;
    if(detect_thread_args->enable_mqtt) {
        enable_mqtt = detect_thread_args->enable_mqtt;
    }
    if(detect_thread_args->topic) {
        topic = detect_thread_args->topic;
    }
    if(detect_thread_args->vpre) {
        vpre = detect_thread_args->vpre;
    }
    if (detect_thread_args->fps) {
        fps = detect_thread_args->fps;
    }
  
    
    //printf("Topic : %s, enable_mqtt : %d\n", topic, enable_mqtt);


    /*
       if(l.type == DETECTION){
       get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
       } else */
    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);


    /*
       int i,j;
       box zero = {0};
       int classes = l.classes;
       for(i = 0; i < demo_detections; ++i){
       avg[i].objectness = 0;
       avg[i].bbox = zero;
       memset(avg[i].prob, 0, classes*sizeof(float));
       for(j = 0; j < demo_frame; ++j){
       axpy_cpu(classes, 1./demo_frame, dets[j][i].prob, 1, avg[i].prob, 1);
       avg[i].objectness += dets[j][i].objectness * 1./demo_frame;
       avg[i].bbox.x += dets[j][i].bbox.x * 1./demo_frame;
       avg[i].bbox.y += dets[j][i].bbox.y * 1./demo_frame;
       avg[i].bbox.w += dets[j][i].bbox.w * 1./demo_frame;
       avg[i].bbox.h += dets[j][i].bbox.h * 1./demo_frame;
       }
    //copy_cpu(classes, dets[0][i].prob, 1, avg[i].prob, 1);
    //avg[i].objectness = dets[0][i].objectness;
    }
     */

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = buff[(buff_index+2) % 3];
    // write image to vpre_writer
    if(vpre) {
        //write it here 
        do_write_vpre(display, vpre, fps);
    }
    draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, enable_mqtt, topic);
    free_detections(dets, nboxes);

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

void *fetch_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *display_in_thread(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}


static int VideoWriter_fourcc(char c1, char c2, char c3, char c4)
{
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24);
}


void dowrite(image im, const char * voutput, int stream, int fps)
{

    int x,y,k;
    //static int frame_counter = 1;

    if(!fps) {
      fps = 25;
    }

    if(!disp) {
      disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    }

    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    
    
    int step = disp->widthStep;
    
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(k= 0; k < im.c; ++k){
                disp->imageData[y*step + x*im.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
            }
        }
    }
    free_image(copy);


    
    CvSize size;
    size.width = disp->width;
    size.height = disp->height;

    if(voutput) {
        if (!writer)
        {
         // printf("\n SRC output_video = %p \n", writer);
          writer = cvCreateVideoWriter(voutput, CV_FOURCC('D', 'I', 'V', 'X'), fps, size, 1);
         // printf("\n cvCreateVideoWriter, DST output_viwriterdeo = %p  \n", writer);
          
        }

        cvWriteFrame(writer, disp);
    }


    if(stream) {
        if(!stream_writer) {
            char *gstreamer_pipeline = "appsrc ! videoconvert ! videoscale ! video/x-raw,width=320,height=240 ! clockoverlay shaded-background=true font-desc=\"Sans 28\" ! theoraenc ! oggmux ! tcpserversink host=127.0.0.1 port=8080 ";
            stream_writer = cvCreateVideoWriter(gstreamer_pipeline, 0, fps, size, 1);
        }

        cvWriteFrame(stream_writer, disp);
    }  
  
}




void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen, int enable_mqtt, char *voutput, char *topic, int stream, char *vpre)
{
    //demo_frame = avg_frames;
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;
    
    thread_args detect_thread_args;
    detect_thread_args.topic = topic;
    detect_thread_args.enable_mqtt = enable_mqtt;
    detect_thread_args.fps = frames;
    detect_thread_args.vpre = vpre;

    int odd_even_flag = 0;

    int fps_dowrite = frames;
    if(!voutput && stream && vpre && fps) {
      fps_dowrite = fps_dowrite/2;
      odd_even_flag = 1;
    }
    
    srand(2222222);

    

    int i;
    demo_total = size_network(net);
    predictions = calloc(demo_frame, sizeof(float*));
    for (i = 0; i < demo_frame; ++i){
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }

    demo_time = what_time_is_it_now();

    while(!demo_done){
        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, (void *)&detect_thread_args)) error("Thread creation failed");
        if(!prefix){
            fps = 1./(what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
            display_in_thread(0);
        }else{
            //char name[256];
            //sprintf(name, "%s_%08d", prefix, count);
            //save_image(buff[(buff_index + 1)%3], name);
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        if(voutput || stream) {
            if(odd_even_flag) {
              if(count % 2 == 0) {
                 dowrite(buff[(buff_index + 1)%3], voutput, stream, fps_dowrite);
              } 
            }
            else {
              dowrite(buff[(buff_index + 1)%3], voutput, stream, fps_dowrite); 
            }  
        }
        ++count;
    }

    if(writer) {
       // printf("Releasing video");
        cvReleaseVideoWriter(&writer);
        writer = 0;
    }
    if(stream_writer) {
      cvReleaseVideoWriter(&stream_writer);
      stream_writer = 0;
    }
    if(vpre_writer) {
      cvReleaseVideoWriter(&vpre_writer);
      vpre_writer = 0;
    }

}

/*
   void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
   {
   demo_frame = avg_frames;
   predictions = calloc(demo_frame, sizeof(float*));
   image **alphabet = load_alphabet();
   demo_names = names;
   demo_alphabet = alphabet;
   demo_classes = classes;
   demo_thresh = thresh;
   demo_hier = hier;
   printf("Demo\n");
   net = load_network(cfg1, weight1, 0);
   set_batch_network(net, 1);
   pthread_t detect_thread;
   pthread_t fetch_thread;

   srand(2222222);

   if(filename){
   printf("video file: %s\n", filename);
   cap = cvCaptureFromFile(filename);
   }else{
   cap = cvCaptureFromCAM(cam_index);

   if(w){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
   }
   if(h){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
   }
   if(frames){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
   }
   }

   if(!cap) error("Couldn't connect to webcam.\n");

   layer l = net->layers[net->n-1];
   demo_detections = l.n*l.w*l.h;
   int j;

   avg = (float *) calloc(l.outputs, sizeof(float));
   for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

   boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
   probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
   for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

   buff[0] = get_image_from_stream(cap);
   buff[1] = copy_image(buff[0]);
   buff[2] = copy_image(buff[0]);
   buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
   ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

   int count = 0;
   if(!prefix){
   cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
   if(fullscreen){
   cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
   } else {
   cvMoveWindow("Demo", 0, 0);
   cvResizeWindow("Demo", 1352, 1013);
   }
   }

   demo_time = what_time_is_it_now();

   while(!demo_done){
buff_index = (buff_index + 1) %3;
if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
if(!prefix){
    fps = 1./(what_time_is_it_now() - demo_time);
    demo_time = what_time_is_it_now();
    display_in_thread(0);
}else{
    char name[256];
    sprintf(name, "%s_%08d", prefix, count);
    save_image(buff[(buff_index + 1)%3], name);
}
pthread_join(fetch_thread, 0);
pthread_join(detect_thread, 0);
++count;
}
}
*/
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen, int enable_mqtt, char *voutput, int stream, char *vpre)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

