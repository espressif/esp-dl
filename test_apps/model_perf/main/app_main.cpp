#include "cat_detect.hpp"
#include "coco_detect.hpp"
#include "coco_pose.hpp"
#include "dl_image_jpeg.hpp"
#include "dog_detect.hpp"
#include "esp_log.h"
#include "human_face_detect.hpp"
#include "human_face_recognition.hpp"
#include "imagenet_cls.hpp"
#include "pedestrian_detect.hpp"
#include "bsp/esp-bsp.h"

extern const uint8_t bus_jpg_start[] asm("_binary_bus_jpg_start");
extern const uint8_t bus_jpg_end[] asm("_binary_bus_jpg_end");

static const char *TAG = "model_perf";

extern "C" void app_main(void)
{
    ESP_ERROR_CHECK(bsp_sdcard_mount());

    dl::image::jpeg_img_t jpeg_img = {.data = (void *)bus_jpg_start, .data_len = (size_t)(bus_jpg_end - bus_jpg_start)};
    auto img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);

    ESP_LOGI(TAG, "cat_224");
    CatDetect *cat_224 = new CatDetect(CatDetect::ESPDET_PICO_224_224_CAT);
    cat_224->run(img);
    delete cat_224;

    ESP_LOGI(TAG, "cat_416");
    CatDetect *cat_416 = new CatDetect(CatDetect::ESPDET_PICO_416_416_CAT);
    cat_416->run(img);
    delete cat_416;

    ESP_LOGI(TAG, "yolo11n_v1");
    COCODetect *yolo11n_v1 = new COCODetect(COCODetect::YOLO11N_S8_V1);
    yolo11n_v1->run(img);
    delete yolo11n_v1;

#if !CONFIG_IDF_TARGET_ESP32S3
    ESP_LOGI(TAG, "yolo11n_v2");
    COCODetect *yolo11n_v2 = new COCODetect(COCODetect::YOLO11N_S8_V2);
    yolo11n_v2->run(img);
    delete yolo11n_v2;
#endif

    ESP_LOGI(TAG, "yolo11n_v3");
    COCODetect *yolo11n_v3 = new COCODetect(COCODetect::YOLO11N_S8_V3);
    yolo11n_v3->run(img);
    delete yolo11n_v3;

    ESP_LOGI(TAG, "yolo11n_320");
    COCODetect *yolo11n_320_v3 = new COCODetect(COCODetect::YOLO11N_320_S8_V3);
    yolo11n_320_v3->run(img);
    delete yolo11n_320_v3;

    ESP_LOGI(TAG, "yolo11n_pose_v1");
    COCOPose *yolo11n_pose_v1 = new COCOPose(COCOPose::YOLO11N_POSE_S8_V1);
    yolo11n_pose_v1->run(img);
    delete yolo11n_pose_v1;

    ESP_LOGI(TAG, "yolo11n_pose_v2");
    COCOPose *yolo11n_pose_v2 = new COCOPose(COCOPose::YOLO11N_POSE_S8_V2);
    yolo11n_pose_v2->run(img);
    delete yolo11n_pose_v2;

    ESP_LOGI(TAG, "dog_224");
    DogDetect *dog_224 = new DogDetect(DogDetect::ESPDET_PICO_224_224_DOG);
    dog_224->run(img);
    delete dog_224;

    ESP_LOGI(TAG, "dog_416");
    DogDetect *dog_416 = new DogDetect(DogDetect::ESPDET_PICO_416_416_DOG);
    dog_416->run(img);
    delete dog_416;

    ESP_LOGI(TAG, "msr_mnp");
    HumanFaceDetect *msr_mnp = new HumanFaceDetect(HumanFaceDetect::MSRMNP_S8_V1);
    msr_mnp->run(img);
    delete msr_mnp;

    ESP_LOGI(TAG, "mfn");
    HumanFaceFeat *mfn = new HumanFaceFeat(HumanFaceFeat::MFN_S8_V1);
    mfn->run(img, {117, 114, 120, 160, 132, 143, 157, 11, 151, 160});
    delete mfn;

    ESP_LOGI(TAG, "mbf");
    HumanFaceFeat *mbf = new HumanFaceFeat(HumanFaceFeat::MBF_S8_V1);
    mbf->run(img, {117, 114, 120, 160, 132, 143, 157, 11, 151, 160});
    delete mbf;

    ESP_LOGI(TAG, "mobilenetv2");
    ImageNetCls *mobilenetv2 = new ImageNetCls(ImageNetCls::MOBILENETV2_S8_V1);
    mobilenetv2->run(img);
    delete mobilenetv2;

    ESP_LOGI(TAG, "pico");
    PedestrianDetect *pico = new PedestrianDetect(PedestrianDetect::PICO_S8_V1);
    pico->run(img);
    delete pico;

    heap_caps_free(img.data);
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
}
