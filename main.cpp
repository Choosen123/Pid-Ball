#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "onnxruntime_cxx_api.h"
#include <cmath>
#include "serialport/serialport.h"

#define ORINGIN_X 328
#define ORINGIN_y 343
#define RADIUS 15

using Array = std::vector<float>;
using Shape = std::vector<long>;

using namespace std;
using namespace std::chrono;

bool use_cuda = true;
int image_size = 640;
double rate = 170.0/536.0;
std::string model_path = "/home/homesickalien/pid_ball/batch_8_ep_200_yolov7.onnx";
std::string image_path = "/home/homesickalien/pid_ball/pid_ball.mp4";

cv::Point origin(ORINGIN_X,ORINGIN_y);

const char *class_names[] = {"ball"};

std::tuple<Array, Shape, cv::Mat> read_image(cv::Mat image_camera, int size)
{
  
    //确保图像不为空且通道数为3
    assert(!image_camera.empty() && image_camera.channels() == 3);
    //调整图片大小
    int width = image_camera.cols;
    int height = image_camera.rows;
    float scale = std::min(static_cast<float>(640) / width, 
                          static_cast<float>(640) / height);
    
    int new_width = static_cast<int>(width * scale);
    int new_height = static_cast<int>(height * scale);
    
    cv::Mat resized;
    cv::resize(image_camera, resized, cv::Size(new_width, new_height));
    
    cv::Mat padded = cv::Mat::zeros(640, 640, CV_8UC3);
    resized.copyTo(padded(cv::Rect((640 - new_width) / 2, 
                                  (640 - new_height) / 2, 
                                  new_width, new_height)));
    //cv::resize(image_camera, image_camera, {size, size});
    //储存图片数据
    Shape shape = {1, padded.channels(), padded.rows, padded.cols};
    //
    cv::Mat nchw = cv::dnn::blobFromImage(padded, 1.0, {}, {}, true) / 255.f;
    Array array(nchw.ptr<float>(), nchw.ptr<float>() + nchw.total());
    return {array, shape, padded};
}

std::pair<Array, Shape> process_image(Ort::Session &session, Array &array, Shape shape)
{
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto input = Ort::Value::CreateTensor<float>(
        memory_info, (float *)array.data(), array.size(), shape.data(), shape.size());

    const char *input_names[] = {"images"};
    const char *output_names[] = {"output"};
    auto output = session.Run({}, input_names, &input, 1, output_names, 1);
    shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
    auto ptr = output[0].GetTensorData<float>();
    return {Array(ptr, ptr + shape[0] * shape[1]), shape};
}

void display_image(cv::Mat image, const Array &output, const Shape &shape ,double & center_x,double & center_y)
{
    
    for (size_t i = 0; i < shape[0]; ++i)
    {
        auto ptr = output.data() + i * shape[1];
     
        int x = ptr[1], y = ptr[2], w = ptr[3] - x, h = ptr[4] - y, c = ptr[5];
        center_x = x+w/2, center_y = y+h/2;
        auto color = CV_RGB(255, 255, 255);
        auto name = std::string(class_names[c]) + ":" + std::to_string(int(ptr[6] * 100)) + "%";
        cv::rectangle(image, {x, y, w, h}, color);
        cv::putText(image, name, {x, y}, cv::FONT_HERSHEY_DUPLEX, 0.5, color);
    }

    cv::imshow("YOLOv7 Output", image);

    cv::waitKey(10);
}

void show_cor(cv::Mat image,double center_x,double center_y){
    cv::Point pointx(20,20),pointy(20,50),points(20,80);
    cv::putText(image,"ball_x:"+to_string(center_x),
                pointx,cv::FONT_HERSHEY_COMPLEX,
                0.5,cv::Scalar(0,255,0),0.5);
    cv::putText(image,"ball_y:"+to_string(center_y),
                pointy,cv::FONT_HERSHEY_COMPLEX,
                0.5,cv::Scalar(0,255,0),0.5);
}

double calculate_position(double angle,double distance,double r){
    return (distance/cos(angle))+r*tan(angle);
}

int main(){
    io_service stm;
    MySerial sp("/dev/ttyUSB0",115200,8,stm);
    
    cv::Mat test_img = cv::imread("/home/homesickalien/chess/chess/programe/frame0180.jpg");
    
    double angle=30.0,angle_tran;
    
    
    // cv::VideoCapture cap(image_path);
    // cv::VideoCapture cap("/dev/camera1");
    cv::VideoCapture cap("/dev/video2");
    // cv::VideoCapture cap("http://admin:admin@192.168.42.29:8081");
    cv::Mat image_camera;
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv7");
    Ort::SessionOptions options;
    if(use_cuda) Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0));
    Ort::Session session(env, model_path.c_str(), options);
    
    cap.read(image_camera);
    
    std::thread io_thread([&stm]() {
        stm.run(); // 持续处理事件，直到 io_service 停止
    });
    
    int x_double,y_double;

    double center_x=0.0,center_y=0.0;
    double distance=0.0;
    double delta_time;
    // double position;
    
    while(1){
        if(!cap.read(image_camera) || !sp.init_and_open()){
            cout<<"break"<<endl;
            break;
        }else{
            try{
                auto pre_time=high_resolution_clock::now();
                
                auto [input_data, shape, image] = read_image(image_camera,image_size);
                
                auto [output_data, output_shape] = process_image(session, input_data, shape);
                
                if(!output_shape.empty()){
                    
                    for (size_t i = 0; i < shape[0]; ++i)
                    {
                        auto ptr = output_data.data() + i * shape[1];
                        
                        int x = ptr[1], y = ptr[2], w = ptr[3] - x, h = ptr[4] - y, c = ptr[5];
                        x_double = (x+ptr[3])/2, y_double = (y+ptr[4])/2;
                        cout<<x_double<<' '<<y_double<<endl;
                        center_x = static_cast<uint8_t>(x_double), center_y = static_cast<uint8_t>(y_double);
                        
                        distance = static_cast<uint8_t>(sqrt((x_double-ORINGIN_X)*(x_double-ORINGIN_X)+(y_double-ORINGIN_y)*(y_double-ORINGIN_y)));
                        
                        sp.flush_buffer();
                        sp.my_async_write();
                    
                        angle_tran = (sp.angle*1.0/10-12.8)*M_PI/180.0;

                        if(x_double>=ORINGIN_X)
                            sp.position=static_cast<uint8_t>(calculate_position(angle_tran,distance,w/2)*rate);
                        else
                            sp.position=-static_cast<uint8_t>(calculate_position(angle_tran,distance,w/2)*rate);
                        sp.position+=75;
                        sp.my_async_read();
                        stm.run();
                        
                        auto current_time = high_resolution_clock::now();
                        delta_time = duration_cast<duration<double>>(current_time-pre_time).count();

                        auto color = CV_RGB(255, 255, 255);
                        auto name = std::string(class_names[c]) + ":" + std::to_string(int(ptr[6] * 100)) + "%";
                        cv::rectangle(image, {x, y, w, h}, color);
                        // cv::putText(image, name, {x, y}, cv::FONT_HERSHEY_DUPLEX, 0.5, color);
                        cv::putText(image,to_string(delta_time), {x, y}, cv::FONT_HERSHEY_DUPLEX, 0.5, color);
                    }

                show_cor(image,center_x,center_y);
            
                cv::imshow("YOLOv7 Output", image);
            
                cv::waitKey(1);
            
                }else{
                    cv::imshow("YOLOv7 Output",image_camera);
                }
            }catch(const exception& e){
                cerr << "Exception caught: " << e.what() << endl;
            }catch(...){
                cerr << "Unknown exception caught!" << endl;
            }
        }
    }

    stm.stop();
    io_thread.join();

    return 0;
}