#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "rotating.h"
#include "scaling.h"

typedef unsigned char byte;

int main() {

    cv::Mat img = cv::imread("lena.jpg", cv::IMREAD_GRAYSCALE);
    assert(!img.empty() && "无法读取 lena.jpg,请确保文件存在于当前目录!");

    cv::Mat img_scaling = cv::imread("./figure_1920_1080.jpg", cv::IMREAD_GRAYSCALE);
    assert(!img_scaling.empty() && "无法读取 figure_1920_1080.jpg,请确保文件存在于当前目录!");

    // Mat 转二维 byte 数组
    std::vector<std::vector<byte>> input(img.rows, std::vector<byte>(img.cols));
    for (int i = 0; i < img.rows; ++i)
        for (int j = 0; j < img.cols; ++j)
            input[i][j] = img.at<byte>(i, j);
    
    //-------------------------------------------------执行图像旋转
    int x0 = img.cols / 2;   // 输入图像旋转中心
    int y0 = img.rows / 2;
    int ow = 600;           // 输出图像宽
    int oh = 600;           // 输出图像高
    int x1 = ow / 2;        // 输出图像旋转中心
    int y1 = oh / 2;

    auto rotated = rotating_func(input, x0, y0, x1, y1, ow, oh, -45);

    cv::Mat rotated_out(oh, ow, CV_8UC1);
    for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x)
            rotated_out.at<byte>(y, x) = rotated[y][x];

    cv::imwrite("rotated_figure.jpg", rotated_out);
    std::cout << "旋转完成！输出分辨率: " << ow << " × " << oh << std::endl;

    // Mat 转二维 byte 数组
    std::vector<std::vector<byte>> input_scale(img_scaling.rows, std::vector<byte>(img_scaling.cols));
    for (int i = 0; i < img_scaling.rows; ++i)
        for (int j = 0; j < img_scaling.cols; ++j)
            input_scale[i][j] = img_scaling.at<byte>(i, j);

    //--------------------------------------------------- 执行图像缩放，输出大小为640x640，使用灰色填充
    const int TARGET_SIZE = 640;
    auto scaled = scale_image(input_scale, TARGET_SIZE, TARGET_SIZE, 128);

    // 保存缩放结果
    cv::Mat scaled_out(TARGET_SIZE, TARGET_SIZE, CV_8UC1);
    for (int y = 0; y < TARGET_SIZE; ++y)
        for (int x = 0; x < TARGET_SIZE; ++x)
            scaled_out.at<byte>(y, x) = scaled[y][x];

    cv::imwrite("scaled_figure.jpg", scaled_out);
    std::cout << "缩放完成！输出分辨率: " << TARGET_SIZE << " × " << TARGET_SIZE << std::endl;

   
    return 0;
}
