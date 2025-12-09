
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// 将浮点型 Mat 归一化并保存为 8-bit 图像，便于观察中间结果
static void saveFloatMatNormalized(const Mat &srcFloat, const string &filename) {
    Mat dst;
    // 将最小值映射到 0，最大值映射到 255
    double minVal, maxVal;
    minMaxLoc(srcFloat, &minVal, &maxVal);
    if (maxVal - minVal < 1e-6) {
        // 常数图像，直接保存为常量灰度
        Mat tmp(srcFloat.size(), CV_8U, Scalar(cvRound(minVal)));
        imwrite(filename, tmp);
        return;
    }
    // 归一化到 0-255
    Mat normalized;
    srcFloat.convertTo(normalized, CV_32F, 1.0, 0.0);
    normalized = (normalized - float(minVal)) * (255.0f / float(maxVal - minVal));
    Mat out8u;
    normalized.convertTo(out8u, CV_8U);
    imwrite(filename, out8u);
}

Mat generate_emboss_style_cpp(const string &image_path) {
    // 1. 加载图像并转换为灰度图
    Mat img_bgr = imread(image_path, IMREAD_COLOR);
    if (img_bgr.empty()) {
        cerr << "Error: cannot read image at " << image_path << endl;
        return Mat();
    }

    Mat gray;
    cvtColor(img_bgr, gray, COLOR_BGR2GRAY);

    // 将图像转换为浮点型，以进行精确的拉普拉斯计算
    Mat gray_float;
    gray.convertTo(gray_float, CV_32F);

    // 2. 预平滑/降噪（高斯模糊）
    Mat smoothed;
    GaussianBlur(gray_float, smoothed, Size(7, 7), 0);
    saveFloatMatNormalized(smoothed, "smoothed.jpg");

    // 3. 核心：应用拉普拉斯算子（输出为 CV_32F）
    Mat laplacian;
    Laplacian(gray_float, laplacian, CV_32F, 3);
    saveFloatMatNormalized(laplacian, "edge.jpg");

    // 将 Laplacian 与平滑图相加（保持 float）
    Mat lap_plus_smoothed;
    add(laplacian, smoothed, lap_plus_smoothed);
    saveFloatMatNormalized(lap_plus_smoothed, "edge_smoothed.jpg");

    // 4. 转换和偏移 (alpha * L + beta)
    float alpha = 0.3f;
    float beta = 128.0f;
    Mat laplacian_scaled = lap_plus_smoothed * alpha;
    laplacian_scaled = laplacian_scaled + beta; // still CV_32F
    saveFloatMatNormalized(laplacian_scaled, "q1.jpg");

    // 5. 限制像素值到 [0,255] 并转换为 8 位无符号整数
    Mat output_8u;
    // convertTo 会进行截断（saturate_cast），相当于 np.clip + astype(uint8)
    laplacian_scaled.convertTo(output_8u, CV_8U);

    // 6. 调整对比度和亮度（这里与 Python 保持一致）
    Mat output_final;
    // Python 中使用 cv2.convertScaleAbs(output_8u, alpha=-1.0, beta=0)
    // 在 C++ 中同样调用 convertScaleAbs
    convertScaleAbs(output_8u, output_final, -1.0, 0.0);

    // 保存最终结果
    imwrite("transformed_output.jpg", output_final);

    return output_final;
}

int main(int argc, char **argv) {
    string input_file = "lena.jpg";
    if (argc > 1) input_file = argv[1];

    Mat result = generate_emboss_style_cpp(input_file);
    if (result.empty()) {
        cerr << "Processing failed." << endl;
        return 1;
    }

    cout << "Saved transformed_output.jpg and intermediate images (smoothed.jpg, edge.jpg, edge_smoothed.jpg, q1.jpg)." << endl;
    return 0;
}
