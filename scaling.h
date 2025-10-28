#ifndef SCALING_H
#define SCALING_H

#include <vector>
#include <opencv2/opencv.hpp>

typedef unsigned char byte;

// 图像缩放函数
// 输入：原始图像矩阵、目标宽度、目标高度、填充灰度值（0-255）
// 输出：640x640的缩放结果，保持原始比例，上下填充灰色
std::vector<std::vector<byte>> scale_image(
    const std::vector<std::vector<byte>>& input,
    int target_width,
    int target_height,
    byte fill_color = 128  // 默认使用中灰色填充
);

#endif // SCALING_H