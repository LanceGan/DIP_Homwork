#include "scaling.h"
#include <algorithm>
#include <cmath>

std::vector<std::vector<byte>> scale_image(
    const std::vector<std::vector<byte>>& input,
    int target_width,
    int target_height,
    byte fill_color
) {
    int src_height = input.size();
    int src_width = input[0].size();

    // 计算保持宽高比的缩放尺寸
    double src_aspect = static_cast<double>(src_width) / src_height;
    double target_aspect = static_cast<double>(target_width) / target_height;
    
    int scaled_width, scaled_height;
    if (src_aspect > target_aspect) {
        // 原图更宽，以宽度为准缩放
        scaled_width = target_width;
        scaled_height = static_cast<int>(target_width / src_aspect);
    } else {
        // 原图更高，以高度为准缩放
        scaled_height = target_height;
        scaled_width = static_cast<int>(target_height * src_aspect);
    }

    // 计算在目标图像中的位置（居中）
    int offset_x = (target_width - scaled_width) / 2;
    int offset_y = (target_height - scaled_height) / 2;

    // 创建目标图像（填充灰色）
    std::vector<std::vector<byte>> output(target_height, 
        std::vector<byte>(target_width, fill_color));

    // 双线性插值缩放
    for (int y = 0; y < scaled_height; ++y) {
        for (int x = 0; x < scaled_width; ++x) {
            // 计算源图像中的对应位置
            double src_x = x * (static_cast<double>(src_width) / scaled_width);
            double src_y = y * (static_cast<double>(src_height) / scaled_height);

            // 计算插值的四个点
            int x1 = static_cast<int>(std::floor(src_x));
            int y1 = static_cast<int>(std::floor(src_y));
            int x2 = std::min(x1 + 1, src_width - 1);
            int y2 = std::min(y1 + 1, src_height - 1);

            // 计算权重
            double dx = src_x - x1;
            double dy = src_y - y1;

            // 双线性插值
            byte value = static_cast<byte>(
                input[y1][x1] * (1 - dx) * (1 - dy) +
                input[y1][x2] * dx * (1 - dy) +
                input[y2][x1] * (1 - dx) * dy +
                input[y2][x2] * dx * dy
            );

            // 写入结果（注意加上偏移）
            output[y + offset_y][x + offset_x] = value;
        }
    }

    return output;
}