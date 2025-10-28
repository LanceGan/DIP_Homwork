#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "rotating.h"
typedef unsigned char byte;

// -------------------- 3x3矩阵乘法 --------------------
std::vector<std::vector<double>> matrixMultiplication(const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));
    for (int i = 0; i < rowsA; i++)
        for (int j = 0; j < colsB; j++)
            for (int k = 0; k < colsA; k++)
                result[i][j] += A[i][k] * B[k][j];

    return result;
}

// -------------------- 3x3矩阵求逆 --------------------
std::vector<std::vector<double>> invertMatrix(const std::vector<std::vector<double>>& matrix) {
    if (matrix.size() != 3 || matrix[0].size() != 3)
        throw std::invalid_argument("Only 3x3 matrices are supported for inversion");

    double det =
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
        matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
        matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);

    if (std::abs(det) < 1e-10)
        throw std::invalid_argument("Matrix is singular, cannot be inverted");

    double invDet = 1.0 / det;
    std::vector<std::vector<double>> inv(3, std::vector<double>(3));

    inv[0][0] = (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * invDet;
    inv[0][1] = (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * invDet;
    inv[0][2] = (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * invDet;

    inv[1][0] = (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * invDet;
    inv[1][1] = (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * invDet;
    inv[1][2] = (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * invDet;

    inv[2][0] = (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * invDet;
    inv[2][1] = (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * invDet;
    inv[2][2] = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * invDet;

    return inv;
}

// -------------------- 双线性插值 --------------------
byte bilinearInterpolation(const std::vector<std::vector<byte>>& image, double x, double y) {
    int x1 = static_cast<int>(std::floor(x));
    int y1 = static_cast<int>(std::floor(y));
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    double dx = x - x1;
    double dy = y - y1;

    int height = image.size();
    int width = image[0].size();

    if (x1 < 0 || x2 >= width || y1 < 0 || y2 >= height)
        return 0;

    byte q11 = image[y1][x1];
    byte q12 = image[y2][x1];
    byte q21 = image[y1][x2];
    byte q22 = image[y2][x2];

    double val = q11 * (1 - dx) * (1 - dy)
        + q21 * dx * (1 - dy)
        + q12 * (1 - dx) * dy
        + q22 * dx * dy;

    return static_cast<byte>(std::clamp(std::round(val), 0.0, 255.0));
}

// -------------------- 应用仿射变换矩阵 --------------------
std::vector<double> applyTransform(const std::vector<std::vector<double>>& T, double x, double y) {
    std::vector<double> res(2);
    res[0] = T[0][0] * x + T[0][1] * y + T[0][2];
    res[1] = T[1][0] * x + T[1][1] * y + T[1][2];
    return res;
}

// -------------------- 主旋转函数 --------------------
std::vector<std::vector<byte>> rotating_func(
    const std::vector<std::vector<byte>>& f,
    float x0, float y0, float x1, float y1,
    int ow, int oh, float a)
{
    // a 为角度（度）
    double angle = a * M_PI / 180.0;
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);

    int inH = f.size();
    int inW = f[0].size();

    // ---------- 正向矩阵：从输入到输出 ----------
    std::vector<std::vector<double>> translateToOrigin = {
        {1, 0, -x0},
        {0, 1, -y0},
        {0, 0, 1}
    };
    std::vector<std::vector<double>> rotationMat = {
        {cosA, -sinA, 0},
        {sinA,  cosA, 0},
        {0, 0, 1}
    };
    std::vector<std::vector<double>> translateBack = {
        {1, 0, x1},
        {0, 1, y1},
        {0, 0, 1}
    };

    auto temp = matrixMultiplication(rotationMat, translateToOrigin);
    auto T = matrixMultiplication(translateBack, temp);

    // ---------- 逆矩阵：从输出到输入 ----------
    auto invT = invertMatrix(T);

    std::vector<std::vector<byte>> out(oh, std::vector<byte>(ow, 0));

    // ---------- 逆映射 + 双线性插值 ----------
    for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
            std::vector<double> src = applyTransform(invT, x, y);
            double sx = src[0];
            double sy = src[1];

            if (sx >= 0 && sx < inW - 1 && sy >= 0 && sy < inH - 1)
                out[y][x] = bilinearInterpolation(f, sx, sy);
        }
    }

    return out;
}

