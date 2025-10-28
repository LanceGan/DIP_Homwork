#ifndef ROTATING_H
#define ROTATING_H

#include <vector>
#include <cmath>

typedef unsigned char byte;

std::vector<std::vector<double>> matrixMultiplication(const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B);
std::vector<std::vector<double>> invertMatrix(const std::vector<std::vector<double>>& matrix);
byte bilinearInterpolation(const std::vector<std::vector<byte>>& img, double x, double y);
std::vector<double> applyTransform(const std::vector<std::vector<double>>& T, double x, double y);
std::vector<std::vector<byte>> rotating_func(
    const std::vector<std::vector<byte>>& f,
    float x0, float y0, float x1, float y1,
    int ow, int oh, float a);   

#endif // ROTATING_H