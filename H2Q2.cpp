
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <random>
#include <vector>

using namespace cv;
using namespace std;

class CustomImageFilter {
public:
    CustomImageFilter(const string &image_path) {
        pil_loaded = false;
        cv2_image = imread(image_path, IMREAD_COLOR);
        if (cv2_image.empty()) {
            throw runtime_error("cv2 cannot read image '" + image_path + "'");
        }
        cv2_image_rgb = cv2_image.clone();
        cvtColor(cv2_image_rgb, cv2_image_rgb, COLOR_BGR2RGB);
    }

    // 保存左右对比图到文件
    void show_comparison(const Mat &original, const Mat &processed, const string &title1 = "Original", const string &title2 = "Processed") {
        Mat orig_vis = original.clone();
        Mat proc_vis = processed.clone();
        // Ensure both are 3-channel BGR for concatenation and saving
        if (orig_vis.channels() == 1) cvtColor(orig_vis, orig_vis, COLOR_GRAY2BGR);
        if (proc_vis.channels() == 1) cvtColor(proc_vis, proc_vis, COLOR_GRAY2BGR);

        // Resize processed to match original height if necessary
        if (orig_vis.rows != proc_vis.rows) {
            resize(proc_vis, proc_vis, Size(orig_vis.cols, orig_vis.rows));
        }

        Mat concat;
        hconcat(orig_vis, proc_vis, concat);

        // safe filename from titles
        string safe1, safe2;
        for (char c : title1) if (isalnum((unsigned char)c) || c=='_'||c==' ') safe1.push_back(c);
        for (char c : title2) if (isalnum((unsigned char)c) || c=='_'||c==' ') safe2.push_back(c);
        for (char &c : safe1) if (c==' ') c = '_';
        for (char &c : safe2) if (c==' ') c = '_';
        string out_name = "comparison_" + safe1 + "_" + safe2 + ".png";
        imwrite(out_name, concat);
        cout << "Saved comparison to " << out_name << endl;
    }

    Mat enhance_colors_vivid(double saturation_factor = 1.8, double contrast_factor = 1.3) {
        Mat hsv;
        cvtColor(cv2_image, hsv, COLOR_BGR2HSV);
        hsv.convertTo(hsv, CV_32F);
        vector<Mat> ch;
        split(hsv, ch);
        // ch[1] is S channel
        ch[1] = ch[1] * (float)saturation_factor;
        ch[1] = min(ch[1], 255.0f);
        ch[1].convertTo(ch[1], CV_8U);
        Mat hsv8;
        merge(ch, hsv8);
        Mat enhanced;
        cvtColor(hsv8, enhanced, COLOR_HSV2BGR);

        // 转到 LAB，应用 CLAHE 到 L 通道
        Mat lab;
        cvtColor(enhanced, lab, COLOR_BGR2Lab);
        vector<Mat> labch;
        split(lab, labch);
        Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8,8));
        Mat lclahe;
        clahe->apply(labch[0], lclahe);
        labch[0] = lclahe;
        Mat lab_out;
        merge(labch, lab_out);
        Mat result;
        cvtColor(lab_out, result, COLOR_Lab2BGR);
        return result;
    }

    Mat sharpen_image(double strength = 2.0) {
        // kernel sharpening
        Mat kernel = (Mat_<float>(3,3) << -1,-1,-1, -1,9,-1, -1,-1,-1);
        Mat sharpened;
        filter2D(cv2_image, sharpened, -1, kernel);

        Mat blurred;
        GaussianBlur(cv2_image, blurred, Size(0,0), 3);
        Mat unsharp_mask;
        addWeighted(cv2_image, 1.0 + strength, blurred, -strength, 0.0, unsharp_mask);

        Mat result;
        addWeighted(sharpened, 0.5, unsharp_mask, 0.5, 0.0, result);
        return result;
    }

    Mat oil_painting_effect(int size = 7, int dynRatio = 20) {
        Mat img = cv2_image.clone();
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        int h = img.rows, w = img.cols;
        Mat oil_img = Mat::zeros(img.size(), img.type());

        int r = size/2;
        // For each pixel, compute histogram of region and choose most frequent gray level
        for (int i = r; i < h - r; ++i) {
            for (int j = r; j < w - r; ++j) {
                // compute histogram for region
                int hist[256] = {0};
                for (int y = i - r; y <= i + r; ++y) {
                    const uchar* prow = gray.ptr<uchar>(y);
                    for (int x = j - r; x <= j + r; ++x) {
                        hist[prow[x]]++;
                    }
                }
                int max_idx = 0, max_val = hist[0];
                for (int k = 1; k < 256; ++k) if (hist[k] > max_val) { max_val = hist[k]; max_idx = k; }

                // compute mean color for pixels in region with gray == max_idx
                Vec3d sum(0,0,0);
                int count = 0;
                for (int y = i - r; y <= i + r; ++y) {
                    const uchar* prow = gray.ptr<uchar>(y);
                    const Vec3b* pcolor = img.ptr<Vec3b>(y);
                    for (int x = j - r; x <= j + r; ++x) {
                        if (prow[x] == (uchar)max_idx) {
                            Vec3b c = pcolor[x];
                            sum[0] += c[0]; sum[1] += c[1]; sum[2] += c[2];
                            ++count;
                        }
                    }
                }
                if (count > 0) {
                    Vec3b meanc((uchar)(sum[0]/count), (uchar)(sum[1]/count), (uchar)(sum[2]/count));
                    oil_img.at<Vec3b>(i,j) = meanc;
                } else {
                    oil_img.at<Vec3b>(i,j) = img.at<Vec3b>(i,j);
                }
            }
        }

        medianBlur(oil_img, oil_img, 3);
        return oil_img;
    }

    Mat vintage_filter() {
        Mat img;
        cv2_image_rgb.convertTo(img, CV_32F, 1.0/255.0);

        // img is RGB in cv2_image_rgb
        // Apply warm tone adjustments (R,G,B channels in RGB order)
        vector<Mat> ch(3);
        split(img, ch);
        ch[0] = min(ch[0] * 1.1f, 1.0f); // R
        ch[1] = min(ch[1] * 1.05f, 1.0f); // G
        ch[2] = min(ch[2] * 0.95f, 1.0f); // B
        merge(ch, img);

        // Sepia tone
        Mat sepia = apply_sepia_tone(img);

        // Add noise
        Mat noise(sepia.size(), sepia.type());
        std::default_random_engine rng((unsigned)time(nullptr));
        std::normal_distribution<float> nd(0.0f, 0.02f);
        for (int y = 0; y < sepia.rows; ++y) {
            Vec3f* prow = sepia.ptr<Vec3f>(y);
            for (int x = 0; x < sepia.cols; ++x) {
                for (int c = 0; c < 3; ++c) prow[x][c] = std::clamp(prow[x][c] + nd(rng), 0.0f, 1.0f);
            }
        }

        // Vignette
        Mat vign = apply_vignette(sepia, 0.8f);

        Mat out;
        vign.convertTo(out, CV_8U, 255.0);
        // out is RGB; convert to BGR for saving with imwrite if desired
        Mat out_bgr;
        cvtColor(out, out_bgr, COLOR_RGB2BGR);
        return out_bgr;
    }

    Mat apply_sepia_tone(const Mat &img_float) {
        // img_float expected CV_32F RGB in [0,1]
        Mat out = img_float.clone();
        for (int y = 0; y < out.rows; ++y) {
            Vec3f* p = out.ptr<Vec3f>(y);
            for (int x = 0; x < out.cols; ++x) {
                float r = p[x][0], g = p[x][1], b = p[x][2];
                float nr = 0.393f*r + 0.769f*g + 0.189f*b;
                float ng = 0.349f*r + 0.686f*g + 0.168f*b;
                float nb = 0.272f*r + 0.534f*g + 0.131f*b;
                p[x][0] = clamp01(nr);
                p[x][1] = clamp01(ng);
                p[x][2] = clamp01(nb);
            }
        }
        return out;
    }

    Mat apply_vignette(const Mat &img_float, float vignette_strength = 0.8f) {
        int rows = img_float.rows, cols = img_float.cols;
        Mat kernel_x = getGaussianKernel(cols, cols/3.0);
        Mat kernel_y = getGaussianKernel(rows, rows/3.0);
        Mat kernel = kernel_y * kernel_x.t();
        double minv, maxv;
        minMaxLoc(kernel, &minv, &maxv);
        Mat mask;
        kernel.convertTo(mask, CV_32F, 1.0/maxv);
        // mask is single channel; convert to 3 channels and apply
        Mat mask3;
        vector<Mat> channels(3, mask);
        merge(channels, mask3);

        Mat out;
        // mask3 in [0,1], apply vignette: 1 - vignette_strength*(1-mask)
        Mat finalMask = 1.0f - vignette_strength*(1.0f - mask3);
        Mat imgf = img_float.clone();
        multiply(imgf, finalMask, out);
        return out;
    }

    Mat cool_tone_filter(float temperature = 0.7f) {
        // cv2_image_rgb is RGB
        Mat imgf;
        cv2_image_rgb.convertTo(imgf, CV_32F);
        imgf = imgf / 255.0f;
        vector<Mat> ch(3);
        split(imgf, ch);
        // ch[0] is R, ch[2] is B
        ch[0] = ch[0] * (1.0f - temperature * 0.3f);
        ch[2] = ch[2] * (1.0f + temperature * 0.2f);
        merge(ch, imgf);

        Mat img8;
        imgf.convertTo(img8, CV_8U, 255.0);
        Mat hsv;
        cvtColor(img8, hsv, COLOR_RGB2HSV);
        hsv.convertTo(hsv, CV_32F);
        vector<Mat> hch;
        split(hsv, hch);
        hch[1] = hch[1] * 0.9f;
        hch[1] = min(hch[1], 255.0f);
        hch[1].convertTo(hch[1], CV_8U);
        Mat hsv8;
        merge(hch, hsv8);
        Mat out;
        cvtColor(hsv8, out, COLOR_HSV2RGB);
        out.convertTo(out, CV_8U);
        Mat out_bgr;
        cvtColor(out, out_bgr, COLOR_RGB2BGR);
        return out_bgr;
    }

    Mat auto_enhance() {
        Mat img_yuv;
        cvtColor(cv2_image, img_yuv, COLOR_BGR2YUV);
        vector<Mat> ych;
        split(img_yuv, ych);
        equalizeHist(ych[0], ych[0]);
        Mat merged;
        merge(ych, merged);
        Mat img_output;
        cvtColor(merged, img_output, COLOR_YUV2BGR);

        Mat kernel = (Mat_<float>(3,3) << 0,-1,0, -1,5,-1, 0,-1,0);
        filter2D(img_output, img_output, -1, kernel);
        return img_output;
    }

private:
    Mat cv2_image;      // BGR
    Mat cv2_image_rgb;  // RGB
    bool pil_loaded;

    // helper
    static float clamp01(float v) {
        if (v < 0.0f) return 0.0f;
        if (v > 1.0f) return 1.0f;
        return v;
    }
};

int main(int argc, char** argv) {
    string image_path = "q2.jpg";
    if (argc > 1) image_path = argv[1];

    try {
        CustomImageFilter filter(image_path);
        Mat original_bgr = imread(image_path, IMREAD_COLOR);

        cout << "应用色彩增强滤镜..." << endl;
        Mat vivid = filter.enhance_colors_vivid(2.0, 1.4);
        filter.show_comparison(original_bgr, vivid, "Original", "Vivid Colors");

        cout << "应用锐化滤镜..." << endl;
        Mat sharp = filter.sharpen_image(2.5);
        filter.show_comparison(original_bgr, sharp, "Original", "Sharpened");

        cout << "应用油画滤镜..." << endl;
        Mat oil = filter.oil_painting_effect(9, 25);
        filter.show_comparison(original_bgr, oil, "Original", "Oil Painting");

        cout << "应用复古滤镜..." << endl;
        Mat vintage = filter.vintage_filter();
        filter.show_comparison(original_bgr, vintage, "Original", "Vintage");

        cout << "应用冷色调滤镜..." << endl;
        Mat cool = filter.cool_tone_filter(0.6f);
        filter.show_comparison(original_bgr, cool, "Original", "Cool Tone");

        cout << "应用自动增强..." << endl;
        Mat autoe = filter.auto_enhance();
        filter.show_comparison(original_bgr, autoe, "Original", "Auto Enhanced");

    } catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
