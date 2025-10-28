# 数字图像处理作业

这个项目实现了两个基本的图像处理功能：图像缩放和图像旋转。

## 项目结构

```
DIP_VScode/
├── main.cpp              # 主程序入口，包含主函数
├── scaling.h            # 图像缩放功能的头文件
├── scaling.cpp          # 图像缩放功能的实现
├── rotating.h           # 图像旋转功能的头文件
└── rotating.cpp         # 图像旋转功能的实现
```

## 功能说明

### 1. 图像缩放
- 将输入图像按比例缩放到指定尺寸（640x640）
- 保持原始图像的宽高比
- 使用灰色（灰度值128）填充未使用的区域
- 输出文件名：`scaled_lena.jpg`

### 2. 图像旋转
- 将输入图像旋转指定角度（-45度）
- 输出尺寸为600x600
- 输出文件名：`rotated_lena.jpg`

## 使用方法

1. 准备输入图像：
   - 将要处理的图像命名为 `lena.jpg`
   - 放置在程序执行目录下

2. 编译运行程序：
   - 使用 g++ 编译器
   - 需要 OpenCV 库支持

3. 查看结果：
   - 程序执行完成后会生成两个输出文件：
     - `scaled_lena.jpg`：缩放后的图像
     - `rotated_lena.jpg`：旋转后的图像
   - 控制台会显示处理结果和输出图像的分辨率信息

## 依赖要求

- C++ 编译器（g++）
- OpenCV 库
- 支持 C++11 或更高版本

## 编译说明

使用以下命令编译程序：

```bash
g++ main.cpp scaling.cpp rotating.cpp -o main -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
```

## 注意事项

1. 确保输入图像存在且格式正确
2. 程序仅支持灰度图像处理
3. 确保系统已正确安装 OpenCV 库
