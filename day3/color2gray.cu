#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

__global__ void rgb_to_gray(unsigned char *d_input, unsigned char *d_output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;  
        int gray_idx = y * width + x;          

        
        d_output[gray_idx] = 0.299f * d_input[idx] + 0.587f * d_input[idx + 1] + 0.114f * d_input[idx + 2];
    }
}


int main()
{
    cv::Mat image = cv::imread("images.jpeg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    Mat gray_img(height, width, CV_8UC1);
    unsigned char *d_input, *d_output;
    size_t img_size = width * height * channels * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);
    cudaMalloc((void **)&d_input, img_size);
    cudaMalloc((void **)&d_output, gray_size);
    // print the dimensions o img.dat
    

    cudaMemcpy(d_input, image.data, img_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);  
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    rgb_to_gray<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    cudaMemcpy(gray_img.data, d_output, gray_size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    imshow("Original Image", image);
    imshow("Grayscale Image", gray_img);
    waitKey(0);

    
    cv::imshow("Loaded Image", image);
    cv::waitKey(0);
}
