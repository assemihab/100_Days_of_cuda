#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
__global__ void blur(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int blurSize = 1;
    if (x < width && y < height) {
        int pixVal = 0;
        int pixels = 0;
        for (int blurRow = -blurSize; blurRow < blurSize + 1; ++blurRow) {
            for (int blurCol = -blurSize; blurCol < blurSize + 1; ++blurCol) {
                int curX = x + blurCol;
                int curY = y + blurRow;
                if (curX > -1 && curX < width && curY > -1 && curY < height) {
                    pixVal += input[curY * width + curX];
                    pixels++;
                }
            }
        }
        output[y * width + x] = (unsigned char)(pixVal / pixels);
    }
}

int main()
{
    Mat image=imread("images.jpeg");
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
    cudaMemcpy(d_input, image.data, img_size, cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    blur<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    cudaMemcpy(gray_img.data, d_output, gray_size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    imwrite("blurred_image.jpeg", gray_img);
    return 0;

}