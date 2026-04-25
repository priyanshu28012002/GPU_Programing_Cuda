
#include <opencv2/opencv.hpp>
#include <iostream>

// ================= CUDA KERNEL =================

__global__ void invert_kernel(unsigned char *img, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x) * channels;

        for (int c = 0; c < channels; ++c)
        {
            img[idx + c] = 255 - img[idx + c];
        }
    }
}





using namespace cv;
using namespace std;

int main()
{
    string path_ = "/home/octobot/Github/GPU_Programing_Cuda/src/inverseImage/";
    string input = "img.png";
    string output = "output.png";
    Mat img = imread(path_ + input, IMREAD_COLOR);
    if (img.empty())
    {
        cout << "Could not read the image: "  << endl;
        return 1;
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    size_t size = width * height * channels;

    unsigned char *d_img;
    cudaMalloc(&d_img, size);

    cudaMemcpy(d_img, img.data, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    invert_kernel<<<blocks, threads>>>(d_img, width, height, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(img.data, d_img, size, cudaMemcpyDeviceToHost);

    cv::imwrite(path_  + output, img);

    cudaFree(d_img);

    // imshow("Display Window", img);
    // waitKey(0);
    std::cout << "Saved output.png\n";
    return 0;
}