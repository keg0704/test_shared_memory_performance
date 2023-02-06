#include "ThreadPool.h"
#include "opencv2/imgcodecs.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <turbojpeg.h>
#include <vector>

void processImg(std::string imgPath) {

  // Load the JPEG data
  std::vector<unsigned char> jpeg_data;
  std::ifstream file(imgPath, std::ios_base::binary | std::ios_base::in);
  if (file) {
    file.seekg(0, std::ios::end);
    jpeg_data.resize(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(jpeg_data.data()), jpeg_data.size());
  }

  // Create a handle for the decompressor
  tjhandle handle = tjInitDecompress();

  // Get the size of the uncompressed image
  int width, height, subsamp, colorSpace;
  int ret = tjDecompressHeader3(handle, jpeg_data.data(), jpeg_data.size(),
                                &width, &height, &subsamp, &colorSpace);
  if (ret == -1) {
    std::cerr << "Error decoding JPEG header: " << tjGetErrorStr() << std::endl;
  }

  // Allocate memory for the uncompressed image
  std::vector<unsigned char> image(width * height * tjPixelSize[TJPF_BGR]);

  // Decompress the JPEG data
  ret = tjDecompress2(handle, jpeg_data.data(), jpeg_data.size(), image.data(),
                      width, 0, height, TJPF_BGR, TJFLAG_FASTDCT);
  if (ret == -1) {
    std::cerr << "Error decoding JPEG data: " << tjGetErrorStr() << std::endl;
  }
  // cv::Mat img = cv::imdecode(image, cv::IMREAD_UNCHANGED);
  cv::Mat img;
  img = cv::Mat(height, width, CV_8UC3, image.data());
  // Clean up
  tjDestroy(handle);
}

int main() {
  std::string imgPath = "/concurrency_benchmarking/1659420200000.jpg";

  int increment = 10;
  for (int t = 0; t < 90; t += increment) {
    if (t == 0)
      t = 1;
    ThreadPool pool(t);
    int reps = 10000 * t;

    // Record the start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Submit the task to the thread pool for execution
    std::vector<std::future<void>> results;
    for (int i = 0; i < reps; i++) {
      results.emplace_back(pool.enqueue([i, imgPath] { processImg(imgPath); }));
    }

    // Wait for all tasks to complete
    for (auto &result : results) {
      result.get();
    }

    // Record the end time
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time)
                        .count();

    // Output the result
    std::cout << "decoding QPS: "
              << float(reps) / (float(duration) / 1000 / 1000) << " per second"
              << std::endl;
  }
  return 0;
}
