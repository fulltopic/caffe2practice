#ifndef UTIL_TENSOR_H
#define UTIL_TENSOR_H

#include <caffe2/core/tensor.h>
#include <caffe2/core/blob.h>
#include <caffe2/core/workspace.h>

namespace caffe2 {

class TensorUtil {
 public:
  TensorUtil(Tensor& tensor) : tensor_(tensor) {}

  void WriteImages(const std::string& name, float mean = 128,
                   bool lossy = false, int index = 0);
  void WriteImage(const std::string& name, int index, float mean = 128,
                  bool lossy = false);
  TensorCPU ScaleImageTensor(int width, int height);
  void ReadImages(const std::vector<std::string>& filenames, int width,
                  int height, std::vector<int>& indices, float mean = 128,
                  TensorProto::DataType type = TensorProto_DataType_FLOAT);
  void ReadImage(const std::string& filename, int width, int height);
  void Print(const std::string& name = "", int max = 100);

  static Blob* ZeroFloats(Workspace& ws, const std::vector<int>& dim, const std::string& name);



 protected:
  Tensor& tensor_;
};

}  // namespace caffe2

#endif  // UTIL_TENSOR_H
