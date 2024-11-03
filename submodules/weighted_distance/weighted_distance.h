#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> weightedDistanceCUDA(
    const torch::Tensor& coefs,
    const torch::Tensor& codebook);

torch::Tensor kmeansPlusPlusDistanceCUDA(
    const torch::Tensor &data_points,
    const torch::Tensor &centroids);