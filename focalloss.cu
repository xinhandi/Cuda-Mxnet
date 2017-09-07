/*!
 * Copyright (c) 2016 by Contributors
 * \file lsoftmax.cu
 * \brief LSoftmax from <Large-Margin Softmax Loss for Convolutional Neural Networks>
 * \author luoyetx
 */
#include "./lsoftmax-inl.h"
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

namespace mshadow {
namespace cuda {

namespace {
// workspace variables
enum LSoftmaxTempSpaceType {kCost, kCosmt, kK, kSin2t, kFo, kCostM};
}

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

MSHADOW_XINLINE int LSPowOfMO(const int k) {
  return 1 - ((k&0x01) << 1);
}
//Elementwise[Etw]---||EtwPower||EtwLog||EtwExp||
template<typename DType>
__global__ void EtwPowerM(const Tensor<gpu, 2, DType> matrix,
		                  const float gamma,
                          Tensor<gpu, 2, DType> Pmatrix) {
  const int dim1 = matrix.size(0);
  const int dim2 = matrix.size(1);

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

	  Pmatrix[i][j] = pow (matrix[i][j],gamma);

  }
}

template<typename DType>
__global__ void EtwPowerV(const Tensor<gpu, 2, DType> vector,
		                  const float gamma,
                          Tensor<gpu, 2, DType> Pvector) {
  const int dim1 = vector.size;

  CUDA_KERNEL_LOOP(i,n){

	  Pvector[i] = pow (vector[i],gamma);
  }

}

template<typename DType>
__global__ void EtwLogM(const Tensor<gpu, 2, DType> matrix,
                        Tensor<gpu, 2, DType> Lmatrix) {
  const int dim1 = matrix.size(0);
  const int dim2 = matrix.size(1);

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

	  Lmatrix[i][j] = log (matrix[i][j] + 1e-40);

  }
}

template<typename DType>
__global__ void EtwLogV(const Tensor<gpu, 1, DType> vector,
                        Tensor<gpu, 1, DType> Lvector) {
  const int n = vector.size;
  CUDA_KERNEL_LOOP(i, n) {

	  Lvector[i] = log (vector[i] + 1e-40);

  }
}

template<typename DType>
__global__ void EtwExpM(const Tensor<gpu, 2, DType> matrix,
                        Tensor<gpu, 2, DType> Ematrix) {
  const int dim1 = matrix.size(0);
  const int dim2 = matrix.size(1);

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

	  Ematrix[i][j] = exp (matrix[i][j]);

  }
}

template<typename DType>
__global__ void EtwExpV(const Tensor<gpu, 2, DType> vector,
                        Tensor<gpu, 1, DType> Evector) {
  const int dim = vector.size;

  CUDA_KERNEL_LOOP(i, dim) {

      Evector[i] = exp (vector[i]);

  }

}

template<typename DType>
__global__ void EtwExpMax(const Tensor<gpu, 2, DType> matrix,
                          Tensor<gpu, 1, DType> Mmatrix) {
  const int dim1 = matrix.size(0);
  const int dim2 = matrix.size(1);

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

  	  Mmatrix[i][j] = thrust::reduce(matrix[i],matrix[i]+dim1,-1,thrust::maximum<float>());

    }

}

template<typename DType>
__global__ void EtwAdd(const Tensor<gpu, 2, DType> matrix,
                       Tensor<gpu, 1, DType> Mmatrix) {
  const int dim1 = matrix.size(0);
  const int dim2 = matrix.size(1);

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

  	  Mmatrix[i][j] = thrust::reduce(matrix[i],matrix[i]+dim1,-1,thrust::maximum<float>());

    }

}

template<typename DType>
__global__ void Etwsub(const Tensor<gpu, 2, DType> matrix,
                       Tensor<gpu, 1, DType> Mmatrix) {
  const int dim1 = matrix.size(0);
  const int dim2 = matrix.size(1);

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

  	  Mmatrix[i][j] = thrust::reduce(matrix[i],matrix[i]+dim1,-1,thrust::maximum<float>());

    }

}

template<typename DType>
__global__ void Etwdiv(const Tensor<gpu, 2, DType> matrix,
                       Tensor<gpu, 1, DType> Mmatrix) {
  const int dim1 = matrix.size(0);
  const int dim2 = matrix.size(1);

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

  	  Mmatrix[i][j] = thrust::reduce(matrix[i],matrix[i]+dim1,-1,thrust::maximum<float>());

    }

}


template<typename DType>
__global__ void FocalLossForwardKernel(const Tensor<gpu, 2, DType> cls_score,
                                      const Tensor<gpu, 1, DType> label,
                                      const Tensor<gpu, 2, DType> pro_,
                                      const Tensor<gpu, 1, DType> _pt,
                                      Tensor<gpu, 2, DType> out,
                                      const float gamma,
                                      const float alpha) {
  const int n = x.size(0);
  const int n_class = x.size(1);
  CUDA_KERNEL_LOOP(i, n) {
    //const int yi = static_cast<int>(label[i]);
    const DType fo_i_yi = out[i][yi];
    const DType cos_t = fo_i_yi / (x_norm[i] * w_norm[yi]);
    const int k = LSFindK(k_table.dptr_, k_table.size(0), cos_t);
    const DType cos_mt = LSCalcCosmt(c_table.dptr_, c_table.size(0), cos_t, margin);
    const DType f_i_yi = (LSPowOfMO(k) * cos_mt - 2*k) * (w_norm[yi] * x_norm[i]);
    out[i][yi] = (f_i_yi + beta * fo_i_yi) / (1 + beta);
  }
}


template<typename DType>
inline void LSoftmaxForward(const Tensor<gpu, 2, DType> &x,
                            const Tensor<gpu, 2, DType> &w,
                            const Tensor<gpu, 1, DType> &label,
                            const Tensor<gpu, 2, DType> &out,
                            const Tensor<gpu, 1, DType> &x_norm,
                            const Tensor<gpu, 1, DType> &w_norm,
                            const Tensor<gpu, 1, DType> &k_table,
                            const Tensor<gpu, 1, DType> &c_table,
                            const int margin,
                            const DType beta) {
  const int n = x.size(0);
  const int m = w.size(0);
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid((n + kBaseThreadNum - 1) / kBaseThreadNum);
  LSCalcNorm<<<dimGrid, dimBlock>>>(x, x_norm);
  dimGrid.x = ((m + kBaseThreadNum - 1) / kBaseThreadNum);
  LSCalcNorm<<<dimGrid, dimBlock>>>(w, w_norm);
  dimGrid.x = ((n + kBaseThreadNum - 1) / kBaseThreadNum);
  LSoftmaxForwardKernel<<<dimGrid, dimBlock>>>(x, w, label, x_norm, w_norm, out, k_table, c_table, margin, beta);
}

template<typename DType>
__global__ void LSoftmaxBackwardRequired(const Tensor<gpu, 2, DType> x,
                                         const Tensor<gpu, 2, DType> w,
                                         const Tensor<gpu, 1, DType> label,
                                         const Tensor<gpu, 1, DType> x_norm,
                                         const Tensor<gpu, 1, DType> w_norm,
                                         Tensor<gpu, 2, DType> workspace,
                                         const Tensor<gpu, 1, DType> k_table,
                                         const Tensor<gpu, 1, DType> c_table,
                                         const int margin) {
  const int n = x.size(0);
  const int feature_dim = x.size(1);
  CUDA_KERNEL_LOOP(i, n) {
    const int yi = static_cast<int>(label[i]);
    // fo_i_yi = dot(w_yi, x_i)
    DType fo_i_yi = 0;
    for (int p = 0; p < feature_dim; ++p) {
      fo_i_yi += w[yi][p] * x[i][p];
    }
    const DType cos_t = fo_i_yi / (x_norm[i] * w_norm[yi]);
    const int k = LSFindK(k_table.dptr_, k_table.size(0), cos_t);
    const DType cos_mt = LSCalcCosmt(c_table.dptr_, c_table.size(0), cos_t, margin);
    const DType sin2_t = 1 - cos_t * cos_t;
    workspace[kCost][i] = cos_t;
    workspace[kCosmt][i] = cos_mt;
    workspace[kK][i] = static_cast<DType>(k);
    workspace[kSin2t][i] = sin2_t;
    workspace[kFo][i] = fo_i_yi;
    workspace[kCostM][i] = pow(cos_t, margin - 1);
  }
}

template<typename DType>
__global__ void LSoftmaxBackwardXKernel(const Tensor<gpu, 2, DType> x,
                                        const Tensor<gpu, 2, DType> w,
                                        const Tensor<gpu, 1, DType> label,
                                        const Tensor<gpu, 1, DType> x_norm,
                                        const Tensor<gpu, 1, DType> w_norm,
                                        const Tensor<gpu, 2, DType> o_grad,
                                        Tensor<gpu, 2, DType> x_grad,
                                        const Tensor<gpu, 2, DType> workspace,
                                        const Tensor<gpu, 1, DType> c_table,
                                        const int margin,
                                        const DType beta) {
  const int nthreads = x.size(0) * x.size(1);
  const int feature_dim = x.size(1);
  CUDA_KERNEL_LOOP(idx, nthreads) {
    const int i = idx / feature_dim;
    const int l = idx % feature_dim;
    const int yi = static_cast<int>(label[i]);
    const DType cos_t = workspace[kCost][i];
    const DType cos_mt = workspace[kCosmt][i];
    const int k = static_cast<int>(workspace[kK][i]);
    const DType sin2_t = workspace[kSin2t][i];
    const DType fo_i_yi = workspace[kFo][i];
    const DType w_norm_yi = w_norm[yi];
    const DType x_norm_i = x_norm[i];

    const DType dcos_dx = w[yi][l] / (w_norm_yi * x_norm_i) - \
                          fo_i_yi * x[i][l] / (w_norm_yi * x_norm_i * x_norm_i * x_norm_i);
    const DType dsin2_dx = -2 * cos_t * dcos_dx;
    DType cos_t_p = workspace[kCostM][i];
    DType sin2_t_p = 1;
    DType dcosm_dx = margin * cos_t_p * dcos_dx;  // p = 0
    for (int p = 1; p <= margin / 2; ++p) {
      cos_t_p /= cos_t * cos_t;
      dcosm_dx += LSPowOfMO(p) * c_table[2*p] * (p * cos_t * dsin2_dx + \
                    (margin - 2*p) * sin2_t * dcos_dx) * cos_t_p * sin2_t_p;
      sin2_t_p *= sin2_t;
    }
    const DType df_dx = (LSPowOfMO(k) * cos_mt - 2*k) * w_norm_yi / x_norm_i * x[i][l] + \
                         LSPowOfMO(k) * w_norm_yi * x_norm_i * dcosm_dx;
    const DType alpha = 1 / (1 + beta);
    x_grad[i][l] += alpha * o_grad[i][yi] * (df_dx - w[yi][l]);
  }
}

template<typename DType>
__global__ void LSoftmaxBackwardWKernel(const Tensor<gpu, 2, DType> x,
                                        const Tensor<gpu, 2, DType> w,
                                        const Tensor<gpu, 1, DType> label,
                                        const Tensor<gpu, 1, DType> x_norm,
                                        const Tensor<gpu, 1, DType> w_norm,
                                        const Tensor<gpu, 2, DType> o_grad,
                                        Tensor<gpu, 2, DType> w_grad,
                                        const Tensor<gpu, 2, DType> workspace,
                                        const Tensor<gpu, 1, DType> c_table,
                                        const int margin,
                                        const DType beta) {
  const int nthreads = w.size(0) * w.size(1);
  const int n = x.size(0);
  const int feature_dim = w.size(1);
  CUDA_KERNEL_LOOP(idx, nthreads) {
    const int j = idx / feature_dim;
    const int l = idx % feature_dim;
    DType dw = 0;
    for (int i = 0; i < n; ++i) {
      const int yi = static_cast<int>(label[i]);
      if (yi == j) {
        const DType cos_t = workspace[kCost][i];
        const DType cos_mt = workspace[kCosmt][i];
        const int k = static_cast<int>(workspace[kK][i]);
        const DType sin2_t = workspace[kSin2t][i];
        const DType fo_i_yi = workspace[kFo][i];
        const DType x_norm_i = x_norm[i];
        const DType w_norm_yi = w_norm[yi];

        const DType dcos_dw = x[i][l] / (w_norm_yi * x_norm_i) - \
                              fo_i_yi * w[yi][l] / (x_norm_i * w_norm_yi * w_norm_yi * w_norm_yi);
        const DType dsin2_dw = -2 * cos_t * dcos_dw;
        DType cos_t_p = workspace[kCostM][i];
        DType sin2_t_p = 1;
        DType dcosm_dw = margin * cos_t_p * dcos_dw;  // p = 0
        for (int p = 1; p <= margin / 2; ++p) {
          cos_t_p /= cos_t * cos_t;
          dcosm_dw += LSPowOfMO(p) * c_table[2*p] * (p * cos_t * dsin2_dw + \
                        (margin - 2*p) * sin2_t * dcos_dw) * cos_t_p * sin2_t_p;
          sin2_t_p *= sin2_t;
        }
        const DType df_dw_j = (LSPowOfMO(k) * cos_mt - 2*k) * x_norm_i / w_norm_yi * w[yi][l] + \
                               LSPowOfMO(k) * w_norm_yi * x_norm_i * dcosm_dw;
        dw += o_grad[i][yi] * (df_dw_j - x[i][l]);
      }
    }
    const DType alpha = 1 / (1 + beta);
    w_grad[j][l] += alpha * dw;
  }
}

template<typename DType>
inline void LSoftmaxBackward(const Tensor<gpu, 2, DType> &x,
                             const Tensor<gpu, 2, DType> &w,
                             const Tensor<gpu, 1, DType> &label,
                             const Tensor<gpu, 1, DType> &x_norm,
                             const Tensor<gpu, 1, DType> &w_norm,
                             const Tensor<gpu, 2, DType> &o_grad,
                             const Tensor<gpu, 2, DType> &x_grad,
                             const Tensor<gpu, 2, DType> &w_grad,
                             const Tensor<gpu, 2, DType> &workspace,
                             const Tensor<gpu, 1, DType> &k_table,
                             const Tensor<gpu, 1, DType> &c_table,
                             const int margin,
                             const DType beta) {
  const int n = x.size(0);
  const int feature_dim = x.size(1);
  const int m = w.size(0);
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid((n + kBaseThreadNum - 1) / kBaseThreadNum);
  LSoftmaxBackwardRequired<<<dimGrid, dimBlock>>>(x, w, label, x_norm, w_norm, workspace,
                                                  k_table, c_table, margin);
  dimGrid.x = ((n * feature_dim + kBaseThreadNum - 1) / kBaseThreadNum);
  LSoftmaxBackwardXKernel<<<dimGrid, dimBlock>>>(x, w, label, x_norm, w_norm, o_grad, x_grad, workspace,
                                                 c_table, margin, beta);
  dimGrid.x = ((m * feature_dim + kBaseThreadNum - 1) / kBaseThreadNum);
  LSoftmaxBackwardWKernel<<<dimGrid, dimBlock>>>(x, w, label, x_norm, w_norm, o_grad, w_grad, workspace,
                                                 c_table, margin, beta);
}

}  // namespace cuda

template<typename DType>
inline void LSoftmaxForward(const Tensor<gpu, 2, DType> &x,
                            const Tensor<gpu, 2, DType> &w,
                            const Tensor<gpu, 1, DType> &label,
                            const Tensor<gpu, 2, DType> &out,
                            const Tensor<gpu, 1, DType> &x_norm,
                            const Tensor<gpu, 1, DType> &w_norm,
                            const Tensor<gpu, 1, DType> &k_table,
                            const Tensor<gpu, 1, DType> &c_table,
                            const int margin,
                            const DType beta) {
  cuda::LSoftmaxForward(x, w, label, out, x_norm, w_norm,
                        k_table, c_table, margin, beta);
}

template<typename DType>
inline void LSoftmaxBackward(const Tensor<gpu, 2, DType> &x,
                             const Tensor<gpu, 2, DType> &w,
                             const Tensor<gpu, 1, DType> &label,
                             const Tensor<gpu, 1, DType> &x_norm,
                             const Tensor<gpu, 1, DType> &w_norm,
                             const Tensor<gpu, 2, DType> &o_grad,
                             const Tensor<gpu, 2, DType> &x_grad,
                             const Tensor<gpu, 2, DType> &w_grad,
                             const Tensor<gpu, 2, DType> &workspace,
                             const Tensor<gpu, 1, DType> &k_table,
                             const Tensor<gpu, 1, DType> &c_table,
                             const int margin,
                             const DType beta) {
  cuda::LSoftmaxBackward(x, w, label, x_norm, w_norm, o_grad, x_grad, w_grad, workspace,
                         k_table, c_table, margin, beta);
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(LSoftmaxParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new LSoftmaxOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
} // namespace mxnet
