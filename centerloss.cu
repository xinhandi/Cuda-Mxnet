/*!
 * Copyright (c) 2017 by Contributors
 * \file centerloss.cu
 * \brief centerloss 
 * \author deepearthgo
 */
#include "./centerloss-inl.h"

namespace mshadow {
namespace cuda {

define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

MSHADOW_XINLINE int LSPowOfMO(const int k) {
  return 1 - ((k&0x01) << 1);
}

template<typename DType>
__global__ void centerlossForwardKernel(const Tensor<gpu, 2, DType> x,
                                        const Tensor<gpu, 2, DType> w,
                                        const Tensor<gpu, 1, DType> label,
                                        Tensor<gpu, 2, DType> out,
                                        const Tensor<cpu, 2, DType> &diff,
                                        const Tensor<cpu, 2, DType> &center,
                                        const int batch_size) {
  const int n = x.size(0);
  const int feature_dim = x.size(1);
  const int m = w.size(0);
  CUDA_KERNEL_LOOP(i, n) {
    const int yi = static_cast<int>(label[i]);
    diff[i] = x[i] - center[yi]
    out += sqrt(diff[i])/(0.5*batch_size)
  }
}

template<typename DType>
inline void centerlossForward(const Tensor<gpu, 2, DType> &x,
                            const Tensor<gpu, 2, DType> &w,
                            const Tensor<gpu, 1, DType> &label,
                            const Tensor<gpu, 2, DType> &out,
                            const Tensor<cpu, 2, DType> &diff,
                            const Tensor<cpu, 2, DType> &center,
                            const int batch_size) {
  const int n = x.size(0);
  const int m = w.size(0);
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid((n + kBaseThreadNum - 1) / kBaseThreadNum);
  dimGrid.x = ((n + kBaseThreadNum - 1) / kBaseThreadNum);
  LSoftmaxForwardKernel<<<dimGrid, dimBlock>>>(x, w, label, out, diff, center);
}

template<typename DType>
__global__ void centerlossBackwardCenterlabel(const Tensor<gpu, 2, DType> diff,
                                         const Tensor<gpu, 2, DType> center,
                                         const int class_num,
                                         const float alpha) {
  const int n = diff.size(0);
  const int feature_dim = diff.size(1);

  CUDA_KERNEL_LOOP(i,class_num){

	  const float sum_[feature_dim] ={0};
	  const int ind1 = 0;

	  // update sum_
	  for(int k=0;k<n;++k){
	    if (<int>(center[k])==i)
	    {
	       sum_ = sum_ + diff[k];
	       ind1 = ind1 + 1;
	    }

	  // update center
	  delta_c = sum_/(1+ind_1);
	  }
	  center[i] += alpha * delta_c;
  }

}

template<typename DType>
__global__ void centerlossBackwardGradKernel(const Tensor<gpu, 2, DType> diff,
                                        const Tensor<gpu, 2, DType> center,
                                        const Tensor<gpu, 1, DType> label,
                                        const Tensor<gpu, 2, DType> o_grad,
                                        Tensor<gpu, 2, DType> x_grad,
                                        const float scale,
                                        const int batch_size) {
  const int n = diff.size(0);
  const float feature_dim = diff.size(1);
  CUDA_KERNEL_LOOP(i,n) {
	x_grad[i]= static_cast<float>(scale/batch_size) * diff[i];
  }

}

template<typename DType>
inline void centerlossBackward(const Tensor<cpu, 2, DType> &diff,
                             const Tensor<cpu, 2, DType> &center,
		                     const Tensor<gpu, 1, DType> &label,
                             const Tensor<gpu, 2, DType> &o_grad,
                             const Tensor<gpu, 2, DType> &x_grad,
                             const float scale,
                             const int batch_size,
                             const int class_num,
                             const float alpha) {
  const int n = diff.size(0);
  const int feature_dim = diff.size(1);
  //const int m = w.size(0);
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid((n + kBaseThreadNum - 1) / kBaseThreadNum);
  dimGrid.x = ((n * feature_dim + kBaseThreadNum - 1) / kBaseThreadNum);
  centerlossBackwardCenterlabel<<<dimGrid, dimBlock>>>(diff, center, class_num, alpha);
  dimGrid.x = ((n * feature_dim + kBaseThreadNum - 1) / kBaseThreadNum);
  centerlossBackwardGradKernel<<<dimGrid, dimBlock>>>(diff, center, label, o_grad, x_grad, scale,batch_size);
}

}  // namespace cuda

template<typename DType>
inline void centerlossForward(const Tensor<gpu, 2, DType> &x,
                            const Tensor<gpu, 2, DType> &w,
                            const Tensor<gpu, 1, DType> &label,
                            const Tensor<gpu, 2, DType> &out,
                            const Tensor<cpu, 2, DType> &diff,
                            const Tensor<cpu, 2, DType> &center,
                            const int batch_size) {
  cuda::centerlossForward(x, w, label, out, diff,center,batch_size);
}

template<typename DType>
inline void centerlossBackward(const Tensor<cpu, 2, DType> &diff,
                             const Tensor<cpu, 2, DType> &center,
		                     const Tensor<gpu, 1, DType> &label,
                             const Tensor<gpu, 2, DType> &o_grad,
                             const Tensor<gpu, 2, DType> &x_grad,
                             const float scale,
                             const int batch_size,
                             const int class_num,
                             const float alpha) {
  cuda::LSoftmaxBackward(diff,center,label,o_grad,x_grad,scale,batch_size,class_num,alpha);
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(centerlossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new LSoftmaxOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
} // namespace mxnet
