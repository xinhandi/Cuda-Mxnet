/*!
 * Copyright (c) 2017 by Contributors
 * \file knnloss.cu
 * \brief Knnloss
 * \author deepearthgo
 */
#include "./knnloss-inl.h"
#include <math.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace mshadow {
namespace cuda {

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

MSHADOW_XINLINE int LSPowOfMO(const int k) {
  return 1 - ((k&0x01) << 1);
}

template<typename DType>
__global__ void SimilarityMatrixKernel(const Tensor<gpu, 2, DType> x,
                           Tensor<gpu, 2, DType> sbm) {
  const int n = x.size(0);

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<n && j <n){

	 float sum = 0;

	 for (int k=0; k <n; ++k){

		 sum += pow (x[i]-x[j],2);

	 }

	 sbm[i][j] = sqrt(sum);
  }

}

template<typename DType>
__global__ void Sort_Keykernel(const Tensor<gpu, 2, DType> sbm,
                           Tensor<gpu, 2, DType> ssbm) {
  const int n = sbm.size(0);
  CUDA_KERNEL_LOOP(i, n) {

	  for (int j=0; j<k_num; ++j){

          ssbm[i][j]=j;
      }
      thrust::sort_by_key{sbm[i],sbm[i]+n,ssbm[i]};

  }

}

template<typename DType>
__global__ void KnnlossForwardKernel(const Tensor<gpu, 2, DType> x,
                                    const Tensor<gpu, 2, DType> sbm,
                                    const Tensor<gpu, 2, DType> ssbm,
                                    Tensor<gpu, 2, DType> out,
                                    Tensor<gpu, 2, DType> dls,
                                    Tensor<gpu, 1, DType> knc,
                                    Tensor<gpu, 1, DType> kls,
                                    const int k_num,
                                    const int batch_size) {
  const int n = x.size(0);
  const int feature_dim = x.size(1);
  CUDA_KERNEL_LOOP(i, n) {

	  float sum1[feature_dim] ={0};
	  float sum2 = 0;

	  for (int j=0; j<k_num; ++j){


          sum1 += (x[i][ssbm[i][j]]-x[i])/k_num;
          sum2 += dls[i][j];

	  }

      kls += sum2;
      knc[i] = sum1;
      dls[i] =  x[i]-knx[i];
      out += kls;
  }
}

template<typename DType>
inline void KnnlossForward(const Tensor<gpu, 2, DType> &x,
                            const Tensor<gpu, 2, DType> &sbm,
                            const Tensor<gpu, 2, DType> &ssbm,
                            const Tensor<gpu, 2, DType> &out,
                            const Tensor<gpu, 2, DType> &dls,
                            const Tensor<gpu, 1, DType> &knc,
                            const Tensor<gpu, 2, DType> &kls,
                            const int k_num,
                            const int batch_size) {
  const int n = x.size(0);
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid((n + kBaseThreadNum - 1) / kBaseThreadNum);
  dimGrid.x = ((n + kBaseThreadNum - 1) / kBaseThreadNum);
  SimilarityMatrixKernel<<<dimGrid, dimBlock>>>(x, sbm);
  dimGrid.x = ((n + kBaseThreadNum - 1) / kBaseThreadNum);
  Sort_Keykernal<<<dimGrid, dimBlock>>>(sbm, ssbm);
  dimGrid.x = ((n + kBaseThreadNum - 1) / kBaseThreadNum);
  KnnlossForwardKernel<<<dimGrid, dimBlock>>>(x,sbm, ssbm,out,dls,knc,kls,k_num,batch_size);
}

template<typename DType>
__global__ void KnnlossBackwardGradKernel(const Tensor<gpu, 2, DType> dls,
                                        const Tensor<gpu, 2, DType> o_grad,
                                        Tensor<gpu, 2, DType> x_grad,
                                        const float lamna) {
  const int n = dls.size(0);
  CUDA_KERNEL_LOOP(i,n) {
	x_grad[i]= static_cast<float>(lamna) * dls[i];
  }

}

template<typename DType>
inline void KnnlossBackward(const Tensor<gpu, 2, DType> &x,
                             const Tensor<gpu, 2, DType> &out,
                             const Tensor<gpu, 2, DType> &o_grad,
                             const Tensor<gpu, 2, DType> &x_grad,
                             const Tensor<gpu, 2, DType> &dls,
                             const int batch_size,
                             const float lamna) {
  const int x = diff.size(0);
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid((n + kBaseThreadNum - 1) / kBaseThreadNum);
  dimGrid.x = ((n * feature_dim + kBaseThreadNum - 1) / kBaseThreadNum);
  knnlossBackwardGradKernel<<<dimGrid, dimBlock>>>(dls, o_grad, x_grad, lamna);

}

}  // namespace cuda

template<typename DType>
inline void KnnlossForward(const Tensor<gpu, 2, DType> &x,
                            const Tensor<gpu, 2, DType> &sbm,
                            const Tensor<gpu, 2, DType> &ssbm,
                            const Tensor<gpu, 2, DType> &out,
                            const Tensor<cpu, 2, DType> &dls,
                            const Tensor<cpu, 2, DType> &knc,
                            const Tensor<cpu, 2, DType> &kls,
                            const int k_num,
                            const int batch_size) {
  cuda::KnnlossForward(x, sbm, ssbm, out, dls,knc, kls, k_num, batch_size);
}

template<typename DType>
inline void KnnlossBackward(const Tensor<cpu, 2, DType> &x,
                             const Tensor<cpu, 2, DType> &out,
		                     const Tensor<gpu, 2, DType> &o_grad,
                             const Tensor<gpu, 2, DType> &x_grad,
                             const Tensor<gpu, 2, DType> &dls,
                             const int batch_size,
                             const float lamna) {
  cuda::KnnlossBackward(x,out,o_grad,x_grad,dls,batch_size,lamna);
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(KnnlossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new KnnlossOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
} // namespace mxnet
