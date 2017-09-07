/*!
 * Copyright (c) 2017 by Contributors
 * \file focalloss.cu
 * \brief focalloss
 * \author deepearthgo
 */
#include "./lsoftmax-inl.h"
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

namespace mshadow {
namespace cuda {

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

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

  	  Mmatrix[i][j] = thrust::reduce(matrix[i],matrix[i]+dim1,-1,thrust::maximum<DType>());

    }

}

template<typename DType>
__global__ void EtwSumM(const Tensor<gpu, 2, DType> matrix,
                          Tensor<gpu, 1, DType> Mmatrix) {
  const int dim1 = matrix.size(0);
  const int dim2 = matrix.size(1);

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

  	  Mmatrix[i][j] = thrust::reduce(matrix[i],matrix[i]+dim1,int(0),thrust::plus<DType>());

    }

}
	
template<typename DType>
__global__ void EtwAdd(const Tensor<gpu, 2, DType> matrix1,
                       const Tensor<gpu, 2, DType> matrix2,
		       Tensor <gpu, 2, DType> matrixadd) {
  const int dim1 = matrix1.size(0);
  const int dim2 = matrix1.size(1);

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

  	  matrixadd[i][j] = matrix1[i][j] + matrix2[i][j];

    }

}

template<typename DType>
__global__ void EtwSub(const Tensor<gpu, 2, DType> matrix1,
                       const Tensor<gpu, 2, DType> matrix2,
		       Tensor <gpu, 2, DType> matrixsub) {
  const int dim1 = matrix1.size(0);
  const int dim2 = matrix1.size(1);

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

  	  matrixsub[i][j] = matrix1[i][j] - matrix2[i][j];

    }

}
	
template<typename DType>
__global__ void EtwDiv(const Tensor<gpu, 2, DType> matrix1,
                       const Tensor<gpu, 2, DType> matrix2,
		       Tensor <gpu, 2, DType> matrixdiv) {
  const int dim1 = matrix1.size(0);
  const int dim2 = matrix1.size(1);

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

  	  matrixdiv[i][j] = matrix1[i][j]/matrix2[i][j];

    }

}

template<typename DType>
__global__ void EtwVal(const int dim1,
		       const int dim2,
                       const  float value,
		       Tensor <gpu, 2, DType> matrixVal) {

  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

  	  matrixVal[i][j] = value;

    }

}
	
template<typename DType>
__global__ void EtwValM(const <gpu, 2, DType> matrix
		       Tensor <gpu, 2, DType> matrixVal) {
  
  const int dim1 = matrix.size(0);
  const int dim2 = matrix.size(1);
  
  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<dim1 && j<dim2){

  	  matrixVal[i][j] = matrix[i][j];

    }

}
	
template<typename DType>
__global__ void FocalLossForward(const Tensor<gpu, 2, DType> &cls_score,
				const Tensor<gpu, 2, DType> &cls_score_max,
				const Tensor<gpu, 2, DType> &cls_score_sub,
                                const Tensor<gpu, 1, DType> &label,
                                const Tensor<gpu, 2, DType> &pro_,
				const Tensor<gpu, 2, DType> &pro_sum,
				const Tensor<gpu, 2, DType> &pro_div,
                                const Tensor<gpu, 1, DType> &_pt,
                                Tensor<gpu, 2, DType> out,
                                const float gamma,
                                const float alpha) {
  const int n = cls_score.size(0);
  const int n_class = cls_score.size(1);
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid((n + kBaseThreadNum - 1) / kBaseThreadNum);
	
  //1.A	
  dimGrid.x = ((n + kBaseThreadNum - 1) / kBaseThreadNum);	
  EtwExpMax<<<dimGrid, dimBlock>>>(cls_score,cls_score_max);	
  //1.B	
  dimGrid.x = ((n + kBaseThreadNum - 1) / kBaseThreadNum);
  EtwSub<<<dimGrid, dimBlock>>>(cls_socre, cls_score_max, cls_score_sub);
  //1.C	  	
  dimGrid.x = ((n + kBaseThreadNum - 1) / kBaseThreadNum);
  EtwExpM<<<dimGrid, dimBlock>>>(cls_score_sub, pro_);
	
  //2.A
  dimGrid.x = ((n + kBaseThreadNum - 1) / kBaseThreadNum);
  EtwSumM<<<dimGrid, dimBlock>>>(pro_, pro_sum);
  //2.B
  dimGrid.x = ((n + kBaseThreadNum - 1) / kBaseThreadNum);
  EtwDiv<<<dimGrid, dimBlock>>>(pro_,pro_sum,pro_div);
  //2.C
  dimGrid.x = ((n + kBaseThreadNum - 1) / kBaseThreadNum);	
  EtwValM<<<dimGrid, dimBlock>>>(pro_div,pro_);
	
  //3.A
  CUDA_KERNEL_LOOP(i,n){
     
      _pt[i] = pro_[i][static_cast<int>(label[i])];
	      
  }
	
 //4.A
 dimGrid.x = ((n + kBaseThreadNum - 1) / kBaseThreadNum);	
 EtwValM<<<dimGrid, dimBlock>>>(pro_,out);

}

template<typename DType>
__global__ void FocalLossBackwardKernel1(const Tensor<gpu, 1, DType> label,
                                    const Tensor<gpu, 1, DType> _pt,
                                    const Tensor<gpu, 2, DType> pro_,
                                    const Tensor<gpu, 2, DType> o_grad,
                                    const Tensor<gpu, 2, DType> x_grad,
                                    const float gamma,
                                    const float alpha) {
  const int n = label.size;
  const int n_class = x.size(1);
  const float esl = 1e-40;
  
  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
  //i!=j
  if (i<n && j<n_class){
	 
	_pt[i] = _pt[i] + es1;  
	x_grad[i][j] = alpha * pow ( 1 - _pt[i], gamma - 1.0 ) * (gamma * (-1 * _pt[i] * pro_[i][j]) * log10f (_pt[i])) + pro_[i][j] * (1 - _pt[i]) * 1.0;
	  
  }	  
}

template<typename DType>
__global__ void FocalLossBackwardKernel2(const Tensor<gpu, 1, DType> label,
                                    const Tensor<gpu, 1, DType> _pt,
                                    const Tensor<gpu, 2, DType> pro_,
                                    const Tensor<gpu, 2, DType> o_grad,
                                    const Tensor<gpu, 2, DType> x_grad,
                                    const float gamma,
                                    const float alpha) {
  const int n = label.size;
  const int n_class = x.size(1);
  const float esl = 1e-40;
  
  unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
  //i==j
  CUDA_KERNEL_LOOP(i, n) {
      
      //no need any more _pt[i] = _pt[i] + es1;	  
      x_grad[i][static_cast<int>(label[i])] = alpha * pow (1 - _pt[i], gamma) * (gamma * _pt[i] * log10f (_pt[i]) + _pt[i] -1 )*(1.0)/static_cast<float>(n);

  }
	
}
	
template<typename DType>
inline void FocalLossBackward(const Tensor<gpu, 1, DType> &label,
                              const Tensor<gpu, 1, DType> &_pt,
                              const Tensor<gpu, 2, DType> &pro_,
                              const Tensor<gpu, 2, DType> &o_grad,
                              const Tensor<gpu, 2, DType> &x_grad,
                              const float gamma,
                              const float alpha) {
  const int n = label.size;
  const int n_class = x.size(1);
  //const int m = w.size(0);
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid((n + kBaseThreadNum - 1) / kBaseThreadNum);
  dimGrid.x = ((n * feature_dim + kBaseThreadNum - 1) / kBaseThreadNum);
  FocalLossBackwardKernel1<<<dimGrid, dimBlock>>>(label, _pt, pro_, o_grad, x_grad, gamma, alpha);
  dimGrid.x = ((n * feature_dim + kBaseThreadNum - 1) / kBaseThreadNum);
  FocalLossBackwardKernel2<<<dimGrid, dimBlock>>>(label, _pt, pro_, o_grad, x_grad, gamma, alpha);

}  // namespace cuda
	
	
template<typename DType>
inline void FocalLossForward(const Tensor<gpu, 2, DType> &cls_score,
		            const Tensor<gpu, 2, DType> &cls_score_max,
			    const Tensor<gpu, 2, DType> &cls_score_sub,
                            const Tensor<gpu, 1, DType> &label,
                            const Tensor<gpu, 2, DType> &pro_,
		            const Tensor<gpu, 2, DType> &pro_sum,
			    const Tensor<gpu, 2, DType> &pro_div,
                            const Tensor<gpu, 1, DType> &_pt,
                            Tensor<gpu, 2, DType> out,
                            const float gamma,
                            const float alpha) {
  cuda::FocalLossForward(cls_score, cls_score_max, cls_score_sub, label, pro_, pro_sum,
                        pro_div, _pt, out, gamma, alpha);
}

template<typename DType>
inline void FocalLossBackward(const Tensor<gpu, 1, DType> &label,
                              const Tensor<gpu, 1, DType> &_pt,
                              const Tensor<gpu, 2, DType> &pro_,
                              const Tensor<gpu, 2, DType> &o_grad,
                              const Tensor<gpu, 2, DType> &x_grad,
                              const float gamma,
                              const float alpha) {
  cuda::FocalLossBackward(label, _pt, pro_, o_grad, x_grad);
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(FocalLossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new LSoftmaxOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
} // namespace mxnet
