/*!
 * Copyright (c) 2016 by Contributors
 * \file lsoftmax.cc
 * \brief LSoftmax from <Large-Margin Softmax Loss for Convolutional Neural Networks>
 * \author luoyetx
 */
#include "./centerloss-inl.h"

namespace mshadow {

template <typename DType>
inline void centerlossForward(const Tensor<cpu, 2, DType> &x,
                            const Tensor<cpu, 2, DType> &w,
                            const Tensor<cpu, 1, DType> &label,
                            const Tensor<cpu, 2, DType> &out,
                            const Tensor<cpu, 2, DType> &diff,
                            const Tensor<cpu, 2, DType> &center,
                            const int batch_size) {
  LOG(FATAL) << "Not Implemented.";
}

template <typename DType>
inline void centerlossBackward(const Tensor<cpu, 2, DType> &diff,
                             const Tensor<cpu, 2, DType> &center,
		                     const Tensor<cpu, 1, DType> &label,
                             const Tensor<cpu, 2, DType> &o_grad,
                             const Tensor<cpu, 2, DType> &x_grad,
                             const float scale,
                             const int batch_size,
                             const int class_num,
                             const float alpha) {
  LOG(FATAL) << "Not Implemented.";
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(centerlossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new centerlossOp<cpu, DType>(param);
  })
  return op;
}

Operator *centerlossProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                         std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(centerlossParam);

MXNET_REGISTER_OP_PROPERTY(centerloss, centerlossProp)
.describe("Centerloss from <Centerloss Loss for Convolutional Neural Networks>")
.add_argument("data", "Symbol", "data")
.add_argument("weight", "Symbol", "weight")
.add_argument("label", "Symbol", "label")
.add_arguments(centerlossParam::__FIELDS__());

}  // namespace op
} // namespace mxne
