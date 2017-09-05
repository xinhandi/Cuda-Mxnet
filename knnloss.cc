/*!
 * Copyright (c) 2017 by Contributors
 * \file knnloss.cc
 * \brief Knnloss
 * \author deepearthgo
 */
#include "./knnloss-inl.h"

namespace mshadow {

template <typename DType>
inline void KnnlossForward(const Tensor<cpu, 2, DType> &x,
		                  const Tensor<cpu, 2, DType> &sbm,
		                  const Tensor<cpu, 1, DType> &ssbm,
                          const Tensor<cpu, 2, DType> &out,
                          const Tensor<cpu, 2, DType> &dls,
                          const Tensor<cpu, 1, DType> &knc,
                          const Tensor<cpu, 1, DType> &kls,
                          const int k_num,
                          const int batch_size) {
  LOG(FATAL) << "Not Implemented.";
}

template <typename DType>
inline void KnnlossBackward( const Tensor<cpu, 2, DType> &x,
                             const Tensor<cpu, 2, DType> &out,
                             const Tensor<cpu, 2, DType> &o_grad,
                             const Tensor<cpu, 2, DType> &x_grad,
                             const Tensor<cpu, 2, DType> &dls,
                             const int batch_size,
                             const float lamna) {
  LOG(FATAL) << "Not Implemented.";
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(KnnlossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new knnlossOp<cpu, DType>(param);
  })
  return op;
}

Operator *KnnlossProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                         std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(KnnlossParam);

MXNET_REGISTER_OP_PROPERTY(Knnloss, KnnlossProp)
.describe("Knnloss")
.add_argument("data", "Symbol", "data")
.add_arguments(KnnlossParam::__FIELDS__());

}  // namespace op
} // namespace mxne
