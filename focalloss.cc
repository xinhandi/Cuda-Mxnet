/*!
 * Copyright (c) 2017 by Contributors
 * \file focalloss.cc
 * \brief focalloss 
 * \author deepearthgo
 */
#include "./focalloss-inl.h"

namespace mshadow {

template <typename DType>
inline void FocallossForward(const Tensor<cpu, 2, DType> &cls_score,
		                        const Tensor<cpu, 2, DType> &cls_score_max,
			                      const Tensor<cpu, 2, DType> &cls_score_sub,
                            const Tensor<cpu, 1, DType> &label,
                            const Tensor<cpu, 2, DType> &pro_,
		                        const Tensor<cpu, 2, DType> &pro_sum,
			                      const Tensor<cpu, 2, DType> &pro_div,
                            const Tensor<cpu, 1, DType> &_pt,
                            Tensor<cpu, 2, DType> out,
                            const float gamma,
                            const float alpha) {
  LOG(FATAL) << "Not Implemented.";
}

template <typename DType>
inline void FocallossBackward(const Tensor<cpu, 1, DType> &label,
                              const Tensor<cpu, 1, DType> &_pt,
                              const Tensor<cpu, 2, DType> &pro_,
                              const Tensor<cpu, 2, DType> &o_grad,
                              const Tensor<cpu, 2, DType> &x_grad,
                              const float gamma,
                              const float alpha) {
  LOG(FATAL) << "Not Implemented.";
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(FocallossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new FocallossOp<cpu, DType>(param);
  })
  return op;
}

Operator *FocallossProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                         std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(centerlossParam);

MXNET_REGISTER_OP_PROPERTY(centerloss, centerlossProp)
.describe("Focalloss")
.add_argument("data", "Symbol", "data")
.add_argument("label", "Symbol", "label")
.add_arguments(centerlossParam::__FIELDS__());

}  // namespace op
} // namespace mxne
