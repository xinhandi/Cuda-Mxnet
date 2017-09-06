/*!
 * Copyright (c) 2017 by Contributors
 * \file Asphere.cc
 * \brief Asphere from <SphereFace:Deep Hypersphere Embedding for Face Recognition 2017CVPR>
 * \author deepearthgo
 * \it's based on [https://github.com/wy1iu/sphereface][https://github.com/luoyetx/mx-lsoftmax]
 */
#include "./Asphere-inl.h"

namespace mshadow {

template <typename DType>
inline void AsphereForward(const Tensor<cpu, 2, DType> &x,
                            const Tensor<cpu, 2, DType> &w,
                            const Tensor<cpu, 1, DType> &label,
                            const Tensor<cpu, 2, DType> &out,
                            const Tensor<cpu, 1, DType> &x_norm,
                            const Tensor<cpu, 1, DType> &w_norm,
                            const Tensor<cpu, 1, DType> &k_table,
                            const Tensor<cpu, 1, DType> &c_table,
                            const int margin,
                            const DType beta) {
  LOG(FATAL) << "Not Implemented.";
}

template <typename DType>
inline void AsphereBackward(const Tensor<cpu, 2, DType> &x,
                             const Tensor<cpu, 2, DType> &w,
                             const Tensor<cpu, 1, DType> &label,
                             const Tensor<cpu, 1, DType> &x_norm,
                             const Tensor<cpu, 1, DType> &w_norm,
                             const Tensor<cpu, 2, DType> &o_grad,
                             const Tensor<cpu, 2, DType> &x_grad,
                             const Tensor<cpu, 2, DType> &w_grad,
                             const Tensor<cpu, 2, DType> &workspace,
                             const Tensor<cpu, 1, DType> &k_table,
                             const Tensor<cpu, 1, DType> &c_table,
                             const int margin,
                             const DType beta) {
  LOG(FATAL) << "Not Implemented.";
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(AsphereParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new AsphereOp<cpu, DType>(param);
  })
  return op;
}

Operator *AsphereProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                         std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(AsphereParam);

MXNET_REGISTER_OP_PROPERTY(Asphere, AsphereProp)
.describe("Asphere from <SphereFace:Deep Hypersphere Embedding for Face Recognition 2017CVPR>")
.add_argument("data", "Symbol", "data")
.add_argument("weight", "Symbol", "weight")
.add_argument("label", "Symbol", "label")
.add_arguments(AsphereParam::__FIELDS__());

}  // namespace op
} // namespace mxne
