/*!
 * Copyright (c) 2017 by Contributors
 * \file focalloss-inl.h
 * \brief focalloss
 * \author deepearthgo
 */
#ifndef MXNET_OPERATOR_FOCALLOSS_INL_H_
#define MXNET_OPERATOR_FOCALLOSS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cmath>
#include <map>
#include <vector>
#include <string>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace focalloss_enum {
enum FocallossOpInputs {kcls_score, klabel};
enum FocallossOpOutputs {kout};
enum FocallossOpAuxiliay {kcls_score_max,kcls_score_sub,kpro_,kpro_sum,kpro_div,k_pt};
}

struct FocallossParam : public dmlc::Parameter<FocallossParam> {
  float gamma;
  float alpha;
  bool verbose;
  DMLC_DECLARE_PARAMETER(FocallossParam) {
    DMLC_DECLARE_FIELD(gamma).set_default(2).set_lower_bound(1)
    .describe("Focalloss gamma");
    DMLC_DECLARE_FIELD(alpha).set_default(0.25).set_lower_bound(0)
    .describe("Focalloss alpha");
    DMLC_DECLARE_FIELD(verbose).set_default(false)
    .describe("Log for beta change");
  }
};

template<typename xpu, typename DType>
class FocallossOp : public Operator {
 public:
  explicit LSoftmaxOp(FocallossParam param) {
    this->param_ = param;
    const float gamma = param.gamma;
    const float alpha = param.alpha;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(req[focalloss_enum::kout], kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int n = in_data[focalloss_enum::kcls_score].size(0);
    const int m = in_data[focalloss_enum::kcls_score].size(0);
    Tensor<xpu, 2, DType> cls_score = in_data[focalloss_enum::kcls_score].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> label = in_data[focalloss_enum::klabel].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 2, DType> out = out_data[focalloss_enum::kout].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> cls_score_max = aux_args[focalloss_enum::kcls_score_max].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> cls_score_sub = aux_args[focalloss_enum::kcls_score_sub].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> pro_ = aux_args[focalloss_enum::kpro_].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> pro_sum = aux_args[focalloss_enum::kpro_sum].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> pro_div = aux_args[focalloss_enum::kpro_div].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> _pt = aux_args[focalloss_enum::k_pt].get_with_shape<xpu, 1, DType>(Shape1(n),s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    if (ctx.is_train) {
      const float gamma = param_.gamma;
      const float alpha = param_.alpha;
      FocalLossForward(cls_score, cls_score_max, cls_score_sub, label, pro_, pro_sum, pro_div, _pt, out, gamma, alpha);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_GE(in_grad.size(), 1);
    CHECK_GE(req.size(), 1);
    CHECK_EQ(req[focalloss_enum::klabel], kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int n = in_data[focalloss_enum::klabel].size;
    Tensor<xpu, 1, DType> label = in_data[focalloss_enum::klabel].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 2, DType> o_grad = out_grad[focalloss_enum::kout].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> x_grad = in_grad[focalloss_enum::kcls_score].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> _pt = aux_args[focalloss_enum::k_pt].get_with_shape<xpu, 1, DType>(Shape1(n),s);
    Tensor<xpu, 2, DType> pro_ = aux_args[focalloss_enum::kpro_].FlatTo2D<xpu, DType>(s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    const float gamma = param_.gamma;
    const float alpha = param_.alpha;
    FocalLossForward(label, _pt, pro_, o_grad, x_grad);
  }

 private:
  FocalLossParam param_;
};  // class FocalLoss

template<typename xpu>
Operator *CreateOp(FocalLossParam param, int dtype);

#if DMLC_USE_CXX11
class FocalLossProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  int NumOutputs() const override {
    return 3;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(focalloss_enum::kcls_score);
    const TShape &lshape = in_shape->at(focalloss_enum::klabel);
    CHECK_EQ(dshape.ndim(), 2) << "data shape should be (batch_size, n_class)";
    CHECK_EQ(lshape.ndim(), 1) << "label shape should be (batch_size,)";
    const int n = dshape[0];
    const int n_class = dshape[1];
    out_shape->clear();
    out_shape->push_back(Shape2(n, m));  // output
    aux_shape->clear();
    return true;
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[focalloss_enum::kcls_score], out_data[focalloss_enum::klabel],
            aux_args[focalloss_enum::k_pt], aux_args[focalloss_enum::kpro_]};
  }

  std::string TypeString() const override {
    return "focalloss";
  }

  OperatorProperty *Copy() const override {
    auto ptr = new focallossProp();
    ptr->param_ = param_;
    return ptr;
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  FocallossParam param_;
};  // class FocallossProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_FOCALLOSS_INL_H_
