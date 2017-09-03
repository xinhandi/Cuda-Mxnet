/*!
 * Copyright (c) 2017 by Contributors
 * \file centerloss-inl.h
 * \brief centerloss 
 * \author deepearthgo
 */
#ifndef MXNET_OPERATOR_LSOFTMAX_INL_H_
#define MXNET_OPERATOR_LSOFTMAX_INL_H_

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

namespace centerloss_enum {
enum centerlossOpInputs {kData, kWeight, kLabel};
enum centerlossOpOutputs {kOut};
enum centerlossOpAuxiliay {kDiff,kCenter};
enum centerlossResource {kTempSpace};
}

struct centerlossParam : public dmlc::Parameter<centerlossParam> {
  float alpha;
  int batch_size;
  int num_class;
  float scale;
  int class_num;
  bool verbose;
  DMLC_DECLARE_PARAMETER(centerlossParam) {
    DMLC_DECLARE_FIELD(alpha).set_default(2).set_lower_bound(0)
    .describe("alpha");
    DMLC_DECLARE_FIELD(scale).set_default(1).set_lower_bound(1)
    .describe("scale");
    DMLC_DECLARE_FIELD(num_class).set_default(1).set_lower_bound(1)
    .describe("num_class");
    DMLC_DECLARE_FIELD(batch_size).set_default(8).set_lower_bound(8)
    .describe("batch_size");
    DMLC_DECLARE_FIELD(verbose).set_default(false)
    .describe("Log for scale change");
    DMLC_DECLARE_FIELD(class_num).set_default(2).set_lower_bound(1)
        .describe("class_num");
  }
};

template<typename xpu, typename DType>
class centerlossOp : public Operator {
 public:
  explicit centerlossOp(centerloss param) {
    this->param_ = param;
    const int batch_size = param.batch_size;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);
    CHECK_EQ(req.size(), 3);
    CHECK_EQ(req[centerloss_enum::kOut], kWriteTo);
    CHECK_EQ(aux_args.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int n = in_data[centerloss_enum::kData].size(0);
    const int m = in_data[centerloss_enum::kWeight].size(0);
    Tensor<xpu, 2, DType> x = in_data[centerloss_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> w = in_data[centerloss_enum::kWeight].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> label = in_data[centerloss_enum::kLabel].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 2, DType> out = out_data[centerloss_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> diff = aux_args[centerloss_enum::kDiff].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> center = aux_args[centerloss_enum::kCenter].FlatTo2D<xpu, DType>(s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    // original fully connected
    out = dot(x, w.T());
    if (ctx.is_train) {
      // large margin fully connected
      const int batch_size = param_.batch_size;
      centerlossForward(x, w, label, out, diff, center, batch_size);
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
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);
    CHECK_GE(in_grad.size(), 1);
    CHECK_GE(req.size(), 2);
    CHECK_EQ(req[centerloss_enum::kData], kWriteTo);
    CHECK_EQ(req[centerloss_enum::kWeight], kWriteTo);
    CHECK_EQ(aux_args.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int n = in_data[centerloss_enum::kData].size(0);
    const int m = in_data[centerloss_enum::kWeight].size(0);
    Tensor<xpu, 2, DType> x = in_data[centerloss_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> w = in_data[centerloss_enum::kWeight].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> label = in_data[centerloss_enum::kLabel].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 2, DType> o_grad = out_grad[centerloss_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> x_grad = in_grad[centerloss_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> diff = aux_args[centerloss_enum::kDiff].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> center = aux_args[centerloss_enum::kCenter].FlatTo2D<xpu, DType>(s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    // center loss
    const int batch_size = param_.batch_size;
    const float scale = param_.scale;
    centerlossBackward(x, w, label, o_grad, diff, center, sum1, batch_size, scale);
  }

 private:
  centerlossParam param_;
};  // class LSoftmaxOp

template<typename xpu>
Operator *CreateOp(centerlossParam param, int dtype);

#if DMLC_USE_CXX11
class centerlossProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "weight", "label"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }
  std::vector<std::string> ListAuxiliaryStates() const override{
	return {"diff", "center", "sum1"};
  }

  int NumOutputs() const override {
    return 1;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, label, weight]";
    const TShape &dshape = in_shape->at(centerloss_enum::kData);
    const TShape &lshape = in_shape->at(centerloss_enum::kLabel);
    CHECK_EQ(dshape.ndim(), 2) << "data shape should be (batch_size, feature_dim)";
    CHECK_EQ(lshape.ndim(), 1) << "label shape should be (batch_size,)";
    const int n = dshape[0];
    const int feature_dim = dshape[1];
    const int m = param_.num_hidden;
    SHAPE_ASSIGN_CHECK(*in_shape, centerloss_enum::kWeight, Shape2(m, feature_dim));
    out_shape->clear();
    out_shape->push_back(Shape2(n, m));  // output
    aux_shape->clear();
    aux_shape->push_back(Shape2(n,m)); // diff
    aux_shape->push_back(Shape2(n,m)); // center
    aux_shape->push_back(Shape1(m)); // sum1
    return true;
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[centerloss_enum::kOut], in_data[centerloss_enum::kData],
            in_data[centerloss_enum::kWeight], in_data[centerloss::kLabel],
            aux_args[centerloss_enum::kDiff], aux_args[centerloss_enum::kCenter],
            aux_args[centerloss_enum::sum1]};
  }

  std::string TypeString() const override {
    return "centerloss";
  }

  OperatorProperty *Copy() const override {
    auto ptr = new centerlossProp();
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
  centerlossParam param_;
};  // class LSoftmaxProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_LSOFTMAX_INL_H_
