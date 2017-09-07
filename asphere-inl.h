/*!
 * Copyright (c) 2017 by Contributors
 * \file Asphere-inl.h.cu
 * \brief Asphere from <SphereFace:Deep Hypersphere Embedding for Face Recognition 2017CVPR>
 * \author deepearthgo
 * \it's based on [https://github.com/wy1iu/sphereface][https://github.com/luoyetx/mx-lsoftmax]
 */

#ifndef MXNET_OPERATOR_ASPHERE_INL_H_
#define MXNET_OPERATOR_ASPHERE_INL_H_

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

namespace Asphere_enum {
enum AsphereOpInputs {kData, kWeight, kLabel};
enum AsphereOpOutputs {kOut, kDataNorm, kWeightNorm};
enum AsphereResource {kTempSpace};
}

struct AsphereParam : public dmlc::Parameter<AsphereParam> {
  int margin;
  float beta;
  float beta_min;
  float scale;
  int num_hidden;
  bool verbose;
  DMLC_DECLARE_PARAMETER(AsphereParam) {
    DMLC_DECLARE_FIELD(margin).set_default(2).set_lower_bound(1)
    .describe("Asphere margin");
    DMLC_DECLARE_FIELD(beta).set_default(1).set_lower_bound(0)
    .describe("Asphere beta, same as lambda to weight original value");
    DMLC_DECLARE_FIELD(beta_min).set_default(0).set_lower_bound(0)
    .describe("Minimum beta");
    DMLC_DECLARE_FIELD(scale).set_default(1).set_range(0, 1)
    .describe("Scale of beta during training for every iteration");
    DMLC_DECLARE_FIELD(num_hidden).set_lower_bound(1)
    .describe("Number of hidden nodes of the output");
    DMLC_DECLARE_FIELD(verbose).set_default(false)
    .describe("Log for beta change");
  }
};

template<typename xpu, typename DType>
class AsphereOp : public Operator {
 public:
  explicit AsphereOp(AsphereParam param) {
    this->param_ = param;
    // setup global lookup table
    k_table_.clear();
    c_table_.clear();
    k_table_.push_back(1);
    c_table_.push_back(1);
    const int margin = param.margin;
    const double pi = std::atan(1) * 4;
    double factor = 1;
    for (int i = 1; i <= margin; ++i) {
      factor = factor * (margin - i + 1) / i;
      k_table_.push_back(std::cos(i * pi / margin));
      c_table_.push_back(factor);
    }
    next_beta_ = param.beta * 0.1f;
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
    CHECK_EQ(req[Asphere_enum::kOut], kWriteTo);
    CHECK(req[Asphere_enum::kDataNorm] == kNullOp ||
          req[Asphere_enum::kDataNorm] == kWriteTo);
    CHECK(req[Asphere_enum::kWeightNorm] == kNullOp ||
          req[Asphere_enum::kWeightNorm] == kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int n = in_data[Asphere_enum::kData].size(0);
    const int m = in_data[Asphere_enum::kWeight].size(0);
    Tensor<xpu, 2, DType> x = in_data[Asphere_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> w = in_data[Asphere_enum::kWeight].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> label = in_data[Asphere_enum::kLabel].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 2, DType> out = out_data[Asphere_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> x_norm = out_data[Asphere_enum::kDataNorm].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 1, DType> w_norm = out_data[Asphere_enum::kWeightNorm].get_with_shape<xpu, 1, DType>(Shape1(m), s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    // original fully connected
    out = dot(x, w.T());
    if (ctx.is_train) {
      // large margin fully connected
      const int margin = param_.margin;
      const DType beta = static_cast<DType>(param_.beta);
      Tensor<cpu, 1, DType> k_table_cpu(k_table_.data(), Shape1(k_table_.size()));
      Tensor<cpu, 1, DType> c_table_cpu(c_table_.data(), Shape1(c_table_.size()));
      Tensor<xpu, 1, DType> k_table_xpu(Shape1(k_table_.size()));
      Tensor<xpu, 1, DType> c_table_xpu(Shape1(c_table_.size()));
      k_table_xpu.set_stream(s);
      c_table_xpu.set_stream(s);
      AllocSpace(&k_table_xpu);
      AllocSpace(&c_table_xpu);
      Copy(k_table_xpu, k_table_cpu, s);
      Copy(c_table_xpu, c_table_cpu, s);
      AsphereForward(x, w, label, out, x_norm, w_norm, k_table_xpu, c_table_xpu, margin, beta);
      FreeSpace(&k_table_xpu);
      FreeSpace(&c_table_xpu);
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
    CHECK_GE(in_grad.size(), 2);
    CHECK_GE(req.size(), 2);
    CHECK_EQ(req[Asphere_enum::kData], kWriteTo);
    CHECK_EQ(req[Asphere_enum::kWeight], kWriteTo);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int n = in_data[Asphere_enum::kData].size(0);
    const int m = in_data[Asphere_enum::kWeight].size(0);
    Tensor<xpu, 2, DType> x = in_data[Asphere_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> w = in_data[Asphere_enum::kWeight].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, DType> label = in_data[Asphere_enum::kLabel].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 1, DType> x_norm = out_data[Asphere_enum::kDataNorm].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 1, DType> w_norm = out_data[Asphere_enum::kWeightNorm].get_with_shape<xpu, 1, DType>(Shape1(m), s);
    Tensor<xpu, 2, DType> o_grad = out_grad[Asphere_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> x_grad = in_grad[Asphere_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> w_grad = in_grad[Asphere_enum::kWeight].FlatTo2D<xpu, DType>(s);
    // workspace is used for cos_t, cos_mt, k, sin2_t, fo and cos_t_m for every data point
    Tensor<xpu, 2, DType> workspace = ctx.requested[Asphere_enum::kTempSpace].get_space_typed<xpu, 2, DType>(Shape2(6, n), s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    // original fully connected
    x_grad = dot(o_grad, w);
    w_grad = dot(o_grad.T(), x);
    // large margin fully connected
    const int margin = param_.margin;
    const DType beta = static_cast<DType>(param_.beta);
    Tensor<cpu, 1, DType> k_table_cpu(k_table_.data(), Shape1(k_table_.size()));
    Tensor<cpu, 1, DType> c_table_cpu(c_table_.data(), Shape1(c_table_.size()));
    Tensor<xpu, 1, DType> k_table_xpu(Shape1(k_table_.size()));
    Tensor<xpu, 1, DType> c_table_xpu(Shape1(c_table_.size()));
    k_table_xpu.set_stream(s);
    c_table_xpu.set_stream(s);
    AllocSpace(&k_table_xpu);
    AllocSpace(&c_table_xpu);
    Copy(k_table_xpu, k_table_cpu, s);
    Copy(c_table_xpu, c_table_cpu, s);
    AsphereBackward(x, w, label, x_norm, w_norm, o_grad, x_grad, w_grad, workspace,
                     k_table_xpu, c_table_xpu, margin, beta);
    FreeSpace(&k_table_xpu);
    FreeSpace(&c_table_xpu);
    // dirty hack, should also work for multi device
    param_.beta *= param_.scale;
    param_.beta = std::max(param_.beta, param_.beta_min);
    if (param_.beta < next_beta_) {
      next_beta_ *= 0.1f;
      if (param_.verbose) {
        LOG(INFO) << "Asphere changes beta to " << param_.beta;
      }
    }
  }

 private:
  AsphereParam param_;
  // global lookup table
  std::vector<DType> k_table_;
  std::vector<DType> c_table_;
  float next_beta_;
};  // class AsphereOp

template<typename xpu>
Operator *CreateOp(AsphereParam param, int dtype);

#if DMLC_USE_CXX11
class AsphereProp : public OperatorProperty {
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
    return {"output", "data_norm", "weight_norm"};
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
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, label, weight]";
    const TShape &dshape = in_shape->at(Asphere_enum::kData);
    const TShape &lshape = in_shape->at(Asphere_enum::kLabel);
    CHECK_EQ(dshape.ndim(), 2) << "data shape should be (batch_size, feature_dim)";
    CHECK_EQ(lshape.ndim(), 1) << "label shape should be (batch_size,)";
    const int n = dshape[0];
    const int feature_dim = dshape[1];
    const int m = param_.num_hidden;
    SHAPE_ASSIGN_CHECK(*in_shape, Asphere_enum::kWeight, Shape2(m, feature_dim));
    out_shape->clear();
    out_shape->push_back(Shape2(n, m));  // output
    out_shape->push_back(Shape1(n));  // data norm
    out_shape->push_back(Shape1(m));  // weight norm
    aux_shape->clear();
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
    return {out_grad[Asphere_enum::kOut], out_data[Asphere_enum::kDataNorm],
            out_data[Asphere_enum::kWeightNorm], in_data[Asphere_enum::kData],
            in_data[Asphere_enum::kWeight], in_data[Asphere_enum::kLabel]};
  }

  std::string TypeString() const override {
    return "Asphere";
  }

  OperatorProperty *Copy() const override {
    auto ptr = new AsphereProp();
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
  AsphereParam param_;
};  // class AsphereProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_Asphere_INL_H_
