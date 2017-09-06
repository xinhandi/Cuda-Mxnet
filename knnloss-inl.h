/*!
 * Copyright (c) 2017 by Contributors
 * \file knnloss-inl.h
 * \brief Knnloss
 * \author deepearthgo
 */
#ifndef MXNET_OPERATOR_KNNLOSS_INL_H_
#define MXNET_OPERATOR_KNNLOSS_INL_H_

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

namespace Knnloss_enum {
enum KnnlossOpInputs {kData};
enum KnnlossOpOutputs {kOut};
enum KnnlossOpAuxiliay {ksbm,kssbm,kknc,kdls,kkls};
}

struct centerlossParam : public dmlc::Parameter<centerlossParam> {
  float lamna;
  int batch_size;
  int k_num;
  bool verbose;
  DMLC_DECLARE_PARAMETER(centerlossParam) {
    DMLC_DECLARE_FIELD(lamna).set_default(2).set_lower_bound(0)
    .describe("lamna");
    DMLC_DECLARE_FIELD(batch_size).set_default(1).set_lower_bound(1)
    .describe("batch_size");
    DMLC_DECLARE_FIELD(k_num).set_default(1).set_lower_bound(1)
    .describe("k_num");
    DMLC_DECLARE_FIELD(verbose).set_default(false)
    .describe("Log for scale change");
  }
};

template<typename xpu, typename DType>
class KnnlossOp : public Operator {
 public:
  explicit KnnlossOp(Knnloss param) {
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
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(req[knnloss_enum::kOut], kWriteTo);
    CHECK_EQ(aux_args.size(), 5);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int n = in_data[Knnloss_enum::kData].size(0);
    const int m = in_data[Knnloss_enum::kData].size(1);
    Tensor<xpu, 2, DType> x = in_data[Knnloss_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[Knnloss_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> sbm = aux_args[Knnloss_enum::ksbm].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> ssbm = aux_args[Knnloss_enum::kssbm].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> knc = aux_args[Knnloss_enum::kknc].get_with_shape<xpu, 1, DType>(Shape1(n), s);
    Tensor<xpu, 2, DType> dls = aux_args[Knnloss_enum::kdls].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> kls = aux_args[Knnloss_enum::kkls].get_with_shape<xpu, 1, DType>(Shape1(1), s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    // original fully connected
    //out = dot(x, w.T());
    if (ctx.is_train) {
      // large margin fully connected
      const int batch_size = param_.batch_size;
      const int k_num = param_.k__num;
      KnnlossForward(x, sbm, ssbm, out, dls,knc, kls, k_num, batch_size);
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
    CHECK_EQ(req[knnloss_enum::kData], kWriteTo);
    CHECK_EQ(aux_args.size(), 5);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int n = in_data[Knnloss_enum::kData].size(0);
    const int m = in_data[Knnloss_enum::kData].size(1);
    Tensor<xpu, 2, DType> x = in_data[Knnloss_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> o_grad = out_grad[Knnloss_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> x_grad = in_grad[Knnloss_enum::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> dls = aux_args[Knnloss_enum::kdls].FlatTo2D<xpu, DType>(s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    // knnloss
    const int batch_size = param_.batch_size;
    const float lamna = param_.lamna;
    KnnlossBackward(x, o_grad, dls, batch_size, lamna);
  }

 private:
  knnlossParam param_;
};  // class knnloss

template<typename xpu>
Operator *CreateOp(KnnlossParam param, int dtype);

#if DMLC_USE_CXX11
class KnnlossProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }
  std::vector<std::string> ListAuxiliaryStates() const override{
	return {"sbm","ssbm","knc","dls","kls"};
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
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(knnloss_enum::kData);
    CHECK_EQ(dshape.ndim(), 2) << "data shape should be (batch_size, feature_dim)";
    const int n = dshape[0];
    const int feature_dim = dshape[1];
    out_shape->clear();
    out_shape->push_back(Shape2(n, m));  // output
    aux_shape->clear();
    aux_shape->push_back(Shape2(n,m)); // sbm
    aux_shape->push_back(Shape2(n,m)); // ssbm
    aux_shape->push_back(Shape2(n,m)); // knc
    aux_shape->push_back(Shape2(n,m)); // dls
    aux_shape->push_back(Shape1(1)); // kls
    return true;
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[Knnloss_enum::kOut], in_data[LKnnloss_enum::kData],
    	    aux_args[Knnloss_enum::ksbm], aux_args[Knnloss_enum::kssbm],
    	    aux_args[Knnloss_enum::kknc], aux_args[Knnloss_enum::kdls],
            aux_args[Knnloss_enum::kkls]};
  }

  std::string TypeString() const override {
    return "Knnloss";
  }

  OperatorProperty *Copy() const override {
    auto ptr = new KnnlossProp();
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
  knnParam param_;
};  // class Knnloss
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_LSOFTMAX_INL_H_
