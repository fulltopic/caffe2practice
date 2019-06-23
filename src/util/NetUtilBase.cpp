#include "util/NetUtilBase.h"

namespace caffe2 {
NetUtilBase::NetUtilBase(std::string netName): NetUtil(netName) {
	// TODO Auto-generated constructor stub
}

NetUtilBase::NetUtilBase(const NetUtilBase& from, const std::string nName): NetUtil(from, nName) {
}

NetUtilBase:: NetUtilBase(NetDef netDef, std::string netName): NetUtil(netDef, netName) {
}

NetUtilBase::~NetUtilBase() {
	// TODO Auto-generated destructor stub
}


OperatorDef* NetUtilBase::AddCreateDbOp(const std::string& reader,
                                    const std::string& db_type,
                                    const std::string& db_path) {
  auto op = AddOp("CreateDB", {}, {reader});
  net_add_arg(*op, "db_type", db_type);
  net_add_arg(*op, "db", db_path);
  return op;
}

OperatorDef* NetUtilBase::AddTensorProtosDbInputOp(const std::string& reader,
                                               const std::string& data,
                                               const std::string& label,
                                               int batch_size) {
  auto op = AddOp("TensorProtosDBInput", {reader}, {data, label});
  net_add_arg(*op, "batch_size", batch_size);
  return op;
}

OperatorDef* NetUtilBase::AddCastOp(const std::string& input,
                                const std::string& output,
                                TensorProto::DataType type) {
  auto op = AddOp("Cast", {input}, {output});
  net_add_arg(*op, "to", type);
  return op;
}



OperatorDef* NetUtilBase::AddXavierFillOp(const std::vector<int>& shape,
										const std::string& param) {
	auto op = AddOp("XavierFill", {}, {param});
	net_add_arg(*op, "shape", shape);

	return op;
}

OperatorDef* NetUtilBase::AddConstantFillOp(const std::vector<int>& shape,
								const std::string& param)
{
	auto op = AddOp("ConstantFill", {}, {param});
	net_add_arg(*op, "shape", shape);

	return op;
}
OperatorDef* NetUtilBase::AddConstantFillOp(const std::vector<int>& shape,
                                        int64_t value,
                                        const std::string& param) {
  auto op = AddOp("ConstantFill", {}, {param});
  net_add_arg(*op, "shape", shape);
  net_add_arg(*op, "value", (int)value);
  net_add_arg(*op, "dtype", TensorProto_DataType_INT64);
  return op;
}

OperatorDef* NetUtilBase::AddConstantFillOp(const std::vector<int>& shape,
                                        float value,
                                        const std::string& param) {
  auto op = AddOp("ConstantFill", {}, {param});
  net_add_arg(*op, "shape", shape);
  net_add_arg(*op, "value", value);
//  net_add_arg(*op, "dtype", TensorProto_DataType_INT64);
  return op;
}

OperatorDef* NetUtilBase::AddConstantFillWithOp(float value,
                                            const std::string& input,
                                            const std::string& output) {
  auto op = AddOp("ConstantFill", {input}, {output});
  net_add_arg(*op, "value", value);
  return op;
}


OperatorDef* NetUtilBase::AddCopyOp(const std::string& input, const std::string& output) {
	  return AddOp("Copy", {input}, {output});
}

OperatorDef* NetUtilBase::AddReshapeOp(const std::string& input,
                                   const std::string& output,
                                   const std::vector<int>& shape) {
	  auto op = AddOp("Reshape", {input}, {output, "_"});
	  net_add_arg(*op, "shape", shape);
	  return op;
}

OperatorDef* NetUtilBase::AddIterOp(const std::string& iter) {
  return AddOp("Iter", {iter}, {iter});
}


OperatorDef* NetUtilBase::AddLearningRateOp(const std::string& iter,
                                        const std::string& rate,
                                        float base_rate, float gamma) {
  auto op = AddOp("LearningRate", {iter}, {rate});
  net_add_arg(*op, "policy", "step");
  net_add_arg(*op, "stepsize", 1);
  net_add_arg(*op, "base_lr", -base_rate);
  net_add_arg(*op, "gamma", gamma);
  return op;
}

OperatorDef* NetUtilBase::AddSumOp(const std::vector<std::string>& inputs,
                               const std::string& sum) {
  return AddOp("Sum", inputs, {sum});
}

OperatorDef* NetUtilBase::AddScaleOp(const std::string& input,
                                 const std::string& output, float scale) {
  auto op = AddOp("Scale", {input}, {output});
  net_add_arg(*op, "scale", scale);
  return op;
}

OperatorDef* NetUtilBase::AddAtomicIterOp(const std::string& mutex, const std::string& iter) {
	return AddOp("AtomicIter", {mutex, iter}, {iter});
}

OperatorDef* NetUtilBase::AddCreateMutexOp(const std::string& param) {
	return AddOp("CreateMutex", {}, {param});
}

OperatorDef* NetUtilBase::AddAccuracyOp(const std::string& pred,
                                    const std::string& label,
                                    const std::string& accuracy, int top_k) {
  auto op = AddOp("Accuracy", {pred, label}, {accuracy});
  if (top_k) {
    net_add_arg(*op, "top_k", top_k);
  }
  return op;
}


OperatorDef* NetUtilBase::AddSummarizeOp(const std::string& param, bool to_file) {
  auto op = AddOp("Summarize", {param}, {});
  if (to_file) {
    net_add_arg(*op, "to_file", 1);
  }
  return op;
}
}
