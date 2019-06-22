/*
 * NetUtil.cpp
 *
 *  Created on: Dec 26, 2018
 *      Author: zf
 */

#include "util/NetUtil.h"
#include "util/modelutil.h"

#include <caffe2/core/operator_gradient.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/text_format.h>
#include <set>
#include <iostream>
namespace caffe2 {

const std::map<std::string, std::string> customer_gradient({
	{"EnsureCPUOutput", "CopyFromCPUInput"},
	{"CopyFromCPUInput", "EnsureCPUOutput"},
}	);
const std::set<std::string> pass_gradient {"Sum"};

Argument* net_add_arg(OperatorDef& op, const std::string& name) {
	auto arg = op.add_arg();
	arg->set_name(name);
	return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name, int value) {
	auto arg = net_add_arg(op, name);
	arg->set_i(value);
	return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name, float value) {
	auto arg = net_add_arg(op, name);
	arg->set_f(value);
	return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name,
                      const std::string& value) {
  auto arg = net_add_arg(op, name);
  arg->set_s(value);
  return arg;
}


Argument* net_add_arg(OperatorDef& op, const std::string& name,
                      std::vector<int> values) {
	auto arg = net_add_arg(op, name);
	for (auto v: values) {
		arg->add_ints(v);
	}

	return arg;
}

Argument* net_add_arg(OperatorDef& op, const std::string& name,
                      std::vector<std::string> values) {
	auto arg = net_add_arg(op, name);
	for (auto v: values) {
		arg->add_strings(v);
	}

	return arg;
}

NetUtil::NetUtil(std::string netName) {
	// TODO Auto-generated constructor stub
	net.set_name(netName);
}

NetUtil::NetUtil(const NetUtil& from, const std::string nName): net(from.net) {
	net.set_name(nName);
}

NetUtil:: NetUtil(NetDef netDef, std::string netName): net(netDef) {
	net.set_name(netName);
}

NetUtil::~NetUtil() {
	// TODO Auto-generated destructor stub
}

const std::set<std::string> trainable_ops{
    "Add",
    "AffineScale",
    "AveragedLoss",
    "AveragePool",
    "BackMean",
    "Concat",
    "Conv",
    "Diagonal",
    "Dropout",
    "EnsureCPUOutput",
    "FC",
    "LabelCrossEntropy",
    "LRN",
    "MaxPool",
    "Mul",
    "RecurrentNetwork",
    "Relu",
    "Reshape",
    "Slice",
    "Softmax",
    "SpatialBN",
    "SquaredL2",
    "SquaredL2Channel",
    "StopGradient",
    "Sum",
};

const std::set<std::string> non_trainable_ops{
    "Accuracy",
    "Cast",
    "Cout",
    "ConstantFill",
    "Iter",
    "Scale",
    "TensorProtosDBInput",
    "TimePlot",
    "ShowWorst",
};

bool NetUtil::isOpTrainable(const string& opType) const {
	return trainable_ops.count(opType) > 0;
}

bool NetUtil::opHasOutput (const OperatorDef& op, const std::set<std::string>& names) {
	for (const auto& output: op.output()) {
		if(names.count(output) > 0) {
			return true;
		}
	}
	return false;
}


OperatorDef* NetUtil::AddOp(const std::string& typeName,
								const std::vector<std::string>& inputs,
								const std::vector<std::string>& outputs) {
	auto op = this->net.add_op();
	op->set_type(typeName);

	for (auto input: inputs){
		op->add_input(input);
	}
	for (auto output: outputs) {
		op->add_output(output);
	}

	return op;
}

OperatorDef* NetUtil::AddXavierFillOp(const std::vector<int>& shape,
										const std::string& param) {
	auto op = AddOp("XavierFill", {}, {param});
	net_add_arg(*op, "shape", shape);

	return op;
}

OperatorDef* NetUtil::AddConstantFillOp(const std::vector<int>& shape,
								const std::string& param)
{
	auto op = AddOp("ConstantFill", {}, {param});
	net_add_arg(*op, "shape", shape);

	return op;
}
OperatorDef* NetUtil::AddConstantFillOp(const std::vector<int>& shape,
                                        int64_t value,
                                        const std::string& param) {
  auto op = AddOp("ConstantFill", {}, {param});
  net_add_arg(*op, "shape", shape);
  net_add_arg(*op, "value", (int)value);
  net_add_arg(*op, "dtype", TensorProto_DataType_INT64);
  return op;
}

OperatorDef* NetUtil::AddConstantFillOp(const std::vector<int>& shape,
                                        float value,
                                        const std::string& param) {
  auto op = AddOp("ConstantFill", {}, {param});
  net_add_arg(*op, "shape", shape);
  net_add_arg(*op, "value", value);
//  net_add_arg(*op, "dtype", TensorProto_DataType_INT64);
  return op;
}

OperatorDef* NetUtil::AddConstantFillWithOp(float value,
                                            const std::string& input,
                                            const std::string& output) {
  auto op = AddOp("ConstantFill", {input}, {output});
  net_add_arg(*op, "value", value);
  return op;
}

void NetUtil::AddInput(const std::string inputName) {
	net.add_external_input(inputName);
}
OperatorDef* NetUtil::AddTensorProtosDbInputOp(const std::string& reader,
                                               const std::string& data,
                                               const std::string& label,
                                               int batch_size) {
  auto op = AddOp("TensorProtosDBInput", {reader}, {data, label});
  net_add_arg(*op, "batch_size", batch_size);
  return op;
}

OperatorDef* NetUtil::AddCastOp(const std::string& input,
                                const std::string& output,
                                TensorProto::DataType type) {
  auto op = AddOp("Cast", {input}, {output});
  net_add_arg(*op, "to", type);
  return op;
}

OperatorDef* NetUtil::AddCreateDbOp(const std::string& reader,
                                    const std::string& db_type,
                                    const std::string& db_path) {
  auto op = AddOp("CreateDB", {}, {reader});
  net_add_arg(*op, "db_type", db_type);
  net_add_arg(*op, "db", db_path);
  return op;
}
OperatorDef* NetUtil::AddConvOp(const std::string& input, const std::string& w,
                                const std::string& b, const std::string& output,
                                int stride, int padding, int kernel, int group,
                                const std::string& order) {
	auto op = AddOp("Conv",
					b.size() > 0? std::vector<std::string>({input, w, b}): std::vector<std::string>({input, w}),
					{output});
	net_add_arg(*op, "stride", stride);
	net_add_arg(*op, "padding", padding);
	//TODO: More kernel size dimension
	//TODO: Channel
	net_add_arg(*op, "kernel", kernel);
	net_add_arg(*op, "order", order);

	return op;
}

OperatorDef* NetUtil::AddMaxPoolOp(const std::string& input,
                                   const std::string& output, int stride,
                                   int padding, int kernel,
                                   const std::string& order) {
	auto op = AddOp("MaxPool", {input}, {output});
	net_add_arg(*op, "stride", stride);
	net_add_arg(*op, "pad", padding);
	net_add_arg(*op, "kernel", kernel);
	net_add_arg(*op, "order", order);
//	TODO: What's legacy_pad?
	net_add_arg(*op, "legacy_pad", 3);
	return op;
}

OperatorDef* NetUtil::AddFcOp(const std::string& input, const std::string& w,
                     const std::string& b, const std::string& output,
                     int axis) {
	auto op = AddOp("FC", {input, w, b}, {output});
	if (axis != 1) {
		net_add_arg(*op, "axis", axis);
	}

	return op;
}

OperatorDef* NetUtil::AddSoftmaxOp(const std::string& input,
                                   const std::string& output, int axis) {
	auto op = AddOp("Softmax", {input}, {output});
	if (axis != 1) {
		net_add_arg(*op, "axis", axis);
	}
	return op;
}

OperatorDef* NetUtil::AddAccuracyOp(const std::string& pred,
                                    const std::string& label,
                                    const std::string& accuracy, int top_k) {
  auto op = AddOp("Accuracy", {pred, label}, {accuracy});
  if (top_k) {
    net_add_arg(*op, "top_k", top_k);
  }
  return op;
}

OperatorDef* NetUtil::AddCopyOp(const std::string& input, const std::string& output) {
	  return AddOp("Copy", {input}, {output});
}

OperatorDef* NetUtil::AddReshapeOp(const std::string& input,
                                   const std::string& output,
                                   const std::vector<int>& shape) {
	  auto op = AddOp("Reshape", {input}, {output, "_"});
	  net_add_arg(*op, "shape", shape);
	  return op;
}

OperatorDef* NetUtil::AddIterOp(const std::string& iter) {
  return AddOp("Iter", {iter}, {iter});
}


OperatorDef* NetUtil::AddLabelCrossEntropyOp(const std::string& pred,
                                             const std::string& label,
                                             const std::string& xent) {
  return AddOp("LabelCrossEntropy", {pred, label}, {xent});
}


OperatorDef* NetUtil::AddLearningRateOp(const std::string& iter,
                                        const std::string& rate,
                                        float base_rate, float gamma) {
  auto op = AddOp("LearningRate", {iter}, {rate});
  net_add_arg(*op, "policy", "step");
  net_add_arg(*op, "stepsize", 1);
  net_add_arg(*op, "base_lr", -base_rate);
  net_add_arg(*op, "gamma", gamma);
  return op;
}

OperatorDef* NetUtil::AddSumOp(const std::vector<std::string>& inputs,
                               const std::string& sum) {
  return AddOp("Sum", inputs, {sum});
}

std::vector<OperatorDef> NetUtil::CollectGradientOps(
    std::map<std::string, std::pair<int, int>>& split_inputs) const {
  std::set<std::string> external_inputs(net.external_input().begin(),
                                        net.external_input().end());
  std::vector<OperatorDef> gradient_ops;
  std::map<std::string, int> input_count;
  for (auto& op : net.op()) {
    if (this->isOpTrainable(op.type())) {
      gradient_ops.push_back(op);
      for (auto& input : op.input()) {
        auto& output = op.output();
        if (std::find(output.begin(), output.end(), input) == output.end()) {
          input_count[input]++;
          if (input_count[input] > 1) {
            split_inputs[input + ModelUtil::GetConstName(GRADIENTSUFFIX)] = {input_count[input],
                                                     input_count[input]};
          }
        }
      }
    } else if (non_trainable_ops.find(op.type()) == non_trainable_ops.end()) {
      CAFFE_THROW("unknown backprop operator type: " + op.type() + "name " + op.name());
    }
  }
  std::reverse(gradient_ops.begin(), gradient_ops.end());
  return gradient_ops;
}


OperatorDef* NetUtil::AddGradientOp(OperatorDef& op) {
	OperatorDef* grad = nullptr;
	std::vector<GradientWrapper> output(op.output_size());
	for (int i = 0; i < output.size(); i ++) {
		output[i].dense_ = op.output(i) + ModelUtil::GetConstName(GRADIENTSUFFIX);
//		std::cout << "To add gradient op output " << output[i].dense_  << " into " << op.name() << std::endl;
	}
	GradientOpsMeta meta = GetGradientForOp(op, output);
	if (meta.ops_.size() > 0) {
		for (auto& m: meta.ops_) {
			auto op = net.add_op();
			op->CopyFrom(m);
			if (grad == nullptr) {
				grad = op;
			}
		}
	}

//	if (grad == nullptr) {
//		std::cout << std::endl << "Not found valid gradient op" << std::endl;
//	} else {
//		std::cout << "Generated gradient op " << std::endl;
//		std::cout << grad->DebugString() << std::endl;
//		std::cout << "The end of gradient op" << std::endl << std::endl << std::endl;
//	}

	return grad;
}

OperatorDef* NetUtil::AddGradientOps(
    OperatorDef& op, std::map<std::string, std::pair<int, int>>& split_inputs,
    std::map<std::string, std::string>& pass_replace,
    std::set<std::string>& stop_inputs)
{
	OperatorDef* grad = nullptr;

	if (customer_gradient.count(op.type()) > 0) {
		grad = net.add_op();
		grad->set_type(customer_gradient.at(op.type()));
		//ZF:  So, in fact the gradient will always be the same shape as input
		for (auto arg: op.arg()) {
			auto copy = grad->add_arg();
			copy->CopyFrom(arg);
		}
		//ZF: And reverse the input / output gradient
		//So, conv op has gradients for both X input and W input
		for (auto output: op.output()) {
			grad->add_input(output + ModelUtil::GetConstName(GRADIENTSUFFIX));
		}
		for (auto input: op.input()) {
			grad->add_output(input + ModelUtil::GetConstName(GRADIENTSUFFIX));
		}
	} else if (pass_gradient.count(op.type()) > 0) {
		//TODO: What's this for?
		for (auto input: op.input()) {
			auto in = input + ModelUtil::GetConstName(GRADIENTSUFFIX);
			if (split_inputs.count(in) > 0 && split_inputs[in].first > 0) {
				split_inputs[in].first --;
				in += "_sum_" + std::to_string(split_inputs[in].first);
			}
			//TODO: As an in-place, it is supposed to have only one output
			pass_replace[in] = op.output(0) + ModelUtil::GetConstName(GRADIENTSUFFIX);
		}
	} else if (op.type() == "StopGradient"
				|| opHasOutput(op, stop_inputs)) {
		for (const auto& input: op.input()) {
			stop_inputs.insert(input);
		}
	} else {
		grad = AddGradientOp(op);
		if (grad == nullptr) {
			std::cerr << "No gradient for operator " << op.type() << std::endl;
			abort();
		}
	}

	if (grad != nullptr) {
		grad->set_is_gradient_op(true);
		for (int i = 0; i < grad->output_size(); i ++) {
			auto output = grad->output(i);
			if (split_inputs.count(output) && split_inputs[output].first > 0) {
				split_inputs[output].first --;
				grad->set_output(i, output + "_sum_" + std::to_string(split_inputs[output].first));
			}
		}

		for (int i = 0; i < grad->input_size(); i ++) {
			auto input = grad->input(i);
			if (pass_replace.count(input)) {
				grad->set_input(i, pass_replace[input]);
				pass_replace.erase(input);
			}
		}

		if (grad->type() == "SpatialBNGradient"
				&& grad->input(2) == grad->output(0)) {
			pass_replace[grad->output(0)] = grad->output(0) + "_fix";
			grad->set_output(0, grad->output(0) + "_fix");
		}
	}

	for (auto& p: split_inputs) {
		if (p.second.first == 0) {
			std::vector<std::string> inputs;
			for (int i = 0; i < p.second.second; i ++) {
				auto input = p.first + "_sum_" + std::to_string(i);
				if (pass_replace.count(input)) {
					auto in = pass_replace[input];
					pass_replace.erase(input);
					input = in;
				}
				inputs.push_back(input);
			}
			AddSumOp(inputs, p.first);
			p.second.first--;
		}
	}
	return grad;
}

OperatorDef* NetUtil::AddSummarizeOp(const std::string& param, bool to_file) {
  auto op = AddOp("Summarize", {param}, {});
  if (to_file) {
    net_add_arg(*op, "to_file", 1);
  }
  return op;
}


std::vector<std::string> NetUtil::CollectParams() {
  std::vector<std::string> params;
  std::set<std::string> external_inputs(net.external_input().begin(),
                                        net.external_input().end());
  for (const auto& op : net.op()) {
    auto& output = op.output();
    if (trainable_ops.find(op.type()) != trainable_ops.end()) {
      for (const auto& input : op.input()) {
        if (external_inputs.find(input) != external_inputs.end()) {
          if (std::find(output.begin(), output.end(), input) == output.end()) {
            params.push_back(input);
          }
        }
      }
    }
  }
  return params;
}

OperatorDef* NetUtil::AddScaleOp(const std::string& input,
                                 const std::string& output, float scale) {
  auto op = AddOp("Scale", {input}, {output});
  net_add_arg(*op, "scale", scale);
  return op;
}

OperatorDef* NetUtil::AddStopGradientOp(const std::string& param) {
  return AddOp("StopGradient", {param}, {param});
}

OperatorDef* NetUtil::AddPrintOp(const std::string& param, bool toFile) {
  auto op = AddOp("Print", {param}, {});
  if (toFile) {
    net_add_arg(*op, "to_file", 1);
  }
  return op;
}

OperatorDef* NetUtil::AddLSTMUnitOp(const std::vector<std::string>& inputs,
										const std::vector<std::string>& outputs,
										int drop_states, float forget_bias) {
	auto op = AddOp("LSTMUnit", inputs, outputs);
	net_add_arg(*op, "drop_states", drop_states);
	net_add_arg(*op, "forget_bias", forget_bias);

	return op;
}

void NetUtil::AddOutput(const std::string output) {
	net.add_external_output(output);
}

void NetUtil::SetType(const std::string& typeName) {
	net.set_type(typeName);
}


std::string NetUtil::Proto() {
  std::string s;
  google::protobuf::io::StringOutputStream stream(&s);
  google::protobuf::TextFormat::Print(net, &stream);
  return s;
}

OperatorDef* NetUtil::AddCreateMutexOp(const std::string& param) {
	return AddOp("CreateMutex", {}, {param});
}

OperatorDef* NetUtil::AddAtomicIterOp(const std::string& mutex, const std::string& iter) {
	return AddOp("AtomicIter", {mutex, iter}, {iter});
}


} /* namespace caffe2 */
