/*
 * NetUtilNN.cpp
 *
 *  Created on: Jun 23, 2019
 *      Author: zf
 */


#include "util/NetUtilNN.h"
#include "util/modelutil.h"
#include <caffe2/core/operator_gradient.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/text_format.h>

#include <vector>


namespace caffe2 {
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


const std::map<std::string, std::string> customer_gradient({
	{"EnsureCPUOutput", "CopyFromCPUInput"},
	{"CopyFromCPUInput", "EnsureCPUOutput"},
}	);

const std::set<std::string> pass_gradient {"Sum"};

NetUtilNN::NetUtilNN(std::string netName): NetUtilBase(netName) {
	// TODO Auto-generated constructor stub
}

NetUtilNN::NetUtilNN(const NetUtilNN& from, const std::string nName): NetUtilBase(from, nName) {
}

NetUtilNN:: NetUtilNN(NetDef netDef, std::string netName): NetUtilBase(netDef, netName) {
}

NetUtilNN::~NetUtilNN() {
}

OperatorDef* NetUtilNN::AddMaxPoolOp(const std::string& input,
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

OperatorDef* NetUtilNN::AddFcOp(const std::string& input, const std::string& w,
                     const std::string& b, const std::string& output,
                     int axis) {
	auto op = AddOp("FC", {input, w, b}, {output});
	if (axis != 1) {
		net_add_arg(*op, "axis", axis);
	}

	return op;
}

OperatorDef* NetUtilNN::AddConvOp(const std::string& input, const std::string& w,
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

OperatorDef* NetUtilNN::AddSoftmaxOp(const std::string& input,
                                   const std::string& output, int axis) {
	auto op = AddOp("Softmax", {input}, {output});
	if (axis != 1) {
		net_add_arg(*op, "axis", axis);
	}
	return op;
}

OperatorDef* NetUtilNN::AddLabelCrossEntropyOp(const std::string& pred,
                                             const std::string& label,
                                             const std::string& xent) {
  return AddOp("LabelCrossEntropy", {pred, label}, {xent});
}

bool NetUtilNN::isOpTrainable(const string& opType) const {
	return trainable_ops.count(opType) > 0;
}

std::vector<std::string> NetUtilNN::CollectParams() {
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


std::vector<OperatorDef> NetUtilNN::CollectGradientOps(
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

OperatorDef* NetUtilNN::AddGradientOp(OperatorDef& op) {
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


OperatorDef* NetUtilNN::AddGradientOps(
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


OperatorDef* NetUtilNN::AddStopGradientOp(const std::string& param) {
  return AddOp("StopGradient", {param}, {param});
}


OperatorDef* NetUtilNN::AddLSTMUnitOp(const std::vector<std::string>& inputs,
										const std::vector<std::string>& outputs,
										int drop_states, float forget_bias) {
	auto op = AddOp("LSTMUnit", inputs, outputs);
	net_add_arg(*op, "drop_states", drop_states);
	net_add_arg(*op, "forget_bias", forget_bias);

	return op;
}


}

