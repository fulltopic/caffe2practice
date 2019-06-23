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


void NetUtil::AddInput(const std::string inputName) {
	net.add_external_input(inputName);
}


OperatorDef* NetUtil::AddPrintOp(const std::string& param, bool toFile) {
  auto op = AddOp("Print", {param}, {});
  if (toFile) {
    net_add_arg(*op, "to_file", 1);
  }
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






} /* namespace caffe2 */
