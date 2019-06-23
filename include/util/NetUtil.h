/*
 * NetUtil.h
 *
 *  Created on: Dec 26, 2018
 *      Author: zf
 */

#ifndef NETUTIL_H_
#define NETUTIL_H_

#include <vector>
#include <map>
#include <set>
#include <caffe2/core/net.h>
//#include "util/modelutil.h"
namespace caffe2 {

Argument* net_add_arg(OperatorDef& op, const std::string& name);
Argument* net_add_arg(OperatorDef& op, const std::string& name,
						int value);
Argument* net_add_arg(OperatorDef& op, const std::string& name,
						float value);
Argument* net_add_arg(OperatorDef& op, const std::string& name,
                      std::vector<int> values);
Argument* net_add_arg(OperatorDef& op, const std::string& name,
                      std::vector<std::string> values);
Argument* net_add_arg(OperatorDef& op, const std::string& name,
                      const std::string& value);

class NetUtil {
protected:
	NetDef net;

public:
//	static const std::map<std::string, std::string> customer_gradient;
	bool opHasOutput (const OperatorDef& op, const std::set<std::string>& names);

//public:
	NetUtil(std::string netName);
	NetUtil(NetDef netDef, std::string netName);
	virtual ~NetUtil();
	NetUtil(const NetUtil& from, const std::string nName);
	NetUtil& operator=(const NetUtil& other) = delete;


/********************************* INPUT ************************************************/
	const static int INPUT_MARK = 1;
	void AddInput(const std::string inputName);

/******************************** General **********************************************/
	const static int GENERAL_MARK = 1;

	OperatorDef* AddOp(const std::string& name,
	                  	  const std::vector<std::string>& inputs,
	                      const std::vector<std::string>& outputs);

	void AddOutput (const std::string output);
	OperatorDef* AddPrintOp(const std::string& param, bool toFile);


	void SetType(const std::string& typeName);

	std::string Proto();

	friend class ModelUtil;

};

} /* namespace caffe2 */

#endif /* NETUTIL_H_ */
