/*
 * NetUtilNN.h
 *
 *  Created on: Jun 23, 2019
 *      Author: zf
 */

#ifndef INCLUDE_UTIL_NETUTILNN_H_
#define INCLUDE_UTIL_NETUTILNN_H_

#include "NetUtilBase.h"

namespace caffe2 {
class NetUtilNN: public NetUtilBase {
public:
	NetUtilNN(std::string netName);
	NetUtilNN(NetDef netDef, std::string netName);
	virtual ~NetUtilNN();
	NetUtilNN(const NetUtilNN& other, const std::string nName);
	NetUtilNN& operator=(const NetUtilNN& other) = delete;

	/******************************** FORWARD *********************************************/
	const static int FORWARD_MARK = 1;
	OperatorDef* AddMaxPoolOp(const std::string& input,
		                      	  const std::string& output, int stride,
		                          int padding, int kernel,
		                          const std::string& order = "NCHW") ;
	OperatorDef* AddFcOp(const std::string& input, const std::string& w,
		                 	 const std::string& b, const std::string& output,
		                     int axis = 1);

	OperatorDef* AddConvOp(const std::string& input, const std::string& w,
		                   	   const std::string& b, const std::string& output,
		                       int stride, int padding, int kernel, int group = 0,
		                       const std::string& order = "NCHW");


	/******************************* OUTPUT ************************************************/
	const static int OUTPUT_MARK = 1;
	OperatorDef* AddSoftmaxOp(const std::string& input,
		                                   const std::string& output, int axis = 1);


	/********************************* LOSS **************************************************/
	const static int LOSS_MARK = 1;
	OperatorDef* AddLabelCrossEntropyOp(const std::string& pred,
	                                             const std::string& label,
	                                             const std::string& xent);
	inline OperatorDef* AddAveragedLossOp(const std::string& input,
	                                        const std::string& loss) {
	  return AddOp("AveragedLoss", {input}, {loss});
	}


	/******************************** BACKWARD ***********************************************/
	bool isOpTrainable(const std::string& opType) const;
	std::vector<std::string> CollectParams();


	const static int BACKWARD_MARK = 1;
	OperatorDef* AddGradientOp(OperatorDef& op);

	std::vector<OperatorDef> CollectGradientOps(
			std::map<std::string, std::pair<int, int>>& split_inputs) const;

	OperatorDef* AddGradientOps (OperatorDef& op, std::map<std::string, std::pair<int, int>>& split_inputs,
			std::map<std::string, std::string>& pass_replace,
			std::set<std::string>& stop_inputs);
	OperatorDef* AddStopGradientOp(const std::string& param);

	/******************************* RNN *****************************************************/
	const static int RNN_MARK = 1;
	OperatorDef* AddLSTMUnitOp(const std::vector<std::string>& inputs,
			const std::vector<std::string>& outputs,
			int drop_states = 0,
			float forget_bias = 0.0f);

	 /***************************** DISPLAY ***************************************************/

};
}



#endif /* INCLUDE_UTIL_NETUTILNN_H_ */
