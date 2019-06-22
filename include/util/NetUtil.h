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
public:
	NetDef net;

//	static const std::map<std::string, std::string> customer_gradient;
	bool opHasOutput (const OperatorDef& op, const std::set<std::string>& names);

//public:
	NetUtil(std::string netName);
	NetUtil(NetDef netDef, std::string netName);
	virtual ~NetUtil();
	NetUtil(const NetUtil& other, const std::string nName);
	NetUtil& operator=(const NetUtil& other) = delete;


/********************************* UTIL *************************************************/
	const static int UTIL_MARK = 1;
/********************************* INPUT ************************************************/
	const static int INPUT_MARK = 1;
	OperatorDef* AddCreateDbOp(const std::string& reader,
	                                    const std::string& db_type,
	                                    const std::string& db_path);
	OperatorDef* AddTensorProtosDbInputOp(const std::string& reader,
	                                               const std::string& data,
	                                               const std::string& label,
	                                               int batch_size) ;
	void AddInput(const std::string inputName);

	OperatorDef* AddCastOp(const std::string& input,
	                                const std::string& output,
	                                TensorProto::DataType type);

	OperatorDef* AddXavierFillOp(const std::vector<int>& shape,
	                             	 const std::string& param);

	OperatorDef* AddConstantFillOp(const std::vector<int>& shape,
									const std::string& param);

	OperatorDef* AddConstantFillOp(const std::vector<int>& shape,
	                                        int64_t value,
	                                        const std::string& param);

	OperatorDef* AddConstantFillOp(const std::vector<int>& shape,
	                                        float value,
	                                        const std::string& param);

	OperatorDef* AddConstantFillWithOp(float value,
	                                            const std::string& input,
	                                            const std::string& output) ;
/******************************** General **********************************************/
	const static int GENERAL_MARK = 1;
	OperatorDef* AddCopyOp(const std::string& input, const std::string& output);

	OperatorDef* AddReshapeOp(const std::string& input,
	                                   const std::string& output,
	                                   const std::vector<int>& shape) ;
	OperatorDef* AddOp(const std::string& name,
	                  	  const std::vector<std::string>& inputs,
	                      const std::vector<std::string>& outputs);

	OperatorDef* AddIterOp(const std::string& iter);
	OperatorDef* AddLearningRateOp(const std::string& iter,
	                                 const std::string& rate, float base_rate,
	                                 float gamma = 0.999f);
	OperatorDef* AddSumOp(const std::vector<std::string>& inputs,
	                               const std::string& sum);
	std::vector<std::string> CollectParams();
	OperatorDef* AddScaleOp(const std::string& input,
	                                 const std::string& output, float scale) ;

	OperatorDef* AddAtomicIterOp(const std::string& mutex, const std::string& iter);

	OperatorDef* AddCreateMutexOp(const std::string& param);

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


	OperatorDef* AddAccuracyOp(const std::string& pred,
	                                    const std::string& label,
	                                    const std::string& accuracy, int top_k = 0);

	void AddOutput (const std::string output);


/********************************* LOSS **************************************************/
	const static int LOSS_MARK = 1;
	OperatorDef* AddLabelCrossEntropyOp(const std::string& pred,
	                                             const std::string& label,
	                                             const std::string& xent);
//	OperatorDef* AddAveragedLossOp(const std::string& input,
//	                                        const std::string& loss);
	inline OperatorDef* AddAveragedLossOp(const std::string& input,
	                                        const std::string& loss) {
	  return AddOp("AveragedLoss", {input}, {loss});
	}

	inline OperatorDef* AddWeightedSumOp(const std::vector<std::string>& inputs,
	                                         const std::string& sum) {
	    return AddOp("WeightedSum", inputs, {sum});
	}

	OperatorDef* AddSummarizeOp(const std::string& param, bool to_file);

/******************************** BACKWARD ***********************************************/
	const static int BACKWARD_MARK = 1;
	bool isOpTrainable(const std::string& opType) const;
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
	const static int DISPLAY_MARK = 1;

	OperatorDef* AddPrintOp(const std::string& param, bool toFile);

	void SetType(const std::string& typeName);

	std::string Proto();

	friend class ModelUtil;

};

} /* namespace caffe2 */

#endif /* NETUTIL_H_ */
