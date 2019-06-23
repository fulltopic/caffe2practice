/*
 * NetUtilBase.h
 *
 *  Created on: Jun 23, 2019
 *      Author: zf
 */

#ifndef INCLUDE_UTIL_NETUTILBASE_H_
#define INCLUDE_UTIL_NETUTILBASE_H_

#include "NetUtil.h"

namespace caffe2 {
class NetUtilBase: public NetUtil {
public:
	NetUtilBase(std::string netName);
	NetUtilBase(NetDef netDef, std::string netName);
	virtual ~NetUtilBase();
	NetUtilBase(const NetUtilBase& other, const std::string nName);
	NetUtilBase& operator=(const NetUtilBase& other) = delete;

	const static int INPUT_MARK = 1;
	OperatorDef* AddCreateDbOp(const std::string& reader,
	                                    const std::string& db_type,
	                                    const std::string& db_path);
	OperatorDef* AddTensorProtosDbInputOp(const std::string& reader,
	                                               const std::string& data,
	                                               const std::string& label,
	                                               int batch_size) ;
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



	const static int GENERAL_MARK = 1;
	OperatorDef* AddCopyOp(const std::string& input, const std::string& output);

	OperatorDef* AddReshapeOp(const std::string& input,
	                                   const std::string& output,
	                                   const std::vector<int>& shape) ;
	OperatorDef* AddIterOp(const std::string& iter);
	OperatorDef* AddLearningRateOp(const std::string& iter,
	                                 const std::string& rate, float base_rate,
	                                 float gamma = 0.999f);
	OperatorDef* AddSumOp(const std::vector<std::string>& inputs,
	                               const std::string& sum);
	OperatorDef* AddScaleOp(const std::string& input,
	                                 const std::string& output, float scale) ;

	OperatorDef* AddAtomicIterOp(const std::string& mutex, const std::string& iter);

	OperatorDef* AddCreateMutexOp(const std::string& param);

	const static int OUTPUT_MARK = 1;
	OperatorDef* AddAccuracyOp(const std::string& pred,
	                                    const std::string& label,
	                                    const std::string& accuracy, int top_k = 0);


	const static int LOSS_MARK = 1;
	inline OperatorDef* AddWeightedSumOp(const std::vector<std::string>& inputs,
	                                         const std::string& sum) {
	    return AddOp("WeightedSum", inputs, {sum});
	}

	OperatorDef* AddSummarizeOp(const std::string& param, bool to_file);

	const static int DISPLAY_MARK = 1;


};
}



#endif /* INCLUDE_UTIL_NETUTILBASE_H_ */
