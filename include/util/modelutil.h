#ifndef MODEL_UTIL_H
#define MODEL_UTIL_H

#include <caffe2/core/workspace.h>
#include <caffe2/core/net.h>
#include "util/NetUtilNN.h"

namespace caffe2 {


#define RESP_OP_1(OpName, Arg1) {		\
	if (modelType == TRAIN) {			\
		return trainModel.OpName(Arg1);	\
	}else {								\
		return initModel.OpName(Arg1);	\
	}									\
}

#define RESP_OP_2(OpName, Arg1, Arg2) {		\
	if (modelType == TRAIN) {					\
		return trainModel.OpName(Arg1, Arg2);	\
	}else {										\
		return initModel.OpName(Arg1, Arg2);	\
	}											\
}

#define RESP_OP_3(OpName, Arg1, Arg2, Arg3) {		\
	if (modelType == TRAIN) {					\
		return trainModel.OpName(Arg1, Arg2, Arg3);	\
	}else {										\
		return initModel.OpName(Arg1, Arg2, Arg3);	\
	}											\
}

#define RESP_OP_4(OpName, Arg1, Arg2, Arg3, Arg4) {			\
	if (modelType == TRAIN) {								\
		return trainModel.OpName(Arg1, Arg2, Arg3, Arg4);	\
	}else {													\
		return initModel.OpName(Arg1, Arg2, Arg3, Arg4);	\
	}														\
}

#define RESP_OP_5(OpName, Arg1, Arg2, Arg3, Arg4, Arg5) {			\
	if (modelType == TRAIN) {								\
		return trainModel.OpName(Arg1, Arg2, Arg3, Arg4, Arg5);	\
	}else {													\
		return initModel.OpName(Arg1, Arg2, Arg3, Arg4, Arg5);	\
	}														\
}


enum ModelType {
	INIT = 0,
	TRAIN = 1,
};
enum ConstItem {
	ITE = 0,
	GRADIENTSUFFIX = 1,
	ITENAME,
	INVALID
};

class ModelUtil {
protected:
	NetUtilNN initModel;
	NetUtilNN trainModel;
public:
	std::string modelName;
public:
//	ModelUtil(NetDef& iModel, NetDef& tModel, std::string mName);
	ModelUtil(std::string mName);
	ModelUtil(const ModelUtil& from, const std::string mName);
	ModelUtil(NetUtilNN &init_net, NetUtilNN &predict_net,
	                     const std::string &name);
	ModelUtil(NetDef &init_net, NetDef& predict_net, std::string name);
	void buildModel(Workspace& ws);
	void runModel(Workspace& ws);


	static const std::string& GetConstName(ConstItem item);


	void AddConvOps(const std::string &input, const std::string &output,
	                  int in_size, int out_size, int stride, int padding,
	                  int kernel, bool test = false);
	void AddMaxPoolOp(const std::string& input,
	                                   const std::string& output, int stride,
	                                   int padding, int kernel,
	                                   const std::string& order = "NCHW");
	void AddFcOps(const std::string &input, const std::string &output,
	                int in_size, int out_size, int axis = 1, bool test = false);
	void AddReluOp(const std::string& input,
	               	   const std::string& output);
	void AddSoftmaxOp(const std::string& input,
	                  	  const std::string& output, int axis = 1);
	OperatorDef* AddAccuracyOp(const std::string& pred,
	                                    const std::string& label,
	                                    const std::string& accuracy, int top_k = 0, ModelType modelType = TRAIN);


	OperatorDef* AddCopyOp(const std::string& input, const std::string& output, ModelType modelType = TRAIN) ;
	OperatorDef* AddReshapeOp(const std::string& input,
	                                   const std::string& output,
	                                   const std::vector<int>& shape) ;

	void AddIterOp(const std::string& iter);
	OperatorDef* AddLearningRateOp(const std::string& iter,
	                                 const std::string& rate, float base_rate,
	                                 float gamma = 0.999f,
	                                 ModelType modelType = TRAIN);
	OperatorDef* AddLabelCrossEntropyOp(const std::string& pred,
	                                             const std::string& label,
	                                             const std::string& xent, ModelType modelType = TRAIN);
//	OperatorDef* AddAveragedLossOp(const std::string& input,
//	                                        const std::string& loss,
//	                                        ModelType modelType = TRAIN) ;
	OperatorDef* AddAveragedLossOp(const std::string& input,
	                                        const std::string& loss,
	                                        ModelType modelType = TRAIN);
	OperatorDef* AddWeightedSumOp(const std::vector<std::string>& inputs,
	                                       const std::string& sum,
	                                       ModelType modelType = TRAIN) ;

	OperatorDef* AddSummarizeOp(const std::string& param, bool to_file, ModelType modelType = TRAIN);


	OperatorDef* AddConstantFillWithOp(float value,
	                                            const std::string& input,
	                                            const std::string& output,
	                                            ModelType modelType = TRAIN) ;

	OperatorDef* AddConstantFillOp(const std::vector<int>& shape,
									const std::string& param, ModelType modelType = TRAIN);

	OperatorDef* AddConstantFillOp(const std::vector<int>& shape,
	                                        int64_t value,
	                                        const std::string& param, ModelType modelType = TRAIN);
	OperatorDef* AddConstantFillOp(const std::vector<int>& shape,
	                                        float value,
	                                        const std::string& param, ModelType modelType = TRAIN);

	void AddInput(const std::string inputName, ModelType modelType = TRAIN);
	OperatorDef* AddCreateDbOp(const std::string& reader,
	                                    const std::string& db_type,
	                                    const std::string& db_path,
	                                    const ModelType modelType = INIT);
	OperatorDef* AddTensorProtosDbInputOp(const std::string& reader,
	                                               const std::string& data,
	                                               const std::string& label,
	                                               int batch_size,
	                                               ModelType modelType = TRAIN);
	OperatorDef* AddCastOp(const std::string& input,
	                                const std::string& output,
	                                TensorProto::DataType type,
	                                ModelType modelType = TRAIN);
	OperatorDef* AddScaleOp(const std::string& input,
	                                 const std::string& output, float scale,
	                                 ModelType modelType = TRAIN);
	OperatorDef* AddStopGradientOp(const std::string& param, ModelType modelType = TRAIN) ;

	void AddIterOps() ;



	std::vector<std::string> Params();

	void AddGradientOps ();

	void AddLSTM(const std::string &input_blob,
            const std::string &seq_lengths, const std::string &hidden_init,
            const std::string &cell_init, int vocab_size, int hidden_size,
            const std::string &scope, std::string *hidden_output,
            std::string *cell_state);

	void AddSGD(float base_learning_rate,
					const std::string& policy, int stepsize, float gamma) ;


	OperatorDef* AddPrintOp(const std::string& param, bool toFile, ModelType modelType = TRAIN);

	OperatorDef* AddRecurrentNetworkOp(const std::string& seq_lengths,
										const std::string& hidden_init,
										const std::string& cell_init,
										const std::string& scope,
										const std::string& hidden_output,
										const std::string& cell_state,
										bool force_cpu);


	std::string Proto();



	virtual ~ModelUtil();
	ModelUtil(const ModelUtil& other);
	ModelUtil& operator=(const ModelUtil& other) = delete;


};


}
#endif
