#include "util/modelutil.h"
#include <caffe2/proto/caffe2_pb.h>

#include <iostream>

namespace caffe2 {
//	ModelUtil::ModelUtil(NetDef& iModel, NetDef& tModel, std::string mName): initModel(iModel), trainModel(tModel), modelName(mName){
//
//	}

	const std::string& ModelUtil::GetConstName(ConstItem item) {
		static const std::string TrainIteName = "TrainItr";
		static const std::string GradientSuffix = "_grad";
		static const std::string Invalid = "Invalid";
		static const std::string IterName = "iter";

		switch(item) {
		case ITE:
			return TrainIteName;
		case GRADIENTSUFFIX:
			return GradientSuffix;
		case ITENAME:
			return IterName;
		default:
			return Invalid;
		}
	}


	void ModelUtil::AddInput(const std::string inputName, ModelType modelType) {
		std::string type = modelType == TRAIN? "TRAIN": "TEST";
//		std::cout << "Add input " << inputName << " into " << this->modelName << " type = " << type << std::endl;
		if (modelType == TRAIN) {
			trainModel.AddInput(inputName);
		}else {
			initModel.AddInput(inputName);
		}
	}

	OperatorDef* ModelUtil::AddTensorProtosDbInputOp(const std::string& reader,
	                                               const std::string& data,
	                                               const std::string& label,
	                                               int batch_size,
	                                               ModelType modelType) {
		RESP_OP_4(AddTensorProtosDbInputOp, reader, data, label, batch_size)
	}

	OperatorDef* ModelUtil::AddCreateDbOp(const std::string& reader,
	                                    const std::string& db_type,
	                                    const std::string& db_path,
	                                    const ModelType modelType) {
		RESP_OP_3(AddCreateDbOp, reader, db_type, db_path)
	}

	void ModelUtil::AddConvOps(const std::string &input, const std::string &output,
		                  int in_size, int out_size, int stride, int padding,
		                  int kernel, bool test) {
		  if (!test) {
			  initModel.AddXavierFillOp({out_size, in_size, kernel, kernel}, output + "_w");
			  initModel.AddConstantFillOp({out_size}, output + "_b");
		  }
		  trainModel.AddInput(output + "_w");
		  trainModel.AddInput(output + "_b");
		  trainModel.AddConvOp(input, output + "_w", output + "_b", output, stride,
		                    padding, kernel);
	}

	void ModelUtil::AddMaxPoolOp(const std::string& input,
            const std::string& output, int stride,
            int padding, int kernel,
            const std::string& order) {
		trainModel.AddMaxPoolOp(input, output, stride, padding, kernel, order);
	}

	void ModelUtil::AddFcOps(const std::string &input, const std::string &output,
	                         int in_size, int out_size, int axis, bool test) {
		const std::string outputW = output + "_w";
		const std::string outputB = output + "_b";
		if (!test) {
			initModel.AddXavierFillOp({out_size, in_size}, outputW);
			initModel.AddConstantFillOp({out_size}, outputB);
		}

		trainModel.AddInput(outputW);
		trainModel.AddInput(outputB);
		trainModel.AddFcOp(input, outputW, outputB, output, axis);
	}

	void ModelUtil::AddReluOp(const std::string& input,
	                                const std::string& output) {
		trainModel.AddOp("Relu", {input}, {output});
	}

	void ModelUtil::AddSoftmaxOp(const std::string& input,
	                                   const std::string& output, int axis) {
		trainModel.AddSoftmaxOp(input, output, axis);
	}



	OperatorDef* ModelUtil::AddAccuracyOp(const std::string& pred,
	                                    const std::string& label,
	                                    const std::string& accuracy, int top_k, ModelType modelType) {
//		if (modelType == INIT) {
//			initModel.AddAccuracyOp(pred, label, accuracy, top_k);
//		} else {
//			trainModel.AddAccuracyOp(pred, label, accuracy, top_k);
//		}
		RESP_OP_4(AddAccuracyOp, pred, label, accuracy, top_k)
	}

	OperatorDef* ModelUtil::AddCopyOp(const std::string& input, const std::string& output, ModelType modelType) {
		RESP_OP_2(AddCopyOp, input, output)
	}

	OperatorDef* ModelUtil::AddReshapeOp(const std::string& input,
	                                   const std::string& output,
	                                   const std::vector<int>& shape) {
		return trainModel.AddReshapeOp(input, output, shape);
	}

	void ModelUtil::AddIterOp(const std::string& iter) {
		  auto op = initModel.AddConstantFillOp({1}, (int64_t)0, GetConstName(ITE));
		  //TODO: Why emphasize CPU device?
//		      ->mutable_device_option()
//		      ->set_device_type(CPU);
		  trainModel.AddInput(GetConstName(ITE));
		  trainModel.AddIterOp(GetConstName(ITE));
	}


	OperatorDef* ModelUtil::AddLabelCrossEntropyOp(const std::string& pred,
	                                             const std::string& label,
	                                             const std::string& xent,
	                                             ModelType modelType) {
		RESP_OP_3(AddLabelCrossEntropyOp, pred, label, xent)
	}

	OperatorDef* ModelUtil::AddAveragedLossOp(const std::string& input,
	                                        const std::string& loss,
	                                        ModelType modelType) {
		RESP_OP_3(AddOp, "AveragedLoss", {input}, {loss})
//	  return AddOp("AveragedLoss", {input}, {loss});
	}

//	OperatorDef* ModelUtil::AddAveragedLossOp(const std::string& input,
//	                                        const std::string& loss,
//	                                        ModelType modelType) {
////	  return AddOp("AveragedLoss", {input}, {loss});
//		RESP_OP_2(AddAveragedLossOp, input, loss)
//	}


	OperatorDef* ModelUtil::AddLearningRateOp(const std::string& iter,
	                                 const std::string& rate, float base_rate,
	                                 float gamma, ModelType modelType) {
		RESP_OP_4(AddLearningRateOp, iter, rate, base_rate, gamma)
	}

	OperatorDef* ModelUtil::AddWeightedSumOp(const std::vector<std::string>& inputs,
	                                       const std::string& sum,
	                                       ModelType modelType) {
		RESP_OP_2(AddWeightedSumOp, inputs, sum)
	}

	OperatorDef* ModelUtil::AddConstantFillWithOp(float value,
	                                            const std::string& input,
	                                            const std::string& output,
	                                            ModelType modelType) {
	   RESP_OP_3(AddConstantFillWithOp, value, input, output)
	}

	OperatorDef* ModelUtil::AddConstantFillOp(const std::vector<int>& shape,
									const std::string& param, ModelType modelType) {
		RESP_OP_2(AddConstantFillOp, shape, param)
	}

	OperatorDef* ModelUtil::AddConstantFillOp(const std::vector<int>& shape,
	                                        int64_t value,
	                                        const std::string& param, ModelType modelType) {
		RESP_OP_3(AddConstantFillOp, shape, value, param)
	}

	OperatorDef* ModelUtil::AddConstantFillOp(const std::vector<int>& shape,
	                                        float value,
	                                        const std::string& param, ModelType modelType) {
		RESP_OP_3(AddConstantFillOp, shape, value, param)
	}

	void ModelUtil::AddGradientOps () {
		  std::map<std::string, std::pair<int, int>> split_inputs;
		  std::map<std::string, std::string> pass_replace;
		  std::set<std::string> stop_inputs;
		  auto ops = trainModel.CollectGradientOps(split_inputs);
		  for (auto op : ops) {
		    trainModel.AddGradientOps(op, split_inputs, pass_replace, stop_inputs);
		  }
	}

	OperatorDef* ModelUtil::AddSummarizeOp(const std::string& param, bool to_file, ModelType modelType) {
	  RESP_OP_2(AddSummarizeOp, param, to_file)
	}


	std::vector<std::string> ModelUtil::Params() {
		return trainModel.CollectParams();
	}

	OperatorDef* ModelUtil::AddCastOp(const std::string& input,
	                                const std::string& output,
	                                TensorProto::DataType type,
	                                ModelType modelType) {
		RESP_OP_3(AddCastOp, input, output, type)
	}

	OperatorDef* ModelUtil::AddScaleOp(const std::string& input,
	                                 const std::string& output, float scale, ModelType modelType) {
		RESP_OP_3(AddScaleOp, input, output, scale)
	}

	OperatorDef* ModelUtil::AddStopGradientOp(const std::string& param, ModelType modelType) {
		RESP_OP_1(AddStopGradientOp, param)
	}

	void ModelUtil::AddIterOps() {
	  initModel.AddConstantFillOp({1}, (int64_t)0, GetConstName(ITENAME));

	  trainModel.AddInput(GetConstName(ITENAME));
	  trainModel.AddIterOp(GetConstName(ITENAME));
	}


	OperatorDef* ModelUtil::AddRecurrentNetworkOp(const std::string& seq_lengths,
										const std::string& hidden_init,
										const std::string& cell_init,
										const std::string& scope,
										const std::string& hidden_output,
										const std::string& cell_state,
										bool force_cpu) {
		NetUtilNN forward(scope);
		forward.SetType("rnn");
		forward.AddInput("input_t");
		forward.AddInput("timestep");
		forward.AddInput(scope + "/hidden_t_prev");
		forward.AddInput(scope + "/cell_t_prev");
		forward.AddInput(scope + "/gates_t_w");
		forward.AddInput(scope + "/gates_t_b");

		auto fc = forward.AddFcOp(scope + "/hidden_t_prev", scope + "/gates_t_w",
									scope + "/gates_t_b", scope + "/gates_t", 2);
		fc->set_engine("CPU"); //TODO
		auto sum = forward.AddSumOp({scope + "/gates_t", "input_t"}, scope + "/gates_t");
		forward.AddInput(seq_lengths);
		auto lstm = forward.AddLSTMUnitOp({scope + "/hidden_t_prev", scope + "/cell_t_prev", scope + "/gates_t", seq_lengths, "timestep"},
											{scope + "/hidden_t", scope + "/cell_t"});

		forward.AddOutput(scope + "/hidden_t");
		forward.AddOutput(scope + "/cell_t");


		NetUtilNN backward("RecurrentBackwardStep");
		backward.SetType("simple");
		backward.AddGradientOp(*lstm);

		auto grad = backward.AddGradientOp(*fc);
		grad->set_output(2, scope + "/hidden_t_prev_grad_split");
		backward.AddSumOp(
				{scope + "/hidden_t_prev_grad", scope + "/hidden_t_prev_grad_split"},
					scope + "/hidden_t_prev_grad");
		backward.AddInput(scope + "/gates_t");
		backward.AddInput(scope + "/hidden_t_grad");
		backward.AddInput(scope + "/cell_t_grad");
		backward.AddInput("input_t");
		backward.AddInput("timestep");
		backward.AddInput(scope + "/hidden_t_prev");
		backward.AddInput(scope + "/cell_t_prev");
		backward.AddInput(scope + "/gates_t_w");
		backward.AddInput(scope + "/gates_t_b");
		backward.AddInput(seq_lengths);
		backward.AddInput(scope + "/hidden_t");
		backward.AddInput(scope + "/cell_t");


		auto op = trainModel.AddOp("RecurrentNetwork",
					{scope + "/i2h", hidden_init, cell_init, scope + "/gates_t_w",
						scope + "/gates_t_b", seq_lengths},
					{scope + "/hidden_t_all", hidden_output, scope + "/cell_t_all",
						cell_state, scope + "/step_workspaces"}
		);

		  net_add_arg(*op, "link_internal",
		              std::vector<std::string>{
		                  scope + "/hidden_t_prev",
						  scope + "/hidden_t",
		                  scope + "/cell_t_prev",
						  scope + "/cell_t",
						  "input_t"});
		  net_add_arg(*op, "link_external",
		              std::vector<std::string>{
		                  scope + "/" + scope + "/hidden_t_prev_states",
		                  scope + "/" + scope + "/hidden_t_prev_states",
		                  scope + "/" + scope + "/cell_t_prev_states",
		                  scope + "/" + scope + "/cell_t_prev_states",
						  scope + "/i2h"});
		  net_add_arg(*op, "link_offset", std::vector<int>{0, 1, 0, 1, 0});

		  net_add_arg(
		      *op, "alias_src",
		      std::vector<std::string>{scope + "/" + scope + "/hidden_t_prev_states",
		                               scope + "/" + scope + "/hidden_t_prev_states",
		                               scope + "/" + scope + "/cell_t_prev_states",
		                               scope + "/" + scope + "/cell_t_prev_states"});
		  net_add_arg(*op, "alias_dst",
		              std::vector<std::string>{
			  	  	  	  scope + "/hidden_t_all",
						  hidden_output,
		                  scope + "/cell_t_all",
						  cell_state});
		  net_add_arg(*op, "alias_offset", std::vector<int>{1, -1, 1, -1});

		  net_add_arg(*op, "timestep", "timestep");
		  net_add_arg(*op, "recompute_blobs_on_backward");
		  net_add_arg(*op, "param", std::vector<int>{3, 4});
		  net_add_arg(*op, "param_grads",
		              std::vector<std::string>{scope + "/gates_t_w_grad",
		                                       scope + "/gates_t_b_grad"});
		  net_add_arg(*op, "outputs_with_grads", std::vector<int>{0});


		  net_add_arg(*op, "backward_link_external",
		              std::vector<std::string>{
		                  scope + "/" + scope + "/hidden_t_prev_states_grad",
		                  scope + "/" + scope + "/hidden_t_prev_states_grad",
		                  scope + "/" + scope + "/cell_t_prev_states_grad",
		                  scope + "/" + scope + "/cell_t_prev_states_grad",
		                  scope + "/i2h_grad"});
		  net_add_arg(*op, "backward_link_internal",
		              std::vector<std::string>{
		                  scope + "/hidden_t_grad",
						  scope + "/hidden_t_prev_grad",
		                  scope + "/cell_t_grad",
						  scope + "/cell_t_prev_grad",
		                  scope + "/gates_t_grad"});
		  net_add_arg(*op, "backward_link_offset", std::vector<int>{1, 0, 1, 0, 0});



		  net_add_arg(
		      *op, "recurrent_states",
		      std::vector<std::string>{scope + "/" + scope + "/hidden_t_prev_states",
		                               scope + "/" + scope + "/cell_t_prev_states"});

		  net_add_arg(*op, "initial_recurrent_state_ids", std::vector<int>{1, 2});


		  net_add_arg(*op, "step_net", forward.Proto());
		  net_add_arg(*op, "backward_step_net", backward.Proto());

		  return op;
	}


	void ModelUtil::AddLSTM(const std::string &input_blob,
            const std::string &seq_lengths, const std::string &hidden_init,
            const std::string &cell_init, int vocab_size, int hidden_size,
            const std::string &scope, std::string *hidden_output,
            std::string *cell_state) {
		*hidden_output = scope + "/hidden_t_last";
		*cell_state = scope + "/cell_t_last";
		AddFcOps(input_blob, scope + "/i2h", vocab_size, 4 * hidden_size, 2);

		initModel.AddXavierFillOp({4 * hidden_size, hidden_size},
				scope + "/gates_t_w");
		trainModel.AddInput(scope + "/gates_t_w");
		initModel.AddConstantFillOp({4 * hidden_size}, scope + "/gates_t_b");
		trainModel.AddInput(scope + "/gates_t_b");
		AddRecurrentNetworkOp(seq_lengths, hidden_init, cell_init,
					scope, *hidden_output, *cell_state,
					true); //TODO: To determine by input flag
	}

	void ModelUtil::AddSGD(float base_learning_rate,
						const std::string& policy, int stepsize, float gamma) {
		auto atomicOp = trainModel.AddAtomicIterOp("iteration_mutex", "optimizer_iteration");
		atomicOp->mutable_device_option()->set_device_type(static_cast<int>(CPU));

		auto op = initModel.AddConstantFillOp({1}, (int64_t)0, "optimizer_iteration");
		op->mutable_device_option()->set_device_type(static_cast<int>(CPU));

		auto mutexOp = initModel.AddCreateMutexOp("iteration_mutex");
		mutexOp->mutable_device_option()->set_device_type(static_cast<int>(CPU));

		trainModel.AddInput("iteration_mutex");
		trainModel.AddInput("optimizer_iteration");
		initModel.AddConstantFillOp({1}, 1.0f, "ONE");
		trainModel.AddInput("ONE");
		trainModel.AddLearningRateOp("optimizer_iteration", "lr", base_learning_rate, gamma);

		std::vector<std::string> params( {
			"LSTM/gates_t_w", "LSTM/i2h_b",
			"char_rnn_blob_0_w", "char_rnn_blob_0_b",
			"LSTM/gates_t_b", "LSTM/i2h_w"
		});
		for (auto &param: params) {
//			std::cout << "Add param into weighted " << param << std::endl;
			trainModel.AddWeightedSumOp({param, "ONE", param + "_grad", "lr"}, param);
		}
	}

}
