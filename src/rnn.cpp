#include <caffe2/core/init.h>
#include <caffe2/core/logging.h>
#include <caffe2/proto/caffe2.pb.h>
#include <caffe2/core/common.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/utils/string_utils.h>
#include <c10/util/Flags.h>
#include <caffe2/core/blob.h>
#include <ATen/core/blob.h>
#include <glog/logging.h>

#include "util/modelutil.h"
#include "util/NetUtil.h"
#include "util/TensorUtil.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace caffe2 {
void AddFC(ModelUtil& model, const std::string& input,
			const std::string& output, int in_size, int out_size) {
	model.AddFcOps(input, output, in_size, out_size, 2, false);
}

void AddLSTM(ModelUtil &model, const std::string &input_blob,
             const std::string &seq_lengths, const std::string &hidden_init,
             const std::string &cell_init, int vocab_size, int hidden_size,
             const std::string &scope, std::string *hidden_output,
             std::string *cell_state) {
	model.AddLSTM(input_blob, seq_lengths, hidden_init, cell_init, vocab_size, hidden_size, scope, hidden_output, cell_state);
}

void AddSGD(ModelUtil &model, float base_learning_rate,
            const std::string &policy, int stepsize, float gamma) {
	model.AddSGD(base_learning_rate, policy, stepsize, gamma);
}


const auto cuda = false;
const int FLAGS_iters = 20000;
const int FLAGS_iters_to_report = 500;
const int FLAGS_seq_length = 32;
const int FLAGS_batch = 8;
const int FLAGS_gen_length = 500; //TODO
const bool FLAGS_dump_model = true;
const int FLAGS_hidden_size = 200;
const string FLAGS_model = "RNN";
const string FLAGS_train_data = "./res/shakespeare.txt";
const auto FLAGS_device = caffe2::DeviceType::CPU;
const bool FLAGS_outputdata = false;

const std::string loss_name = "loss";


void outputData(Workspace& workspace, const string paramName, const string outputParamName, const int iterNum) {
	if (!FLAGS_outputdata) {
//		std::cout << "Do not output data " << paramName << std::endl;
		return;
	}

	std::cout << "To output data of " << paramName << std::endl;
	std::stringstream fileNameStream;
	fileNameStream << "./logs/datafile_";
	fileNameStream << outputParamName << "_" << iterNum << ".txt";
	std::string dataFileName = fileNameStream.str();
	std::cout << "The file name " << dataFileName << std::endl;

	std::ofstream dataFile(dataFileName);
	auto paramBlob = workspace.GetBlob(paramName);
	auto paramTensor = BlobGetMutableTensor(paramBlob, DeviceType::CPU);
	float* paramData = paramTensor->template mutable_data<float>();
	for (int gIndex = 0; gIndex < paramTensor->numel(); gIndex ++) {
		dataFile << paramData[gIndex] << ", ";
			if ((gIndex + 1) % 8 == 0) {
				dataFile << std::endl;
		}
	}
	dataFile.close();
	std::cout << "End of param " << paramName << std::endl;
}

void checkArguments() {
	  std::cout << std::endl;
	  std::cout << "## Caffe2 RNNs and LSTM Tutorial ##" << std::endl;
	  std::cout << "https://caffe2.ai/docs/RNNs-and-LSTM-networks.html"
	            << std::endl;
	  std::cout << std::endl;


	  if (!std::ifstream(FLAGS_train_data).good()) {
	    std::cerr << "error: Text file missing: " << FLAGS_train_data << std::endl;
	    std::cerr << "Make sure to first run ./script/download_resource.sh"
	              << std::endl;
	    abort();
	  }

	  std::cout << "model: " << FLAGS_model << std::endl;
	  std::cout << "train-data: " << FLAGS_train_data << std::endl;
	  std::cout << "iters: " << FLAGS_iters << std::endl;
	  std::cout << "seq-length: " << FLAGS_seq_length << std::endl;
	  std::cout << "batch: " << FLAGS_batch << std::endl;
	  std::cout << "iters-to-report: " << FLAGS_iters_to_report << std::endl;
	  std::cout << "hidden-size: " << FLAGS_hidden_size << std::endl;
	  std::cout << "gen-length: " << FLAGS_gen_length << std::endl;

	  std::cout << "device: " << FLAGS_device << std::endl;
	  std::cout << "using cuda: " << (cuda ? "true" : "false") << std::endl;
	  ;
	  std::cout << "dump-model: " << (FLAGS_dump_model ? "true" : "false")
	            << std::endl;

	  std::cout << std::endl;
}

void inputPreprocess(std::string& text,
		std::vector<char>& vocab, std::map<char, int>& char_to_idx, std::map<int, char>& idx_to_char) {
	  std::ifstream infile(FLAGS_train_data);
	  std::stringstream buffer;
	  buffer << infile.rdbuf();
	  text = buffer.str();

	  if (!text.size()) {
	    std::cerr << "unable to read input text" << std::endl;
	    return;
	  }

	  std::set<char> vocab_set(text.begin(), text.end());
	  vocab.assign(vocab_set.begin(), vocab_set.end());

	  auto index = 0;
	  for (auto c : vocab) {
	    char_to_idx[c] = index;
	    idx_to_char[index++] = c;
	  }

	  std::cout << "Input has " << vocab.size()
	            << " characters. Total input size: " << text.size() << std::endl;
}

void configureBaseModel(ModelUtil& model, const int D, std::string& hidden_output, std::string& cell_state) {
	  model.AddInput("input_blob");
	  model.AddInput("seq_lengths");
	  model.AddInput("hidden_init");
	  model.AddInput("cell_init");
	  model.AddInput("target");

	  AddLSTM(model, "input_blob", "seq_lengths", "hidden_init", "cell_init", D,
			  	 FLAGS_hidden_size, "LSTM", &hidden_output, &cell_state);

	  AddFC(model, "LSTM/hidden_t_all", "char_rnn_blob_0", FLAGS_hidden_size, D);

	  model.AddSoftmaxOp("char_rnn_blob_0", "softmax", 2);
	  model.AddReshapeOp("softmax", "softmax_reshaped", {-1, D});
}

void configureTrainingModel(ModelUtil& trainModel) {
	  trainModel.AddLabelCrossEntropyOp("softmax_reshaped", "target", "xent");
	  trainModel.AddAveragedLossOp("xent", "loss");
	  trainModel.AddConstantFillWithOp(1.0f, "loss", "loss_grad");
	  trainModel.AddGradientOps();
	  AddSGD(trainModel, 0.1 * FLAGS_seq_length, "step", 1, 0.9999);
}

void configurePrepareModel(ModelUtil& prepare, std::string hidden_output, std::string cell_state) {
	//TODO: Not safe for input hidden_output, as it may be changed, that cause unexpected outcome of original model
	  prepare.AddCopyOp(hidden_output, "hidden_init");
	  prepare.AddCopyOp(cell_state, "cell_init");
	  prepare.AddInput(hidden_output);
	  prepare.AddInput(cell_state);
}

void initNets(ModelUtil& model, ModelUtil& trainModel, ModelUtil& prepare, Workspace& workspace) {
	  workspace.CreateBlob("input_blob");
	  workspace.CreateBlob("seq_lengths");
	  workspace.CreateBlob("target");

	  prepare.buildModel(workspace);
	  model.buildModel(workspace);
	  trainModel.buildModel(workspace);
//
//	  std::string findNetName = "trainCopy_train";
//	  auto getNet = workspace.GetNet(findNetName);
//	  if (getNet == nullptr) {
//		  std::cout << findNetName << " null" << std::endl;
//	  } else {
//		  std::cout << "Get net " << getNet->Name() << std::endl;
//	  }
}

void createSeqLenInput(Workspace& workspace) {
	  //TODO: Math::set
	  auto blob = workspace.CreateBlob("seq_lengths");
	  std::vector<int64_t> vDim{FLAGS_batch};
	  auto tensor = BlobGetMutableTensor(blob, vDim, DeviceType::CPU);
	  auto data = tensor->mutable_data<int>();
	  for (int i = 0; i < tensor->numel(); i ++) {
		  data[i] = FLAGS_seq_length;
	  }
}

void createInputTargetBlobs(Workspace& workspace,
		const int D, const int N,
		std::vector<int>& text_block_positions,
		std::vector<int>& text_block_starts,
		std::vector<int>& text_block_sizes,
		std::map<char, int>& char_to_idx,
		std::string& text,
		int& progress
		) {
	  std::vector<float> input (FLAGS_seq_length * FLAGS_batch * D);
	  std::vector<int> target(FLAGS_seq_length * FLAGS_batch);

	  for (auto e = 0; e < FLAGS_batch; e ++) {
		  for (auto i = 0; i < FLAGS_seq_length; i ++) {
			  auto pos = text_block_starts[e] + text_block_positions[e];
			  input[i * FLAGS_batch * D + e * D + char_to_idx[text[pos]]] = 1;
			  target[i * FLAGS_batch + e] = char_to_idx[text[(pos + 1) % N]];
			  text_block_positions[e] =
					  (text_block_positions[e] + 1) % text_block_sizes[e];
			  progress ++;
		  }
	  }

	  {
		  auto blob = workspace.CreateBlob("input_blob");
		  std::vector<int64_t> vDim{FLAGS_seq_length, FLAGS_batch, D};

		  auto tensor = BlobGetMutableTensor(
		            blob, vDim, DeviceType::CPU);
		  auto data = tensor->mutable_data<float>();
		  for (int i = 0; i < (tensor->numel()); i ++) {
			  data[i] = input[i];
		  }
	  }

	  {
		  auto blob = workspace.CreateBlob("target");
		  std::vector<int64_t> dim{FLAGS_seq_length * FLAGS_batch};
		  auto tensor = BlobGetMutableTensor(blob, dim, DeviceType::CPU);
		  auto data = tensor->mutable_data<int>();
		  for (int i = 0; i < tensor->numel(); i ++) {
			  data[i] = target[i];
		  }
	  }

}

void reportPerformance(clock_t& last_time, int& progress, const int num_iter) {
	  auto new_time = clock();
	  std::cout << "Characters Per Second: "
			  	  << ((size_t)progress * CLOCKS_PER_SEC / (new_time - last_time))
				  << std::endl;

	  std::cout << "Iterations Per Second: "
			  	  << ((size_t)FLAGS_iters_to_report * CLOCKS_PER_SEC / (new_time - last_time))
				  << std::endl;

	  last_time = new_time;
	  progress = 0;
	  std::cout << "-------------------------------- Iteration " << num_iter << "--------------------------"
			  	  << std::endl;
}

void updateLoss(Workspace& workspace, double& smooth_loss, float& last_n_loss) {
	  auto lossBlob = workspace.GetBlob(loss_name);
	  auto lossTensor = lossBlob->Get<TensorCPU>();
	  auto loss = lossTensor.data<float>()[0] * FLAGS_seq_length;

	  smooth_loss = 0.999 * smooth_loss + 0.001 * loss;
	  last_n_loss += loss;
}

void generateText(Workspace& workspace, ModelUtil& model, ModelUtil& prepare,
		const int D, const string predictions,
		std::vector<char>& vocab,
		std::map<char, int>& char_to_idx,
		std::map<int, char>& idx_to_char) {
	  std::stringstream text;
	  auto ch = vocab[(int)(vocab.size() * (float)rand() / RAND_MAX)];
	  text << ch;


	  for (auto i = 0; i < FLAGS_gen_length; i ++) {

		  {
			  auto blob = workspace.CreateBlob("seq_lengths");
			  std::vector<int64_t> dim{FLAGS_batch};
			  auto tensor = BlobGetMutableTensor(blob, dim, DeviceType::CPU);
			  auto data = tensor->mutable_data<int>();
			  for (int dataIndex = 0; dataIndex < tensor->numel(); dataIndex ++) {
				  data[dataIndex] = 1;
			  }
		  }

		  prepare.runModel(workspace);

		  std::vector<float> input(FLAGS_batch * D, 0);
		  input[char_to_idx[ch]] = 1;

		  {
			  auto blob = workspace.CreateBlob("input_blob");
			  std::vector<int64_t> dim{1, FLAGS_batch, D};
			  auto tensor = BlobGetMutableTensor(blob, dim, DeviceType::CPU);
			  auto data = tensor->mutable_data<float>();
			  for (int dataIndex = 0; dataIndex < tensor->numel(); dataIndex ++) {
				  data[dataIndex] = input[dataIndex];
			  }
		  }


		  model.runModel(workspace);

		  auto predBlob = workspace.GetBlob(predictions)->Get<TensorCPU>();
		  auto data = predBlob.data<float>();

		  auto r = (float)rand() / RAND_MAX;
		  auto next = vocab.size() - 1;
		  for (auto j = 0; j < vocab.size(); j ++) {
			  r -= data[j];
			  if (r <= 0) {
				  next = j;
				  break;
			  }
		  }

		  ch = idx_to_char[next];
		  text << ch;
	  }

	  std::cout << "/************************************** Generated text *****************************************/" << std::endl;
	  std::cout << text.str() << std::endl;
	  std::cout << "/**************************************** END **************************************************/" << std::endl;

}

void run() {
	checkArguments();

/************************************************** Input Preprocess *****************************************/
  std::vector<char> vocab;
  std::map<char, int> char_to_idx;
  std::map<int, char> idx_to_char;
  std::string text;

  inputPreprocess(text, vocab, char_to_idx, idx_to_char);
  auto D = (int)char_to_idx.size();


/*********************************************** char_rnn model **********************************************/

  std::string hidden_output;
  std::string cell_state;
  ModelUtil model("char_rnn");
  configureBaseModel(model, D, hidden_output, cell_state);

/******************************************** Train model ****************************************************/
  ModelUtil trainModel(model, "trainCopy");
  configureTrainingModel(trainModel);
//  std::cout << trainModel.Proto() << std::endl;
  auto predictions = "softmax";

/******************************************** Prepare model **************************************************/
  ModelUtil prepare("prepare_state");
  configurePrepareModel(prepare, hidden_output, cell_state);
//  std::cout << prepare.Proto() << std::endl;

/*********************************************** Prepare and create net **************************************/
  Workspace workspace("tmp");
  {
	  TensorUtil::ZeroFloats(workspace, std::vector<int>{1, FLAGS_batch, FLAGS_hidden_size}, cell_state);
	  TensorUtil::ZeroFloats(workspace, std::vector<int>{1, FLAGS_batch, FLAGS_hidden_size}, hidden_output);
  }
  initNets(model, trainModel, prepare, workspace);


  auto smooth_loss = -log(1.0 / D) * FLAGS_seq_length;
  auto last_n_iter = 0;
  auto last_n_loss = 0.0f;
  auto num_iter = 0;
  auto N = text.size();

  std::vector<int> text_block_positions(FLAGS_batch);
  auto text_block_size = N / FLAGS_batch;
  std::vector<int> text_block_starts;
  for (auto i = 0; i < N; i += text_block_size) {
	  text_block_starts.push_back(i);
  }

  std::vector<int> text_block_sizes(FLAGS_batch, text_block_size);
  text_block_sizes[FLAGS_batch - 1] += N % FLAGS_batch;
  CAFFE_ENFORCE_EQ(std::accumulate(text_block_sizes.begin(),
		  	  	  	  	  	  	  	  text_block_sizes.end(), 0, std::plus<int>()),
		  	  	  	  	  	  	  	  N);


  auto last_time = clock();
  auto progress = 0;

/*************************************************** Run ******************************************************/
  while (num_iter < FLAGS_iters) {
	  num_iter ++;
	  last_n_iter ++;

	  createSeqLenInput(workspace);

	  prepare.runModel(workspace);
//	  std::cout << "Prepare model run done" << std::endl;

	  createInputTargetBlobs(workspace,
	  		D, N,
	  		text_block_positions,
	  		text_block_starts,
	  		text_block_sizes,
	  		char_to_idx,
	  		text,
	  		progress
	  		);

	  trainModel.runModel(workspace);
//	  std::cout << "Train net run done " << std::endl;


	  updateLoss(workspace, smooth_loss, last_n_loss);

	  if (num_iter % FLAGS_iters_to_report == 0) {
		  reportPerformance(last_time, progress, num_iter);

		  generateText(workspace, model, prepare,
		  		D, predictions,
		  		vocab, char_to_idx, idx_to_char);

		  std::cout << std::endl;
		  std::cout << "Loss since last report: " << (last_n_loss / last_n_iter) << std::endl;
		  std::cout << "Smooth loss: " << smooth_loss << std::endl;

		  last_n_loss = 0.f;
		  last_n_iter = 0;
	  }
  }

}

void testZeros() {
	Workspace ws("test");
	std::string blobName = "Test";

	const int iterNum = 10000;
	for (int i = 0; i < iterNum; i ++) {
	std::vector<int> dim{1, 2};
	TensorUtil::ZeroFloats(ws, dim, blobName);

	std::vector<int> dim2{64, 32, 32};
	TensorUtil::ZeroFloats(ws, dim2, blobName);


	std::vector<int> dim3{1};
	TensorUtil::ZeroFloats(ws, dim3, blobName);


	std::vector<int> dim4{1200, 16};
	TensorUtil::ZeroFloats(ws, dim4, blobName);
	}

	return;
}

}

int main(int argc, char **argv) {
	google::InitGoogleLogging("TESTRNN");
	google::SetCommandLineOption("GLOG_minloglevel", "0");
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
