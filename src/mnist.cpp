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
//#include "util/NetUtil.h"
//#include "util/ModelDisplayUtil.h"

#include <iostream>
#include <string>

namespace caffe2 {
using std::cout;
using std::endl;

//const std::string trainDbPath("res/lmdb/mnist-train-nchw-lmdb");
const std::string lmDbPath("../res/lmdb/mnist-train-nchw-lmdb");
const std::string lmDbTestPath("../res/lmdb/mnist-test-nchw-lmdb");

const std::string miniDbPath("../res/mnist-test-nhwc-minidb");
const std::string levelDbPath("../res/leveldb/mnist-train-nchw-leveldb");

const std::string testDbPath("../build/res");
const int batchSize = 16;


void addInputByUtil(ModelUtil& model, int batch_size, const std::string &db, const std::string& db_type) {
	model.AddCreateDbOp("dbreader", db_type, db);
	model.AddInput("dbreader", TRAIN);
	model.AddTensorProtosDbInputOp("dbreader", "data_uint8", "label", batch_size, TRAIN);
	model.AddCastOp("data_uint8", "data", TensorProto_DataType_FLOAT);
	model.AddScaleOp("data", "data", 1.0f / 255);
	model.AddStopGradientOp("data");
}

void addTrainModel(ModelUtil& model, bool test) {
	model.AddConvOps("data", "conv1", 1, 20, 1, 0, 3, test);
	model.AddMaxPoolOp("conv1", "pool1", 2, 0, 2);
	model.AddConvOps("pool1", "conv2", 20, 50, 1, 0, 5, test);
	model.AddMaxPoolOp("conv2", "pool2", 2, 0, 2);
	model.AddFcOps("pool2", "fc3", 1250, 500, 1, test);
	model.AddReluOp("fc3", "fc3");
	model.AddFcOps("fc3", "predz", 500, 10, 1, test);
	model.AddSoftmaxOp("predz", "softmax");
}

void addAccuracy(ModelUtil &model, std::string outputName) {
	model.AddAccuracyOp("softmax", "label", outputName, TRAIN); //outputName = "accuracy"

//	displayModel.AddTimePlotOp("accuracy");
	model.AddIterOps();
}

void addBookkeepingOperators(ModelUtil& model) {
	model.AddPrintOp("accuracy", true);
	model.AddPrintOp("loss", true);

	for (auto param: model.Params()) {
		model.AddSummarizeOp(param, true);
		model.AddSummarizeOp(param + "_grad", true);
	}
}

void addBackward(ModelUtil& model) {
	model.AddLabelCrossEntropyOp("softmax", "label", "xent");
	model.AddAveragedLossOp("xent", "loss");

//	displayModel.AddShowWorstOp("softmax", "label", "data", 256, 0);
//	displayModel.AddTimePlotOp("loss");

	addAccuracy(model, "accuracy");

	model.AddConstantFillWithOp(1.0, "loss", "loss_grad");
	model.AddGradientOps();

//	model.AddIterOps();
	model.AddLearningRateOp("iter", "LR", 0.1);

	model.AddConstantFillOp({1}, (float)1.0, "ONE", INIT);
	model.AddInput("ONE");

	for (auto param: model.Params()) {
		model.AddWeightedSumOp({param, "ONE", param + "_grad", "LR"}, param);
	}
}

void run() {
	Workspace ws("mnisttest");
//	NetDef initModel;
//	initModel.set_name("mnist_init");
//	NetDef trainModel;
//	trainModel.set_name("mnist_train");
	ModelUtil model("mnist_train");
//	ModelDisplayUtil displayModel(initModel, trainModel);

	addInputByUtil(model, 64, lmDbPath, "lmdb");
	addTrainModel(model, false);
	addBackward(model);
	addBookkeepingOperators(model);

	model.buildModel(ws);
//	ws.RunNetOnce(initModel);
//	ws.CreateNet(trainModel);

	for(int i = 0; i < 200; i ++) {
//		ws.RunNet(trainModel.name());
		model.runModel(ws);
	}

}

void runWithTest() {
	Workspace ws("mnisttest");
//	NetDef initModel;
//	initModel.set_name("mnist_init");
//	NetDef trainModel;
//	trainModel.set_name("mnist_train");
	ModelUtil model("mnist_train");
//	ModelDisplayUtil displayModel(initModel, trainModel);

//	addInputByUtil(model, 64, levelDbPath, "leveldb");
	addInputByUtil(model, 64, lmDbPath, "lmdb");
	addTrainModel(model, false);
	addBackward(model);
	addBookkeepingOperators(model);

	model.buildModel(ws);
//	ws.RunNetOnce(initModel);
//	ws.CreateNet(trainModel);


////////////////////////////////////////////////////////////////////////////////////
//	NetDef testInitNet;
//	testInitNet.set_name("test_mnist_ini");
//	NetDef testTrainNet;
//	testTrainNet.set_name("test_mnist_train");
	ModelUtil testModel("mnist_test");
//	ModelDisplayUtil testDisplayModel(testInitNet, testTrainNet);

	addInputByUtil(testModel, 16, lmDbTestPath, "lmdb");
	addTrainModel(testModel, true);


	addAccuracy(testModel, "test_accuracy");
	testModel.AddPrintOp("test_accuracy", true);

	testModel.buildModel(ws);
//	ws.RunNetOnce(testInitNet);
//	ws.CreateNet(testTrainNet);

	std::cout << "Train model start ... ..." << std::endl;
	for(int i = 0; i < 100; i ++) {
		model.runModel(ws);
//		ws.RunNet(trainModel.name());
		const std::string accName = "accuracy";
		const auto blobUtil = ws.GetBlob(accName);
		Tensor* tensor = BlobGetMutableTensor(blobUtil, caffe2::DeviceType::CPU);
		const auto& data = tensor->data<float>();
		auto accu = data[0];
		std::cout << "Accuracy = " << accu << std::endl;

		testModel.runModel(ws);
//		ws.RunNet(testTrainNet.name());
		auto testBlob = ws.GetBlob("test_accuracy");
		Tensor* testTensor = BlobGetMutableTensor(testBlob, caffe2::DeviceType::CPU);
		const auto& testData = testTensor->data<float>();
		auto testAccu = testData[0];
		std::cout << "test_accuracy = " << testAccu << std::endl;
	}

//	for (int i = 0; i < 50; i ++) {
//		std::cout << "test " << std::endl;
//		ws.RunNet(testTrainNet.name());
//		auto blobUtil = ws.GetBlob("test_accuracy");
//		Tensor* tensor = BlobGetMutableTensor(blobUtil, caffe2::DeviceType::CPU);
//		const auto& data = tensor->data<float>();
//		auto accu = data[0];
//		std::cout << "Accuracy = " << accu << std::endl;
//	}
}

}

int main(int argc, char** argv) {
	google::InitGoogleLogging("Test");	//初始化
	FLAGS_log_dir = "/home/zf/workspaces/workspace_cpp/caffe2practice/build/log.txt";	//重定向日志输出到指定文件夹D://log下 我不需要日志输出 所以并没有指定
	FLAGS_stderrthreshold = google::INFO;	//在命令只打印google::ERROR级别以及该级别以上的日志信息


	caffe2::GlobalInit(&argc, &argv);
//	caffe2::run();
	caffe2::runWithTest();
	google::protobuf::ShutdownProtobufLibrary();

	return 0;
}
