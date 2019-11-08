#include "util/modelutil.h"
#include <sstream>

namespace caffe2 {
//	ModelUtil::ModelUtil(NetDef& iModel, NetDef& tModel, std::string mName): initModel(iModel), trainModel(tModel), modelName(mName){
//
//	}

	ModelUtil::ModelUtil(std::string mName): initModel(mName + "_init"), trainModel(mName + "_train"), modelName(mName) {

	}

	ModelUtil::ModelUtil(const ModelUtil& from, const std::string mName):
			initModel(from.initModel, mName + "_init"), trainModel(from.trainModel, mName + "_train") {
	}

	ModelUtil::ModelUtil(NetUtilNN &init_net, NetUtilNN &predict_net,
	                     const std::string &name)
	    : modelName(name), initModel(init_net), trainModel(predict_net) {
	}

	ModelUtil::ModelUtil(NetDef &init_net, NetDef& predict_net, std::string name)
		:modelName(name), initModel(init_net, name + "_init"), trainModel(predict_net, name + "_train"){

	}

	ModelUtil::~ModelUtil() {}

	void ModelUtil::buildModel(Workspace& ws) {
		ws.RunNetOnce(initModel.net);
//		ws.RunNetOnce(initModel.net);
		ws.CreateNet(trainModel.net);
	}

	void ModelUtil::runModel(Workspace& ws) {
		ws.RunNet(trainModel.net.name());
	}

	std::string ModelUtil::Proto() {
		std::stringstream s;

		std::string trainProto = this->trainModel.Proto();
		std::string initProto = this->initModel.Proto();

		s << this->modelName << ": " << std::endl;
		s << "TrainProto " << trainProto << std::endl;
		s << "InitPtoto " << initProto << std::endl;

		return s.str();
	}

}
