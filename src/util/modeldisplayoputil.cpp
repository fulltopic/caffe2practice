/*
 * modeldisplayoputil.cpp
 *
 *  Created on: Mar 23, 2019
 *      Author: zf
 */

#include "util/modelutil.h"

namespace caffe2 {
	OperatorDef* ModelUtil::AddPrintOp(const std::string& param, bool toFile, ModelType modelType) {
	RESP_OP_2(AddPrintOp, param, toFile)
}
}

