#include <caffe2/core/db.h>
//#include <caffe2/proto/caffe2.ph.h>
#include <c10/util/Registry.h>
#include <iostream>
#include <string>

const std::string trainDbPath("/home/zf/workspaces/workspace_cpp/testcaffe2/build/res/lmdb/mnist-train-nchw-lmdb");
const std::string levelDbPath("/home/zf/workspaces/workspace_cpp/testcaffe2/build/res/leveldb/mnist-train-nchw-leveldb");
const std::string miniDbPath("/home/zf/workspaces/workspace_cpp/testcaffe2/build/res/mnist-test-nhwc-minidb");
const std::string lmDbPath("/home/zf/workspaces/workspace_cpp/testcaffe2/build/res/lmdb/mnist-train-nchw-lmdb");


int main(int argc, char** argv) {
//	auto e = caffe2::db::DBExists("lmdb", trainDbPath);
//	std::cout << e << "\n";

//	caffe2::db::DBReader a("lmdb", trainDbPath);
//	std::cout << a << "\n";

//	auto b = caffe2::db::CreateDB("leveldb", levelDbPath, caffe2::db::READ);
//	if (!(b)) {
//		std::cout << "not created" << "\n";
//	} else {
//		std::cout << "created" << "\n";
//	}

	caffe2::db::DBReader reader("lmdb", lmDbPath);
//	reader.SeekToFirst();

	std::string key;
	std::string value;
	for (int i = 0; i < 2; i ++) {
		reader.Read(&key, &value);
		std::cout << key << std::endl;
		std::cout << value << std::endl;

//      TensorProtos protos;

	}
//	reader.Read(&key, &value);
//	std::cout << key << std::endl;

//	caff2::db::Caffe2DBRegistry().Has("lmdb");
//	std::cout << "LEVELDB" << std::endl;
//	auto f = caffe2::db::DBExists("leveldb", levelDbPath);
}
