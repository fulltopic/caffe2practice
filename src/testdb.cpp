#include <caffe2/core/db.h>
#include <c10/util/Registry.h>
#include <iostream>
#include <string>

const std::string lmDbPath("../res/lmdb/mnist-train-nchw-lmdb");


int main(int argc, char** argv) {

	caffe2::db::DBReader reader("lmdb", lmDbPath);
//	reader.SeekToFirst();

	std::string key;
	std::string value;
	for (int i = 0; i < 2; i ++) {
		reader.Read(&key, &value);
		std::cout << key << std::endl;
		std::cout << value << std::endl;
	}
//	reader.Read(&key, &value);
//	std::cout << key << std::endl;

//	caff2::db::Caffe2DBRegistry().Has("lmdb");
//	std::cout << "LEVELDB" << std::endl;
//	auto f = caffe2::db::DBExists("leveldb", levelDbPath);
}
