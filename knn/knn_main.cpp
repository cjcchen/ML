#include "knn.h"
#include "knn_index.h"
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <map>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <string.h>
#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>


using namespace std;

vector<string> get_dirs( const string &dir_name ) {
	vector<string> dir_list;
	struct dirent * filename;    // return value for readdir()
	DIR * dir = opendir( dir_name.c_str() );
	if( NULL == dir ) {
		printf("open dir fail\n");
		return dir_list;
	}
	
	/* read all the files in the dir ~ */
	while( ( filename = readdir(dir) ) != NULL ) {
		// get rid of "." and ".."
		if( strcmp( filename->d_name , "." ) == 0 || 
			strcmp( filename->d_name , "..") == 0    )
			continue;
		
		char full_name[128];
		sprintf(full_name, "%s/%s", dir_name.c_str(), filename->d_name);
		struct stat s;
		lstat( full_name, &s );
//		printf("get filename %s\n",full_name);
		if( !S_ISDIR( s.st_mode ) ) {
			continue;
		}
		dir_list.push_back(full_name);
	//	if(dir_list.size()>100)
	//		break;
	}

	closedir(dir);
	return dir_list;
}

vector<string> get_files( const string &dir_name ) {
	vector<string> files_list;
	struct dirent * filename;    // return value for readdir()
	DIR * dir = opendir( dir_name.c_str() );
	if( NULL == dir ) {
		printf("dir is invalid %s, error %s\n",dir_name.c_str(), strerror(errno));
		return files_list;
	}
	//printf("read dir %p dir name %s\n",dir,dir_name.c_str());	
	/* read all the files in the dir ~ */
	while( ( filename = readdir(dir) ) != NULL ) {
		// get rid of "." and ".."
		if( strcmp( filename->d_name , "." ) == 0 || 
			strcmp( filename->d_name , "..") == 0    )
			continue;
		
		char full_name[1024];
		sprintf(full_name, "%s/%s", dir_name.c_str(), filename->d_name);
		struct stat s;
		lstat( full_name, &s );
		if( S_ISDIR( s.st_mode ) ) {
			continue;
		}
		
		//printf("get filename %s\n",full_name);
		files_list.push_back(full_name);
	}

	closedir(dir);
	return files_list;
}

void read_lfw(const char *dir_path, vector<string> &file_list, vector<int> &label_list) {
	vector<string> dir_list = get_dirs(dir_path);	
//	printf("get dir list size %zu\n",dir_list.size());
	for(int i = 0; i < dir_list.size();++i) {
		vector<string> files = get_files(dir_list[i]);
		assert(files.size()>0);
		for(int j = 0; j < files.size();++j) {
			file_list.push_back(files[j]);
			label_list.push_back(i);
//			printf("get file %s label %d\n",files[j].c_str(), i);
		}
	}
}

void get_data_set(const vector<int> &label_list, vector<int> *train_idx, vector<int> *test_idx) {
	map<int, vector<int> > label_map;
	for(int i = 0; i < label_list.size();++i) {
		label_map[ label_list[i] ].push_back(i);
	}
	for(auto &it : label_map) {
		if(it.second.size()>1) {
			int len = it.second.size()*2/3;
			int t_len = it.second.size()-len;
			for(int j = 0; j < len; ++j) {
				train_idx->push_back(it.second[j]);
			}
			for(int j = len; j < it.second.size();++j) {
				test_idx->push_back(it.second[j]);
			}
		}
	}
}

void read_file(const vector<string> &file_list, vector<vector<DataType>> &file_data) {

	for(int i = 0; i < file_list.size();++i) {
		int fd = open(file_list[i].c_str(), O_RDONLY);
		if(fd<0) {
			printf("file %s open fail, error %s\n",file_list[i].c_str(), strerror(errno));
			continue;
		}
		vector<DataType> data;
		while(1) {
			float v;
			int ret = read(fd, &v, sizeof(v));
		//	printf("read file %s ret %d\n",file_list[i].c_str(), ret);
			if(ret<=0) {
				break;
			}
			data.push_back(v);
		}
		//printf("read data size %zu\n",data.size());
		file_data.push_back(data);
		close(fd);
	}
}

int find_label(const Data &key, const vector<Data> &data_list, const vector<int> &label_list) {
	for(size_t i = 0; i < data_list.size();++i) {
		double dis = KNN::distance(key, data_list[i]);
		if( fabs(dis)<1e-12) {
			return i;
		}
	}
	assert(1==0);
}

void get_lfw_data(const string &path_file_name, const string &data_path,
		vector<string> &file_list, vector<vector<DataType>>&file_data_list, vector<int> &label_list) {
	FILE * fp = fopen(path_file_name.c_str(),"r");
	assert(fp);
	char path[1024];
	int label;
	while(fscanf(fp, "%s %d", path,&label)>0) {
		printf("read %s label %d\n",path, label);
		char full_path[1024];
		sprintf(full_path,"%s/%s",data_path.c_str(), path);
		file_list.push_back(full_path);
		label_list.push_back(label);
	}
	read_file(file_list, file_data_list);
}

void create_data(const string &data_file_path, 
	vector<Data> &train_list, vector<Data> &test_list,
	vector<int> &train_label_list, vector<int> &test_label_list,
	vector<string> &train_img_list, vector<string> &test_img_list){

	int test_type = 1;
	vector<string> file_list;
	vector<int> label_list;
	vector<vector<DataType>> feature_list;

	read_lfw(data_file_path.c_str(), file_list, label_list);
	read_file(file_list, feature_list);
	printf("get file list %zu, label list %zu, data list %zu\n",file_list.size(), label_list.size(), feature_list.size());
		
	if(test_type==0) {
		for(int i = 0; i < file_list.size(); ++i) {
			train_list.push_back( Data(&feature_list[i],make_pair(0,feature_list[0].size())) );
			test_list.push_back( Data(&feature_list[i],make_pair(0,feature_list[0].size())) );

			train_label_list.push_back(label_list[i]);
			test_label_list.push_back(label_list[i]);

			train_img_list.push_back(file_list[i]);
			test_img_list.push_back(file_list[i]);
		}
	}
	else {
		vector<int> train_idx,test_idx;
		get_data_set(label_list, &train_idx, &test_idx);
		for(int i = 0; i < train_idx.size();++i) {
			int index = train_idx[i];
			train_list.push_back( Data(&feature_list[index],make_pair(0,feature_list[0].size())) );

			train_label_list.push_back(label_list[index]);
			train_img_list.push_back(file_list[index]);

			printf("i %d train data %s %p\n",i, file_list[index].c_str(), &feature_list[index]);
			for(int j = 0; j < 10; ++j) {
				printf("%lf ",feature_list[index][j]);
			}
			printf("\n");
		}

		for(int i = 0; i < test_idx.size();++i) {
			int index = test_idx[i];
			test_list.push_back( Data(&feature_list[index],make_pair(0,feature_list[0].size())) );
			test_label_list.push_back(label_list[index]);
			test_img_list.push_back(file_list[index]);
		}
	}

}

int get_time() {
	struct timeval tv;
	gettimeofday(&tv,NULL);
	return tv.tv_sec;
}

int main(int argc, char ** argv) {

	string data_path = argv[1];

	vector<vector<DataType>> train_org_data_list, test_org_data_list;
	vector<Data> train_list, test_list;
	vector<int> train_label_list, test_label_list;
	vector<string> train_img_list, test_img_list;
	
	get_lfw_data("lfw_db_index", data_path,
		train_img_list, train_org_data_list, train_label_list);

	get_lfw_data("lfw_query_index", data_path,
		test_img_list, test_org_data_list, test_label_list);

	for(const auto &data : train_org_data_list) {
		train_list.push_back( Data(&data,make_pair(0,data.size())) );
	}

	for(const auto &data : test_org_data_list) {
		test_list.push_back( Data(&data,make_pair(0,data.size())) );
	}

	printf("train data size %zu, test data size %zu\n",train_list.size(), test_list.size());

	/*
	for(int i = 0; i < file_list.size(); ++i) {
		printf("train data list %d %p, data %p\n",i,&(train_list[i]), train_list[i].datas_);
	}
	printf("train list size %zu\n",train_list.size());
	*/

	printf("train size %zu test size %zu\n",train_list.size(), test_list.size());
	KNN_Index index(256);
	index.build(train_list);
	//index.print();

	printf("build done\n");
	int start_time = get_time();
	int acc = 0;
	for(int i = 0; i < test_list.size();++i) {
		vector<DataInfo> res;
	//	printf("search %d\n",i);	
		index.search(test_list[i], &res, 5);
		int search_label_idx = find_label(*res[res.size()-1].data_, train_list, train_label_list);
		if(test_label_list[i] != train_label_list[search_label_idx]) {
			/*
			for(auto data : res) {
				printf("get index %d res %p, dis %.07lf\n",i, data.data_, data.dis_);
			}
			printf("query %d label %d, search label %d\n",i,test_label_list[i], train_label_list[search_label_idx]);
			printf("img: %s %s\n",test_img_list[i].c_str(), train_img_list[search_label_idx].c_str());
			*/
		}
		else {
			acc++;
		}
	}
	int end_time = get_time();
	printf("acc %lf\n",(double)acc/test_list.size()*100);
	printf("run %d\n",end_time-start_time);
	return 0;
}

