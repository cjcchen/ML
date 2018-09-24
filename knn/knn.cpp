#include "knn.h"
#include <assert.h>
#include <math.h>

#define N_EPOCH 200
#define THREADS_HOLD 0.0001

using namespace std;

Data::Data(const vector<DataType> *data, const pair<int,int> &interval):datas_(data), interval_(interval){
}

void Data::get_data(const DataType **data_ptr, int *data_len) const{
	*data_ptr = &((*datas_)[interval_.first]);
	*data_len = interval_.second - interval_.first;
}

void Data::get_data_len(int *data_len) const{
	*data_len = interval_.second - interval_.first;
}

CentroidData::CentroidData(int data_len) :centroid_data_len_(data_len){
	centroid_data_=NULL;
	//printf("new %p\n",this);
}

CentroidData::CentroidData(const CentroidData &centroid) :centroid_data_len_(centroid.centroid_data_len_){
	this->centroid_data_ = NULL;
	if(centroid.centroid_data_!=NULL) {
		init();
		memcpy(this->centroid_data_, centroid.centroid_data_, centroid_data_len_*sizeof(DataType));
	}
	//printf("copy new %p, data %p\n",this, this->centroid_data_);
}

CentroidData::~CentroidData() {
	//printf("%p free %p\n",this, centroid_data_);
	free(centroid_data_);
}

void CentroidData::init() {
	centroid_data_ = (DataType*)malloc(centroid_data_len_*sizeof(DataType));
	//printf("%p alloc %p\n",this, centroid_data_);
	clear();
}

void CentroidData::clear() {
	memset(centroid_data_, 0, centroid_data_len_ * sizeof(DataType));
	data_collected_num_ = 0;
}

void CentroidData::add(const DataType * ptr, int data_len) {
	if( data_collected_num_<0) {
		printf("please clear data first after calling done\n");
		return;
	}
	if(centroid_data_len_ != data_len) {
		printf("data error, centroid data len %d, update data len %d not match\n", centroid_data_len_, data_len);
		assert(centroid_data_len_ == data_len);
	}

	//printf("add data to centroid %p:", centroid_data_);
	for(int i = 0; i < data_len;++i) {
		centroid_data_[i] += ptr[i];
	//	printf("%lf ",centroid_data_[i]);
	}
	//printf("\n");
	data_collected_num_++;
}

void CentroidData::done() {
	if( data_collected_num_==-1) {
		printf("done func has been called\n");
		return;
	}
	
	if(data_collected_num_>0) {
		for(int i = 0; i < centroid_data_len_;++i) {
			centroid_data_[i] /= data_collected_num_;
		}
	}
	else {
		printf("centroid no data\n");
	}
	data_collected_num_=-1;
}

const DataType * CentroidData::get_data() const{
	return centroid_data_;
}

int CentroidData::get_data_len() const{
	return centroid_data_len_;
}

KNN :: KNN(int k_centroids) : k_centroids_(k_centroids){
}
	
void KNN::init_centroids(const vector<Data> &data_list) {
	int centroid_num = k_centroids_< data_list.size()?k_centroids_:data_list.size();
	printf("k num %d\n",centroid_num);
	int data_len = 0;
	data_list[0].get_data_len(&data_len);
	printf("data len %d\n",data_len);

	for(int i = 0; i < centroid_num; ++i) {
		centroids_.push_back(CentroidData(data_len));
		printf("cen i %d %p\n",i, &centroids_[i]);
	}

	for(int i = 0; i < centroid_num;++i) {
		printf("cen i %d %p\n",i, &centroids_[i]);
		centroids_[i].init();
	}

	centroid_clear();
	for(int i = 0; i < centroid_num; ++i) {
		update_centroid(data_list[i],centroids_[i]);
	}
	centroid_done();
	
	/*
	for(int i = 0; i < centroids_.size();++i) {
		const DataType * data = centroids_[i].get_data();
		printf("centroid %d:",i);
		for(int j = 0; j < data_len;++j) {
			printf("%lf ", data[j]);
		}
		printf("\n");
	}
	*/
}

void KNN::centroid_clear() {
	if(centroids_.size() != k_centroids_) {
		printf("centroid len error, centroid len %zu, k num %d\n",centroids_.size(), k_centroids_);
		assert( centroids_.size() == k_centroids_);
	}

	for(int i = 0; i < k_centroids_;++i) {
		centroids_[i].clear();
	}
}

void KNN::update_centroid(const Data &data,CentroidData &centroid) {
	const DataType * data_ptr = NULL;
	int data_len = 0;
	data.get_data(&data_ptr, &data_len);
	centroid.add(data_ptr, data_len);
}

void KNN::centroid_done() {
	for(int i = 0; i < k_centroids_;++i) {
		centroids_[i].done();
	}
}

void KNN::train(const vector<Data> &data_list) {
	init_centroids(data_list);	
	if(data_list.size()<k_centroids_) {
		printf("data len %zu less than centroids num %d\n",data_list.size(), k_centroids_);
		return;
	}
	
	double old_dis = 0;
	for(int epoch = 0; epoch < N_EPOCH; epoch++) {
		double total_dis = 0;
		vector<pair<int,double>> centroid_ids;
		for(int i = 0; i < data_list.size();++i) {
			centroid_ids.push_back(find_nn_centroid(data_list[i]));
		}
		centroid_clear();
		for(int i = 0; i < centroid_ids.size();++i) {
			update_centroid(data_list[i],centroids_[centroid_ids[i].first]);
			total_dis += centroid_ids[i].second;
		}
		centroid_done();
		total_dis/=centroid_ids.size();
		printf("epoch %d, dis %lf\n",epoch, total_dis);
		if (fabs(total_dis-old_dis) < THREADS_HOLD) {
			break;
		}
		old_dis = total_dis;
	}
}

pair<int,double> KNN::find_nn_centroid(const Data &data) {
	const DataType * data_ptr = NULL;
	int data_len = 0;
	data.get_data(&data_ptr, &data_len);

	if(data_len != centroids_[0].get_data_len()) {
		printf("data len error, data len %d centrid data len %d\n",data_len, centroids_[0].get_data_len());
		assert(data_len==centroids_[0].get_data_len());
	}

	int min_dist_idx = -1;
	double min_dist = 0;
	
	/*
	for(int i = 0; i < centroids_.size();++i) {
		const DataType * data = centroids_[i].get_data();
		printf("centroid %d:",i);
		for(int j = 0; j < min(10,data_len);++j) {
			printf("%lf ", data[j]);
		}
		printf("\n");
	}
	*/

	for(int i = 0; i < centroids_.size();++i) {
		double dist = distance(centroids_[i].get_data(), data_ptr, data_len);
		if(min_dist_idx==-1 || dist < min_dist) {
			min_dist_idx = i;
			min_dist = dist;
		}
	}
	/*
	printf("data:");
	for(int i = 0; i < data_len;++i) {
		printf("%lf ",data_ptr[i]);
	}
	printf("\n");
	printf("find min centroid %d, dist %lf\n",min_dist_idx, min_dist);
	*/
	return make_pair(min_dist_idx, min_dist);
}

double KNN::distance(const DataType *centroid_data, const DataType *db_data, const int data_len) {
	double sum = 0;
	for(int i = 0; i < data_len; ++i) {
		double x = centroid_data[i] - db_data[i];
		sum += x*x;
	}
	return sum;
}

double KNN::distance(const Data &data1, const Data &data2) {
	const DataType * data_ptr1 = NULL;
	int data_len1 = 0;
	data1.get_data(&data_ptr1, &data_len1);

	const DataType * data_ptr2 = NULL;
	int data_len2 = 0;
	data2.get_data(&data_ptr2, &data_len2);


	return distance(data_ptr1, data_ptr2, data_len1);
}

double KNN::get_centroid_dis(int centroid_idx, const Data &data) {
	const DataType * data_ptr = NULL;
	int data_len = 0;
	data.get_data(&data_ptr, &data_len);
	return distance(centroids_[centroid_idx].get_data(), data_ptr, data_len);
}

