#pragma once

#include <vector>
#define DataType double

class Data{ 
	public:
		Data(const std::vector<DataType> *data, const std::pair<int,int> &interval);
	public:
		void get_data(const DataType **data_ptr, int *data_len)const;
		void get_data_len(int *data_len)const;
	public:
		const std::vector<DataType> *datas_;
		const std::pair<int,int> interval_;
};

class CentroidData{
	
	public:
		CentroidData(int data_len);
		CentroidData(const CentroidData &centroid);
		~CentroidData();

	public:
		void init();
		void clear();
		void add(const DataType * ptr, int data_len);
		void done();
		const DataType * get_data()const;
		int get_data_len()const;

	private:
		DataType* centroid_data_;
		const int centroid_data_len_;
		int data_collected_num_;
};

class KNN {
	public:
		KNN(int k_centroids);
		void train(const std::vector<Data> &data_list);
		std::pair<int,double> find_nn_centroid(const Data &data);
		double get_centroid_dis(int centroid_idx, const Data &data);

		static double distance(const DataType *centroid_data, const DataType *db_data, const int data_len);
		static double distance(const Data &data1, const Data &data2);

	private:
		void init_centroids(const std::vector<Data> &data_list);
		void centroid_clear();
		void update_centroid(const Data &data,CentroidData &centroid);
		void centroid_done();
		
	private:
		int k_centroids_;
		std::vector<CentroidData> centroids_;
};


