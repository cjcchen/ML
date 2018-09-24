#pragma once

#include "knn.h"
#include <set>

class IndexNode {
public:
	IndexNode() {
		max_dis_ = 0;
		data_list_.clear();
	}

public:
	void add(const Data *data, double dis) {
		data_list_.push_back(data);
		max_dis_ = std::max(max_dis_, dis);
		data_dis_list_.insert(dis);
		//printf("node add index %p dis %lf node max dis %lf\n",data, dis, max_dis_);
	}

	bool check(double dis, double thresh_hold) {
		auto it = data_dis_list_.lower_bound(dis);
		double check_dis = 10000;
		if(it!=data_dis_list_.end()) {
			check_dis = *it-dis;
		}
		if(it!=data_dis_list_.begin()) {
			it--;
			check_dis = std::min(check_dis, dis-*it);
		}
		return check_dis > thresh_hold;
	}

	const std::vector<const Data *> &get_data()const {
		return data_list_;
	}

	double get_max_dis()const{
		return max_dis_;
	}
private:
	std::vector<const Data*> data_list_;
	std::set<double> data_dis_list_;
	double max_dis_;
};

struct NodeSearchInfo{
	int node_idx;
	double dis;
	double min_dis;

	NodeSearchInfo(){}
	NodeSearchInfo(int idx, double dis, double min_dis): node_idx(idx),dis(dis), min_dis(min_dis){}
	bool operator<(const NodeSearchInfo &a) const{
		return this->dis < a.dis;
	}
	bool operator()(const NodeSearchInfo &a) const{
		return this->dis > a.dis;
	}
};

struct DataInfo {
	const Data *data_;
	double dis_;
	public:
		DataInfo(const Data *data, double dis): data_(data), dis_(dis){}

		bool operator<(const DataInfo &a) const{
			return this->dis_ < a.dis_;
		}
		bool operator()(const DataInfo &a) const{
			return this->dis_ > a.dis_;
		}
};

class KNN_Index {
	public:
		KNN_Index(int size);
	public:
		void build(const std::vector<Data> &key_data_list);

		void search(const Data &key, std::vector<DataInfo> *res_list, int top_k);

		void print();
	private:
		void search_data(const Data &key, const IndexNode &index_node, std::vector<DataInfo> *idx_res_list, int top_k);

	private:
		int node_size_;
		KNN knn_;
		std::vector<IndexNode>index_nodes_;
};

