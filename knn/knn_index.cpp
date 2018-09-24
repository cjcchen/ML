
#include "knn_index.h"

#include <queue>
#include <math.h>

using namespace std;


KNN_Index::KNN_Index(int size) :knn_(size), node_size_(size){
	index_nodes_.resize(node_size_);
}

void KNN_Index::build(const std::vector<Data> &key_data_list) {
	
	knn_.train(key_data_list);
	printf("train list %zu done\n",key_data_list.size());
	for(size_t i = 0; i < key_data_list.size();++i) {
		std::pair<int,double> idx_pair = knn_.find_nn_centroid(key_data_list[i]);
//		printf("data index %zd data %p find nn index %d, dis %lf\n",i,&key_data_list[i], idx_pair.first, idx_pair.second);
		index_nodes_[idx_pair.first].add(&(key_data_list[i]), idx_pair.second);
	}
}

void KNN_Index::print() {
	for(size_t i = 0; i < index_nodes_.size();++i) {
		const std::vector<const Data *> & data_list = index_nodes_[i].get_data();
		printf("node %zd\n",i);
		for(auto data : data_list) {
			printf("%p, data %p\n",data, data->datas_);
		}
	}
}

void KNN_Index::search(const Data &key, vector<DataInfo> *res_list, int top_k) {

	vector<NodeSearchInfo> res_idxs;
	for(int i = 0; i < node_size_; ++i) {
		double dis = knn_.get_centroid_dis(i, key);
//		printf("get node %d dis %lf\n",i,dis);
		res_idxs.push_back( NodeSearchInfo(i, dis, fabs(dis-index_nodes_[i].get_max_dis())) );
	}

	sort(res_idxs.begin(), res_idxs.end());

	int search_num=0;
	priority_queue<DataInfo> q;
	for(size_t i = 0; i < res_idxs.size();++i) {
		int node_idx = res_idxs[i].node_idx;
		//printf("search node %d\n",node_idx);

		if(q.size()>=top_k) {
			if(index_nodes_[node_idx].check(res_idxs[i].dis, q.top().dis_)) {
				continue;
			}
			/*
			if(res_idxs[i].dis/3 > q.top().dis_) {
				//printf("node index %d min dis %lf less than top dis %lf\n",node_idx, res_idxs[i].min_dis, q.top().dis_);
				//continue;
				//break;
			}
			*/
		}

		vector<DataInfo> idx_res_list;
		search_data(key, index_nodes_[node_idx], &idx_res_list, top_k);
		search_num++;

		for(size_t j = 0; j < idx_res_list.size();++j) {
			if(q.size() < top_k || idx_res_list[j].dis_ < q.top().dis_) {
				q.push(idx_res_list[j]);
			}
			if(q.size()>top_k) {
				q.pop();
			}
		}
	}
	//printf("search done, queue size %zu, search num %d\n",q.size(), search_num);
	while(!q.empty()) {
		const DataInfo &info = q.top();
		//printf("get res info data %p, dis %lf\n",info.data_, info.dis_);
		res_list->push_back(info);
		q.pop();
	}
}

void KNN_Index::search_data(const Data &key, const IndexNode &index_node, vector<DataInfo> *idx_res_list, int  top_k){
	const vector< const Data * > &data_list = index_node.get_data();
	priority_queue<DataInfo> q;
	for(size_t i = 0; i < data_list.size();++i) {
		double dis = KNN::distance(*data_list[i], key);	
		//printf("get key %p db data %p, dis %lf\n",&key, data_list[i], dis);
		DataInfo info(data_list[i], dis);
		q.push(info);
		if(q.size()>top_k) {
			q.pop();
		}
	}

	while(!q.empty()) {
		//printf("get search res %p dif %lf\n",q.top().data_, q.top().dis_);
		idx_res_list->push_back(q.top());
		q.pop();
	}
}

