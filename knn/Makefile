


all: knn_main

knn:knn.cpp
	g++ -g knn.cpp -o knn

knn_main:knn.cpp knn_main.cpp knn_index.cpp
	g++ -g $^ -o $@ -std=c++11

