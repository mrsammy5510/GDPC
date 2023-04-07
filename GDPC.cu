#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>    
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <iomanip> // Header file needed to use setprecision
#include <numeric> // for using iota
#include "./common/common.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <cuda_runtime.h>

using namespace std;

ifstream dataset;
ofstream ids_left_file("ids_left.txt");
ofstream ids_right_file("ids_right.txt");
ofstream ids_file("ids.txt");
ofstream cluster_result_file("cluster_result.txt");
ofstream dist_tmp_file("dist_tmp.txt");
ofstream VP_tree_file("VP_tree.txt");
ofstream leaf_file("leaf.txt");
ofstream tmp_file("tmp.txt");


typedef struct VP_node
{
    int vp = -1;
    float rad = -1;
}VP_node;

class leaf_node
{
    public:
        int id[32] = {};
        leaf_node()
        {
            memset(id, -1, 32 * sizeof(int));
        }
};


template<typename T> class d_vector
{
private:
    T* begin;
    T* end;

    int length;
    int capacity;
public:
    __device__ __host__ d_vector(): length(0), capacity(16)
    {
        begin = new T [capacity];
        end = begin - 1;
    }
    __device__ __host__ ~d_vector()
    {
        delete[] begin;
        begin = nullptr;
    }
    __device__ void expand()
    {
        capacity *= 2;
        T* tmp_begin = new T[capacity];
        memcpy(tmp_begin, begin, sizeof(T) * length);
        delete[] begin;
        begin = tmp_begin;
        end = begin + length - 1;
    }
    __device__ void push_back(T data)
    {
        if(length >= capacity)
        {
            expand();
        }

        end++;
        length++;

        *end = data;
    }
    __device__ T pop_back()
    {
        T end_element = *end;
        end--;
        length--;
        return end_element;
    }
    __device__ int size()
    {
        return length;
    }
    __device__ T pos(int position)
    {
        return *(begin + position);
    }
};


vector<vector<float>> readcsv(string filename)
{
    ifstream file(filename);

    vector<vector<float>> data; // 用二維vector來儲存表格數據

    if (file) {
        string line;
        while (getline(file, line)) { // 逐行讀取檔案
            vector<float> row;
            stringstream ss(line);
            string field;
            while (getline(ss, field, ',')) { // 以逗號分隔字段
                row.push_back(stof(field)); // 將字段轉換為float並加入vector中
            }
            data.push_back(row); // 將這一行數據加入到二維vector中
        }

    } else {
        cout << "Error: failed to open file" << endl;
    }

    return data;
}

inline float dist(vector<float> point1, vector<float> point2)
{
    float sum = 0;
    for(int i = 0;i<point1.size();i++)
    {
        sum+= pow(point1[i]-point2[i],2);
    }
    return sqrt(sum);
}


inline void normalize(vector<vector<float>> &datapoints)
{
    vector<float> attributes;
    for(int j = 0; j<datapoints[0].size(); j++)     //j defines which attribute is modifying now
    {
        for(int i = 0; i<datapoints.size(); i++)    //i is which point is modifying now
        {
            attributes.push_back(datapoints[i][j]);
        }
        auto max = *max_element(attributes.begin(), attributes.end());
        auto min = *min_element(attributes.begin(), attributes.end());
        for(int i = 0; i<datapoints.size(); i++)
        {
            datapoints[i][j] = (datapoints[i][j]-min)/(max-min);
        }
    }
}

inline void flatten(thrust::host_vector<float> &_1d_datapoints, vector<vector<float>> &datapoints)
{
    for(int row = 0; row < datapoints.size(); row++)
    {
        for(int col = 0; col< datapoints[0].size(); col++)
        {
            _1d_datapoints.push_back(datapoints[row][col]);
        }
    }
}

inline void decide_first_VP(vector<vector<float>> &datapoints, vector<VP_node> &VP, int data_size)
{
    vector<float> first_VP_cand;
    VP_node *first_VP = new VP_node;

    for(int i = 0; i < data_size; i++)
    {
        first_VP_cand.push_back(dist(datapoints[0], datapoints[i]));
    }
    auto max = max_element(first_VP_cand.begin(), first_VP_cand.end());
    first_VP->vp = max - first_VP_cand.begin();
    VP[0] = (*first_VP);
}



__device__ __host__ inline float dist(float* point1, float* point2, int dim)
{
    float sum = 0;
    for(int i = 0;i< dim;i++)
    {
        sum+= pow((point1[i]-point2[i]),2);
    }
    return sqrt(sum);
}

__global__ void get_dist(float* datapoints, int dim, int* ids, VP_node* VP, int VP_id, float* dist_tmp, int size)
{
    
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size)
    {
            dist_tmp[i] = dist(&datapoints[VP[VP_id].vp * dim], &datapoints[ids[i] * dim], dim);
    }
}


void quickSort(float* arr, int left, int right, int* ids) {
    int i = left, j = right;
    float pivot = arr[(left + right) / 2];

    /* partition */
    while (i <= j) {
        while (arr[i] < pivot)
            i++;
        while (arr[j] > pivot)
            j--;
        if (i <= j) {
            swap(arr[i],arr[j]);
            swap(ids[i],ids[j]);
            i++;
            j--;
        }
    };

    /* recursion */
    if (left < j)
        quickSort(arr, left, j, ids);
    if (i < right)
        quickSort(arr, i, right, ids);
}


void recur_build_VP_tree(int dim, int* ids, int ids_size, 
                        VP_node* VP, int VP_id, leaf_node* leaf, int height, int total_size, float* d_datapoints)
{
    int size = ids_size;    //Total points in this subtree
    float* dist_tmp = new float[size];
    int mid = size/2;       //The medium
    

    //For CUDA computing
    dim3 block (32);
    dim3 grid ((size + block.x - 1)/block.x);
    int* d_ids;
    VP_node* d_VP;
    float* d_dist_tmp;


    //------------------------------------------------------------------------------------------------

    if(size>32)
    {
        
        cudaMalloc((void** )&d_ids, size * sizeof(int));
        cudaMalloc((void** )&d_VP, (int)pow(2, height) * sizeof(VP_node));
        cudaMalloc((void** )&d_dist_tmp, size * sizeof(float));
        
        cudaMemcpy(d_ids, ids, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_VP, VP, (int)pow(2, height) * sizeof(VP_node), cudaMemcpyHostToDevice);
        
        
        get_dist<<<grid, block>>>(d_datapoints, dim, d_ids, d_VP, VP_id, d_dist_tmp, size);
        cudaDeviceSynchronize();


        thrust::sort_by_key(thrust::device, d_dist_tmp, d_dist_tmp + size, d_ids);
        cudaMemcpy(dist_tmp, d_dist_tmp, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(ids, d_ids, size * sizeof(int), cudaMemcpyDeviceToHost);
        
        /*
        cudaFree(d_ids);
        cudaFree(d_VP);
        cudaFree(d_dist_tmp);
        */
        
        VP[VP_id].rad = dist_tmp[mid];
        int* ids_left = (int* )malloc(sizeof(int) * mid);
        int* ids_right = (int* )malloc(sizeof(int) * (size - mid));

        memcpy(ids_left, ids, mid * sizeof(int));
        memcpy(ids_right, ids + mid, (size - mid) * sizeof(int));
        //Recursively building left subtree and right subtree
        if(mid>32)
        {
            VP[2*VP_id+1].vp = ids[mid-1];
            VP[2*VP_id+2].vp = ids[size-1];
        }
        

        recur_build_VP_tree(dim,  ids_left,        mid, VP, 2 * VP_id + 1, leaf, height, total_size, d_datapoints);
        recur_build_VP_tree(dim, ids_right, size - mid, VP, 2 * VP_id + 2, leaf, height, total_size, d_datapoints);
        //---------------------------------------------------------------------------------------------
        
    }
    else
    {
        memcpy(leaf[VP_id-(int)pow(2, height)+1].id, ids, size * sizeof(int)); 
    }
    

    
}
void initialize(float* datapoints, vector<vector<float>> datapoints_vector, int dim)
{
    for(int i = 0; i < datapoints_vector.size(); i++)
    {
        for(int j = 0; j < datapoints_vector[0].size(); j++)
        {
            datapoints[i * dim + j] = datapoints_vector[i][j];
        }
    }
}

inline void decide_first_VP(vector<vector<float>> &datapoints, VP_node* VP, int data_size)
{
    vector<float> first_VP_cand;
    VP_node *first_VP = new VP_node;

    for(int i = 0; i < data_size; i++)
    {
        first_VP_cand.push_back(dist(datapoints[0], datapoints[i]));
    }
    auto max = max_element(first_VP_cand.begin(), first_VP_cand.end());
    first_VP->vp = max - first_VP_cand.begin();
    VP[0] = (*first_VP);
}

void print_VP_tree(VP_node* VP, leaf_node* leaf, int VP_size, int leaf_size)
{
    for(int i = 0; i < VP_size; i++)
    {
        VP_tree_file<<"VP["<<i<<"]: "<<VP[i].vp<<"\t"<<"rad: "<<VP[i].rad<<endl;
    }
    for(int j = 0; j < leaf_size; j++)
    {
        leaf_file<<"leaf "<<j<<": ";
        int k = 0;
        while(leaf[j].id[k] != -1)
        {
            leaf_file<<leaf[j].id[k]<<" ";
            k++;
        }
        leaf_file<<endl;
    }
}

__global__ void GDPC_rho(float* d_datapoints, VP_node* d_VP, leaf_node* d_leaf, int VP_size, int dim, float dc, int* d_rho,
                        d_vector<int>* cover_leaves)
{   
    unsigned int pid = blockIdx.x * blockDim.x + threadIdx.x;

    cover_leaves[pid] = *(new d_vector<int>);

    d_vector<int> stack;
    stack.push_back(0);

    while(stack.size()!=0)
    {

        int vp_now = stack.pop_back();
        if(vp_now >= VP_size)
        {           
            int leaf_tmp = 0;
            while(d_leaf[vp_now - VP_size].id[leaf_tmp] != -1 && leaf_tmp <= 31)
            {
                cover_leaves[pid].push_back(d_leaf[vp_now - VP_size].id[leaf_tmp]);
                leaf_tmp++;
            }
        }
        else
        {
            if(dist(&d_datapoints[d_VP[vp_now].vp * dim], &d_datapoints[pid * dim], dim) - dc <= d_VP[vp_now].rad)
            {
                stack.push_back(2 * vp_now + 1);
            }
            if(dist(&d_datapoints[d_VP[vp_now].vp * dim], &d_datapoints[pid * dim], dim) + dc >= d_VP[vp_now].rad)
            {
                stack.push_back(2 * vp_now + 2);
            }
            
        }
    }
    
    for(int i = 0; i < cover_leaves[pid].size(); i++)
    {   
        if(dist(&d_datapoints[cover_leaves[pid].pos(i) * dim], &d_datapoints[pid * dim], dim) <= dc)
        {
            d_rho[pid]++;
        }
    }

    /*
    unsigned int pid = blockIdx.x * blockDim.x + threadIdx.x;
    d_vector<int> cover_leaves;
    d_vector<int> stack;
    stack.push_back(0);

    while(stack.size()!=0)
    {

        int vp_now = stack.pop_back();
        if(vp_now >= VP_size)
        {           
            int leaf_tmp = 0;
            while(d_leaf[vp_now - VP_size].id[leaf_tmp] != -1 && leaf_tmp <= 31)
            {
                cover_leaves.push_back(d_leaf[vp_now - VP_size].id[leaf_tmp]);
                leaf_tmp++;
            }
        }
        else
        {
            if(dist(&d_datapoints[d_VP[vp_now].vp * dim], &d_datapoints[pid * dim], dim) - dc <= d_VP[vp_now].rad)
            {
                stack.push_back(2 * vp_now + 1);
            }
            if(dist(&d_datapoints[d_VP[vp_now].vp * dim], &d_datapoints[pid * dim], dim) + dc >= d_VP[vp_now].rad)
            {
                stack.push_back(2 * vp_now + 2);
            }
            
        }
    }
    
    for(int i = 0; i < cover_leaves.size(); i++)
    {   
        if(dist(&d_datapoints[cover_leaves.pos(i) * dim], &d_datapoints[pid * dim], dim) <= dc)
        {
            d_rho[pid]++;
        }
    }
    */
}

__global__ void GDPC_result()
{

}

int main()
{

    vector<vector<float>> datapoints_vector;
    datapoints_vector = readcsv("./csv file/S2.csv");
    //datapoints_vector = readcsv("./csv file/3D_spatial_network.csv");
    //datapoints_vector = readcsv("./GDPC/csv file/S2.csv");
    //datapoints_vector = readcsv("./csv file/3D_spatial_network.csv");
    normalize(datapoints_vector);
    int data_size = datapoints_vector.size();
    int dim = datapoints_vector[0].size();
    
    float dc = 0.00969449058;
    
    
    //flatten the datapoints into 1D
    float* datapoints = new float[data_size * dim];
    initialize(datapoints, datapoints_vector, dim);
    //----------------------------------------------------------------------


    //For VP tree
    int h = ceil(log(data_size/32)/log(2));     //tree height (h-1)
    VP_node* VP = new VP_node[(int)pow(2, h) - 1];
    decide_first_VP(datapoints_vector, VP, data_size);
    int ids [data_size];
    for(int i = 0; i < data_size; i++){
        ids[i] = i;
    }
    leaf_node leaf[(int)pow(2, h)] = {};
    

    //-----------------------------------------------------------------
    cudaDeviceReset();  //Clean up the device
    cudaDeviceSynchronize();    // Warm up

    float* d_datapoints;
    cudaMalloc((void** )&d_datapoints, data_size * dim * sizeof(float));
    cudaMemcpy(d_datapoints, datapoints, data_size * dim * sizeof(float), cudaMemcpyHostToDevice);

    timespec t_building_VP_tree, tmp;

    clock_gettime(CLOCK_MONOTONIC, &t_building_VP_tree);
    
    recur_build_VP_tree(dim, ids, data_size, VP, 0, leaf, h, data_size, d_datapoints);

    clock_gettime(CLOCK_MONOTONIC, &tmp);
    cout << "time passed for building VP_tree parallelly is: " << 
    (tmp.tv_sec - t_building_VP_tree.tv_sec)*1000 + 
    (tmp.tv_nsec - t_building_VP_tree.tv_nsec)/1000000 << "ms" << endl;
    
    print_VP_tree(VP, leaf, (int)pow(2, h) - 1, (int)pow(2, h));

    dim3 block (32);
    dim3 grid ((data_size + block.x - 1)/block.x); 
    int* d_rho;
    VP_node* d_VP;
    leaf_node* d_leaf;
    cudaMalloc((void**)&d_rho, data_size * sizeof(int));
    cudaMalloc((void**)&d_VP, ((int)pow(2, h) - 1) * sizeof(VP_node));
    cudaMalloc((void**)&d_leaf, (int)pow(2, h) * sizeof(leaf_node));

    cudaMemset(d_rho, -1, data_size * sizeof(int));
    cudaMemcpy(d_VP, VP, ((int)pow(2, h) - 1) * sizeof(VP_node), cudaMemcpyHostToDevice);
    cudaMemcpy(d_leaf, leaf, (int)pow(2, h) * sizeof(leaf_node), cudaMemcpyHostToDevice);

    d_vector<int>* cover_leaves;
    cudaMalloc((void**)&cover_leaves, data_size * sizeof(d_vector<int>)); 

    GDPC_rho<<<grid, block>>>(d_datapoints, d_VP, d_leaf, (int)pow(2, h) - 1, dim, dc, d_rho, cover_leaves);
    cudaDeviceSynchronize();
    GDPC_result<<<grid, block>>();


    int* rho;
    rho = (int*)malloc(sizeof(int) * data_size);
    cudaMemcpy(rho, d_rho, sizeof(int) * data_size, cudaMemcpyDeviceToHost);

    
    for(int i = 0; i <data_size; i++)
    {
        //printf("rho %d = %d\t", i, rho[i]);
    }
    
     
    return 0;
}