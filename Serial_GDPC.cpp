#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>    
#include <math.h>
#include <sstream>
#include <iomanip> // Header file needed to use setprecision

using namespace std;
ifstream dataset;
ofstream ids_left_file("ids_left.txt");
ofstream ids_right_file("ids_right.txt");
ofstream cluster_result_file("cluster_result.txt");
ofstream VP_tree_file("VP_tree.txt");
ofstream leaf_file("leaf.txt");

typedef struct VP_node
{
    int vp = -1;
    float rad = -1;
}VP_node;

typedef struct leaf_node
{
    vector<int> id;
}leaf_node;

template <typename T>

class Stack {
public:
    void push(T val) {
        stack_.push_back(val);
    }

    T pop() {
        if (stack_.empty()) {
            std::cerr << "Stack is empty" << std::endl;
            return T();
        }
        int data = stack_.back();
        stack_.pop_back();
        return data;
    }

    bool empty() {
        return stack_.empty();
    }

private:
    std::vector<T> stack_;
};



void normalize(vector<vector<float>> &datapoints)
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

void normalize(vector<float> &arr)
{
    auto max = *max_element(arr.begin(), arr.end());
    auto min = *min_element(arr.begin(), arr.end());
    for(int i = 0; i<arr.size(); i++)
    {
        arr[i] = (arr[i]-min)/(max-min);
    }    
}


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


inline void swap(int & a,int & b) {
    int temp = a;
    a = b;
    b = temp;
}

inline void swap(float & a,float & b) {
    float temp = a;
    a = b;
    b = temp;
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


void quickSort(vector<float> &arr, int left, int right, vector<int> &ids) {
    int i = left, j = right;
    float tmp;
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

void quickSort(vector<float> &arr, int left, int right) {
    int i = left, j = right;
    float tmp;
    float pivot = arr[(left + right) / 2];

    /* partition */
    while (i <= j) {
        while (arr[i] < pivot)
            i++;
        while (arr[j] > pivot)
            j--;
        if (i <= j) {
            swap(arr[i],arr[j]);
            i++;
            j--;
        }
    };

    /* recursion */
    if (left < j)
        quickSort(arr, left, j);
    if (i < right)
        quickSort(arr, i, right);
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

inline void initialize_ids(vector<int> &ids, int data_size)
{
    for(int i = 0; i<data_size; i++)
    {
        ids.push_back(i);
    }
}

void recur_build_VP_tree(vector<vector<float>> &datapoints, vector<int> ids, vector<VP_node> &VP, int VP_id, vector<leaf_node> &leaf, int height)
{
    int size = ids.size();  //Total points in this subtree
    int mid = size/2;       //The medium
    int tmp;
    vector<float> dist_tmp;
    if(size>32)
    {
        
        
        //Calculate the distance between vp and the points in this subtree
        for(int i = 0; i<ids.size();i++)
        {
            dist_tmp.push_back(dist(datapoints[VP[VP_id].vp], datapoints[ids[i]]));
        }
        //---------------------------------------------------------------------------------------------

        quickSort(dist_tmp, 0, dist_tmp.size()-1, ids);
        VP[VP_id].rad = dist_tmp[mid];
        

        //Divide into left half and right half
        vector<int> ids_left;
        vector<int> ids_right;
        
        ids_left.assign(ids.begin(), ids.begin()+mid);    
        /*The range used is [first,last), 
        which includes all the elements between first and last, 
        including the element pointed by first but "not" the element pointed by last.*/
        ids_right.assign(ids.begin()+mid, ids.end());


        /*
        FILE* fp1;
        FILE* fp2;

        fp1 = fopen("ids_left.txt","w");
        fp2 = fopen("ids_right.txt","w");
        fclose(fp1);
        fclose(fp2);
        for(int i = 0; i<ids_left.size(); i++)
        {
            ids_left_file<<i<<". "<<ids_left[i]<<endl;
        }
        for(int i = 0; i<ids_right.size(); i++)
        {
            ids_right_file<<i<<". "<<ids_right[i]<<endl;
        }
        */
        
        /*
        cout<<"ids_left size: "<<ids_left.size()<<endl;
        cout<<"ids_right size: "<<ids_right.size()<<endl;
        */
        
        /*
        for(int i = 0; i<mid-1 ;i++)
        {
            ids_left.push_back(ids[i]);
            ids_right.push_back(ids[i+mid]);
        }*/
        //---------------------------------------------------------------------------------------------
        //Recursively building left subtree and right subtree
        if(ids_left.size()>32)
            VP[2*VP_id+1].vp = ids[mid-1];
        //cout<<"2*VP_id-1: "<<2*VP_id+1<<endl;
        //cout<<"ids_left.size: "<<ids_left.size()<<endl;
        recur_build_VP_tree(datapoints, ids_left , VP, 2*VP_id+1, leaf, height);

        if(ids_right.size()>32)
            VP[2*VP_id+2].vp= ids[size-1];
        //cout<<"2*VP_id-2: "<<2*VP_id+1<<endl;
        //cout<<"ids_right.size: "<<ids_right.size()<<endl;
        recur_build_VP_tree(datapoints, ids_right, VP, 2*VP_id+2, leaf, height);

       

        
        //---------------------------------------------------------------------------------------------
    }
    else
    {
        
        //Put the ids in this leaf in leaf array, the position formula is the same with the paper
        leaf[VP_id-pow(2, height)+1].id.assign(ids.begin(), ids.end());
        //cout<<"VP_id-pow(2, height)+1 = "<<VP_id-pow(2, height)+1<<endl;
        //---------------------------------------------------------------------------------------------
    }

    return;
}


void serial_GDPC(vector<vector<float>> &datapoints, vector<VP_node> &VP, vector<leaf_node> &leaf, float dc, vector<int> &cluster_result, int VP_size)
{
    const int data_size = datapoints.size();
    vector<vector<int>> cover_leaves(data_size);
    vector<vector<float>> cover_leaves_rho(data_size);


    vector<float> rho(data_size,-1);
    vector<float> dep_dist(data_size);
    vector<int> dep_neighbor(data_size);
    vector<float> gamma(data_size);
    Stack<int> stack;

    int n = VP_size;      //number of internal nodes
    
    //Counting the number of VP(internal nodes)

    //---------------------------------------------------------------------------------------------

    //Deciding cover leaves(dc range query candidate) and counting rho by it
    for(int pid = 0; pid<data_size; pid++)
    {
        stack.push(0);
        while(!stack.empty())
        {
            int vp_now = stack.pop();
            if(vp_now>=n)
            {
                cover_leaves[pid].insert(cover_leaves[pid].begin(), leaf[vp_now-n].id.begin(), leaf[vp_now-n].id.end());
            }
            //If the distance between point i and the vantage point now - dc is less then vp radius, we need to search the left subtree
            else
            {
                //cout<<"dist1 = "<<dist(datapoints[VP[vp_now].vp],datapoints[pid])-dc<<"<="<<VP[vp_now].rad<<endl;
                if(dist(datapoints[VP[vp_now].vp],datapoints[pid])-dc <= VP[vp_now].rad)
                {
                    stack.push(2*vp_now+1);
                }
                //cout<<"dist2 = "<<dist(datapoints[VP[vp_now].vp],datapoints[pid])+dc<<">="<<VP[vp_now].rad<<endl;
                if(dist(datapoints[VP[vp_now].vp],datapoints[pid])+dc >= VP[vp_now].rad)
                {
                    stack.push(2*vp_now+2);
                }
            }
        }

        for(int j = 0; j<cover_leaves[pid].size(); j++)
        {
            if(dist(datapoints[cover_leaves[pid][j]],datapoints[pid])<=dc)
            {
                rho[pid]++;
            }
        }
        
    }
    //---------------------------------------------------------------------------------------------

    //normalize(rho);

    //Assigning the corresponding rho value of points in the cover leaves
    for(int pid = 0; pid< data_size; pid++)
    {
        for(int j = 0; j<cover_leaves[pid].size(); j++)
        {
            cover_leaves_rho[pid].push_back(rho[cover_leaves[pid][j]]);
        }
    }
    //---------------------------------------------------------------------------------------------

    //Calculating the dependent distance and dependent neighbor

    /*A big problem here is that, for point p, it's nearest point do not always in it's cover leaves. Moreover, DPC
    requires nearest "denser" point, make some case can't find the true dependent neighbor without considering all points.
    Since it is hard to distinguish these points, render this VP tree would cause some wrong calculation.*/

    vector<int> dep_neighbor_cand;
    vector<float> dep_neighbor_cand_dist;
    
    for(int pid = 0; pid<data_size; pid++)
    {
        //Finding the maximum rho in cover leaves, can't use max_element since it could be multiple max rho in leaves
        float rho_pid = rho[pid];       //The rho value of the point we checking now
        int max = -1;
        for(int i = 0; i < cover_leaves_rho[pid].size(); i++)
        {
            if(cover_leaves_rho[pid][i] > rho_pid)
            {
                max = i;
            } 
        }
        //--------------------------------------------------------------------------------

        if(max!=-1)   //The dependent neighbor is within coverleafs
        {
            
            for(int j = 0; j < cover_leaves[pid].size(); j++)
            {
                if(cover_leaves_rho[pid][j] > rho_pid)
                {
                    dep_neighbor_cand.push_back(cover_leaves[pid][j]);
                    dep_neighbor_cand_dist.push_back(dist(datapoints[dep_neighbor_cand.back()], datapoints[pid]));
                }
            }
            auto min = min_element(dep_neighbor_cand_dist.begin(), dep_neighbor_cand_dist.end());
            dep_dist[pid] = *min;
            dep_neighbor[pid] = dep_neighbor_cand[min - dep_neighbor_cand_dist.begin()];
            
        }
        else    //Have to find the dependent neighbor and distance globally 
        {

            for(int j = 0; j < data_size; j++)
            {
                if(rho[j]>rho[pid])
                {
                    dep_neighbor_cand.push_back(j);
                    dep_neighbor_cand_dist.push_back(dist(datapoints[pid], datapoints[j]));
                }
            }
            
            if(dep_neighbor_cand.empty())
            {
                for(int j = 0; j < data_size; j++)
                {
                    dep_neighbor_cand_dist.push_back(dist(datapoints[pid], datapoints[j]));
                }
                dep_dist[pid] = *max_element(dep_neighbor_cand_dist.begin(), dep_neighbor_cand_dist.end());
                dep_neighbor[pid] = pid;

            }
            else
            {
                auto min = min_element(dep_neighbor_cand_dist.begin(), dep_neighbor_cand_dist.end());
                dep_dist[pid] = *min;
                dep_neighbor[pid] = dep_neighbor_cand[min - dep_neighbor_cand_dist.begin()];
                
            }
        }
        dep_neighbor_cand.clear();
        dep_neighbor_cand_dist.clear();
    }
    //---------------------------------------------------------------------------------------------
    
    //-------------------------------------------------
    int max = 0;
    for(int i = 0; i < cover_leaves.size(); i++)
    {
        cout<<"cover_leaves "<<i<<"size: "<<cover_leaves[i].size();
        if(cover_leaves[i].size() > max)
            max = cover_leaves[i].size();
    }
    cout<<"max cover_leaves size is : "<<max<<endl;
    //-------------------------------------------------


    vector<int> ids(data_size);
    int peak_num = 15;
    for(int i = 0; i < data_size; i++)
    {
        ids[i] = i;
        gamma[i] = rho[i]*dep_dist[i];
    }
    quickSort(gamma, 0, gamma.size()-1, ids);

    /*
    cout<<"The top 50 gamma values are: ";
    for(int i = 0; i<50; i++)
    {
        cout<<i<<"th. "<<gamma[data_size-i-1]<<endl;
        cout<<i<<"th. "<<ids[data_size-i-1]<<endl;
    }
    */

    vector<int> peak_cand;
    cout<<endl;
    for(int i = 0; i<peak_num; i++)
    {
        peak_cand.push_back(ids[data_size-i-1]);
    }
    
    //Get the clustering result
    for(int i = 0; i<data_size; i++)
    {
        int pt_now = i;
        while(find(peak_cand.begin(), peak_cand.end(), pt_now)==peak_cand.end())     //If point i is not a density peak
        {
            pt_now = dep_neighbor[pt_now];                                           //Then move to it's dependent neighbor
        }
        //Put the point to the cluster number of it's dependent peaks number
        cluster_result.push_back(distance(peak_cand.begin(), find(peak_cand.begin(), peak_cand.end(), pt_now))); 
        //-------------------------------------------------------------------------------------
    }
    //-------------------------------------------------------------------------------------


    
    return;
}


void print_cluster_result(vector<int> cluster_result)
{
    for(int i = 0; i<cluster_result.size(); i++)
    {
        cout<<"Point "<<setw(4)<<i<<" belongs to cluster "<<setw(2)<<cluster_result[i]<<endl;
        cluster_result_file<<"Point "<<setw(4)<<i<<" belongs to cluster "<<setw(2)<<cluster_result[i]<<endl;
    }
}

void print_VP_tree(vector<VP_node> &VP, vector<leaf_node> &leaf, int VP_size)
{
    for(int i = 0; i < VP_size; i++)
    {
        VP_tree_file<<"VP["<<i<<"]: "<<VP[i].vp<<"\t"<<"rad: "<<VP[i].rad<<endl;
    }
    for(int j = 0; j < leaf.size(); j++)
    {
        leaf_file<<"leaf "<<j<<": ";
        for(int k = 0; k< leaf[j].id.size(); k++)
        {
            leaf_file<<leaf[j].id[k]<<" ";
        }
        leaf_file<<endl;
    }
}

int main()
{
    timespec t1, t2;

    vector<vector<float>> datapoints;
    datapoints = readcsv("./csv file/3D_spatial_network.csv");
    normalize(datapoints);
    int data_size = datapoints.size();
    float dc = 0.00969449058;
    vector<int> cluster_result;

    //For VP tree
    

    vector<int> ids;
    initialize_ids(ids, data_size);

    
    int h = ceil(log(data_size/32)/log(2));     //tree height (h-1) 
    vector<VP_node> VP((int)pow(2, h)-1);
    vector<leaf_node> leaf((int)pow(2, h));
    //---------------------------------------------------------------------------------------------
    
    
    clock_gettime(CLOCK_MONOTONIC, &t1);

    //Calculate the distance of the first point to every point, choose the maximum as the initial VP
    decide_first_VP(datapoints , VP, data_size);
    //---------------------------------------------------------------------------------------------

    //Building the VP_tree recursively
    recur_build_VP_tree(datapoints, ids, VP, 0, leaf, h);
    //---------------------------------------------------------------------------------------------

    print_VP_tree(VP, leaf, (int)pow(2, h) - 1);

    #ifdef DEBUG
    //See the number of finding in which leaf of VP tree
    int j = 0;
    int finding = 1910;
    while(!leaf[j].id.empty())
    {
        for(int i = 0; i<leaf[j].id.size(); i++)
        {
            if(leaf[j].id[i] == finding)
            {
                cout<<"Find "<<finding<<endl;
                cout<<"In leaf: "<<j<<" id"<<i<<endl;
            }
        }
        j++;
    }
    //---------------------------------------------------------------------------------------------

    //Check the total point in the VP tree
    int sum = 0;
    for(int i = 0; i<leaf.size(); i++)
    {
        sum += leaf[i].id.size();
    }

    cout<<"sum is"<<sum;
    //---------------------------------------------------------------------------------------------
    #endif

    //GPDC
    serial_GDPC(datapoints, VP, leaf, dc, cluster_result, VP.size());
    //---------------------------------------------------------------------------------------------

    clock_gettime(CLOCK_MONOTONIC, &t2);
    cout << "time passed for serial GDPC is: " << 
    (t2.tv_sec - t1.tv_sec)*1000 + 
    (t2.tv_nsec - t1.tv_nsec)/1000000 << "ms" << endl;
    return 0;
}