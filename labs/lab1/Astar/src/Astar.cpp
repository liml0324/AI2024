#include<vector>
#include<iostream>
#include<queue>
#include<map>
#include<fstream>
#include<sstream>
#include<string>
#include<algorithm>
#include<time.h>

using namespace std;

struct Map_Cell
{
    int type;
    // TODO: 定义地图信息
    int i, j;
};

struct Search_Cell
{
    int h;
    int g;
    // TODO: 定义搜索状态
    Map_Cell *map_cell;
    int t;
    char action;
    Search_Cell * parent;
};

// 自定义比较函数对象，按照 Search_Cell 结构体的 g + h 属性进行比较
struct CompareF {
    bool operator()(const Search_Cell *a, const Search_Cell *b) const {
        return (a->g + a->h) > (b->g + b->h); // 较小的 g + h 值优先级更高
    }
};

// 标准库的优先队列不能修改元素，因此需要自定义优先队列
class MinHeap {
    private:
    vector<Search_Cell *> heap;
    map<tuple<int, int, int>, int> index_map;
    public:
    MinHeap() {
        heap.push_back(nullptr);
    }
    void push(Search_Cell *cell) {
        heap.push_back(cell);
        int index = heap.size() - 1;
        index_map[make_tuple(cell->map_cell->i, cell->map_cell->j, cell->t)] = index;
        while(index > 1) {
            int parent_index = index / 2;
            if(heap[parent_index]->g + heap[parent_index]->h <= cell->g + cell->h) {
                break;
            }
            swap(heap[parent_index], heap[index]);
            index_map[make_tuple(heap[parent_index]->map_cell->i, heap[parent_index]->map_cell->j, heap[parent_index]->t)] = parent_index;
            index_map[make_tuple(heap[index]->map_cell->i, heap[index]->map_cell->j, heap[index]->t)] = index;
            index = parent_index;
        }
    }
    Search_Cell *top() {
        return heap[1];
    }
    void pop() {
        if(heap.size() == 1) {
            return;
        }
        if(heap.size() == 2) {
            index_map.erase(make_tuple(heap[1]->map_cell->i, heap[1]->map_cell->j, heap[1]->t));
            heap.pop_back();
            return;
        }
        index_map.erase(make_tuple(heap[1]->map_cell->i, heap[1]->map_cell->j, heap[1]->t));
        heap[1] = heap.back();
        heap.pop_back();
        index_map[make_tuple(heap[1]->map_cell->i, heap[1]->map_cell->j, heap[1]->t)] = 1;
        int index = 1;
        while(index * 2 < heap.size()) {
            int left_index = index * 2;
            int right_index = index * 2 + 1;
            int min_index = index;
            if(heap[left_index]->g + heap[left_index]->h < heap[min_index]->g + heap[min_index]->h) {
                min_index = left_index;
            }
            if(right_index < heap.size() && heap[right_index]->g + heap[right_index]->h < heap[min_index]->g + heap[min_index]->h) {
                min_index = right_index;
            }
            if(min_index == index) {
                break;
            }
            swap(heap[min_index], heap[index]);
            index_map[make_tuple(heap[min_index]->map_cell->i, heap[min_index]->map_cell->j, heap[min_index]->t)] = min_index;
            index_map[make_tuple(heap[index]->map_cell->i, heap[index]->map_cell->j, heap[index]->t)] = index;
            index = min_index;
        }
    }
    void update(Search_Cell *cell) {
        int index = index_map[make_tuple(cell->map_cell->i, cell->map_cell->j, cell->t)];
        while(index > 1) {
            int parent_index = index / 2;
            if(heap[parent_index]->g + heap[parent_index]->h <= cell->g + cell->h) {
                break;
            }
            swap(heap[parent_index], heap[index]);
            index_map[make_tuple(heap[parent_index]->map_cell->i, heap[parent_index]->map_cell->j, heap[parent_index]->t)] = parent_index;
            index_map[make_tuple(heap[index]->map_cell->i, heap[index]->map_cell->j, heap[index]->t)] = index;
            index = parent_index;
        }
        while(index * 2 < heap.size()) {
            int left_index = index * 2;
            int right_index = index * 2 + 1;
            int min_index = index;
            if(heap[left_index]->g + heap[left_index]->h < heap[min_index]->g + heap[min_index]->h) {
                min_index = left_index;
            }
            if(right_index < heap.size() && heap[right_index]->g + heap[right_index]->h < heap[min_index]->g + heap[min_index]->h) {
                min_index = right_index;
            }
            if(min_index == index) {
                break;
            }
            swap(heap[min_index], heap[index]);
            index_map[make_tuple(heap[min_index]->map_cell->i, heap[min_index]->map_cell->j, heap[min_index]->t)] = min_index;
            index_map[make_tuple(heap[index]->map_cell->i, heap[index]->map_cell->j, heap[index]->t)] = index;
            index = min_index;
        }
    }
    bool empty() {
        return heap.size() == 1;
    }
    Search_Cell *find(Search_Cell *cell) {
        if(index_map.find(make_tuple(cell->map_cell->i, cell->map_cell->j, cell->t)) == index_map.end()) {
            return nullptr;
        }
        return heap[index_map[make_tuple(cell->map_cell->i, cell->map_cell->j, cell->t)]];
    }
    Search_Cell *find(tuple<int, int, int> key) {
        if(index_map.find(key) == index_map.end()) {
            return nullptr;
        }
        return heap[index_map[key]];
    }
    Search_Cell *end() {
        return nullptr;
    }
};

// TODO: 定义启发式函数
int Heuristic_Funtion(Map_Cell **Map, Search_Cell *search_cell, pair<int, int> start_point, pair<int, int> end_point)
{
    // return abs(search_cell->map_cell->i - end_point.first) + abs(search_cell->map_cell->j - end_point.second);
    return 0;
}

int Astar_search(const string input_file, int &step_nums, string &way)
{
    int times = 0;
    ifstream file(input_file);
    if (!file.is_open()) {
        cout << "Error opening file!" << endl;
        return times;
    }

    string line;
    getline(file, line); // 读取第一行
    stringstream ss(line);
    string word;
    vector<string> words;
    while (ss >> word) {
        words.push_back(word);
    }
    int M = stoi(words[0]);
    int N = stoi(words[1]);
    int T = stoi(words[2]);

    pair<int, int> start_point; // 起点
    pair<int, int> end_point;   // 终点
    Map_Cell **Map = new Map_Cell *[M];
    // 加载地图
    for(int i = 0; i < M; i++)
    {
        Map[i] = new Map_Cell[N];
        getline(file, line);
        stringstream ss(line);
        string word;
        vector<string> words;
        while (ss >> word) {
            words.push_back(word);
        }
        for(int j = 0; j < N; j++)
        {
            Map[i][j].type = stoi(words[j]);
            Map[i][j].i = i;
            Map[i][j].j = j;
            if(Map[i][j].type == 3)
            {
                start_point = {i, j};
            }
            else if(Map[i][j].type == 4)
            {
                end_point = {i, j};
            }
        }
    }
    // 以上为预处理部分
    // ------------------------------------------------------------------

    Search_Cell *search_cell = new Search_Cell;
    search_cell->g = 0;
    search_cell->map_cell = &Map[start_point.first][start_point.second];
    //cout << search_cell->map_cell->i << " " << search_cell->map_cell->j << endl;
    search_cell->t = T;
    search_cell->parent = nullptr;
    search_cell->h = Heuristic_Funtion(Map, search_cell, start_point, end_point); // Heuristic_Funtion();

    MinHeap open_list;
    // map<tuple<int, int, int>, Search_Cell *> open_list;
    // vector<Search_Cell *> close_list;
    map<tuple<int, int, int>, Search_Cell *> close_list;
    // vector<vector<Search_Cell *>> parent = vector<vector<Search_Cell *>>(M, vector<Search_Cell *>(N, nullptr));
    open_list.push(search_cell);
    Search_Cell *end_cell = nullptr;

    while(!open_list.empty())
    {
        // TODO: A*搜索过程实现
        times++;
        Search_Cell *current_cell = open_list.top();
        open_list.pop();
        // open_list.erase(make_tuple(current_cell->map_cell->i, current_cell->map_cell->j, current_cell->t))
        //cout << current_cell->map_cell->i << " " << current_cell->map_cell->j << " " << current_cell->g << " " << current_cell->t << endl;
        // close_list.push_back(current_cell);
        if(current_cell->map_cell->type == 4)
        {
            // 到达终点
            end_cell = current_cell;
            break;
        }
        if(current_cell->t == 0) 
        {
            continue;
        }
        vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        vector<char> actions = {'U', 'D', 'L', 'R'};
        for(int k = 0; k < 4; k++) {
            auto new_i = current_cell->map_cell->i + directions[k].first;
            auto new_j = current_cell->map_cell->j + directions[k].second;
            if(new_i < 0 || new_i >= M || new_j < 0 || new_j >= N || Map[new_i][new_j].type == 1) {
                continue;
            }
            auto new_t = Map[new_i][new_j].type == 2 ? T : current_cell->t - 1; // 考虑补给点
            // if(new_t <= 0) {    // 体力归零
            //     continue;
            // }
            auto next_cell = new Search_Cell;
            next_cell->map_cell = &Map[new_i][new_j];
            next_cell->g = current_cell->g + 1;
            next_cell->t = new_t;
            next_cell->h = Heuristic_Funtion(Map, next_cell, start_point, end_point);
            next_cell->action = actions[k];
            if(open_list.find(make_tuple(new_i, new_j, new_t)) == open_list.end() && close_list.find(make_tuple(new_i, new_j, new_t)) == close_list.end()){
                next_cell->parent = current_cell;
                open_list.push(next_cell);
            } 
            else {
                auto temp = open_list.find(make_tuple(new_i, new_j, new_t));
                if(temp != nullptr) {   // 在open_list中
                    if(next_cell->g < temp->g) {    // 如果g值更小，则更新g值
                        // cout << "update" << endl;
                        // cout << "temp: " << temp->map_cell->i << " " << temp->map_cell->j << " " << temp->t << endl;
                        // cout << "next_cell: " << next_cell->map_cell->i << " " << next_cell->map_cell->j << " " << next_cell->t << endl;
                        // cout << "current_cell: " << current_cell->map_cell->i << " " << current_cell->map_cell->j << " " << current_cell->t << endl;
                        temp->g = next_cell->g;
                        temp->parent = current_cell;
                        temp->action = next_cell->action;
                        temp->h = next_cell->h;
                        open_list.update(temp);
                    }
                    delete[] next_cell;
                }
                else {
                    temp = close_list[make_tuple(new_i, new_j, new_t)];
                    if(next_cell->g < temp->g) {
                        temp->g = next_cell->g;
                        temp->parent = current_cell;
                        temp->action = next_cell->action;
                        temp->h = next_cell->h;
                        open_list.push(temp);
                        close_list.erase(make_tuple(new_i, new_j, new_t));
                    }
                    delete[] next_cell;
                }
            }
        }
        close_list[make_tuple(current_cell->map_cell->i, current_cell->map_cell->j, current_cell->t)] = current_cell;
    }
    // cout << "finish search" << endl;

    // ------------------------------------------------------------------
    // TODO: 填充step_nums与way
    if(end_cell == nullptr)
    {
        step_nums = -1;
        return times;
    }
    step_nums = end_cell->g;
    
    while(end_cell && end_cell->map_cell->type != 3) {
        // cout << end_cell->map_cell->i << " " << end_cell->map_cell->j << " " << end_cell->action << endl;
        way.push_back(end_cell->action);
        end_cell = end_cell->parent;
    }
    reverse(way.begin(), way.end());

    // ------------------------------------------------------------------
    // 释放动态内存
    while(!open_list.empty())
    {
        auto temp = open_list.top();
        open_list.pop();
        delete[] temp;
    }
    for(int i = 0; i < M; i++)
    {
        delete[] Map[i];
    }
    delete[] Map;
    
    // for(int i = 0; i < close_list.size(); i++)
    // {
    //     delete[] close_list[i];
    // }
    delete[] search_cell;
    return times;
}

void output(const string output_file, int &step_nums, string &way)
{
    ofstream file(output_file);
    if(file.is_open())
    {
        file << step_nums << endl;
        if(step_nums >= 0)
        {
            file << way << endl;
        }

        file.close();
    }
    else
    {
        cout << "Can not open file: " << output_file << endl;
    }
    return;
}

int main(int argc, char *argv[])
{
    auto begin_time = clock();
    string input_base = "../input/input_";
    string output_base = "../output/output_";
    // input_0为讲义样例，此处不做测试
    for(int i = 0; i < 11; i++)
    {
        int step_nums = -1;
        string way = "";
        Astar_search(input_base + to_string(i) + ".txt", step_nums, way);
        output(output_base + to_string(i) + ".txt", step_nums, way);
    }
    // cout << endl;
    auto end_time = clock();
    cout << "time: " << (double)(end_time-begin_time) / CLOCKS_PER_SEC << endl;
    return 0;
}