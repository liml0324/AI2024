#include<vector>
#include<iostream>
#include<queue>
#include<map>
#include<fstream>
#include<sstream>
#include<string>
#include<algorithm>

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
};

// 自定义比较函数对象，按照 Search_Cell 结构体的 g + h 属性进行比较
struct CompareF {
    bool operator()(const Search_Cell *a, const Search_Cell *b) const {
        return (a->g + a->h) > (b->g + b->h); // 较小的 g + h 值优先级更高
    }
};

// TODO: 定义启发式函数
int Heuristic_Funtion(Map_Cell **Map, Search_Cell *search_cell, pair<int, int> start_point, pair<int, int> end_point)
{
    return abs(search_cell->map_cell->i - end_point.first) + abs(search_cell->map_cell->j - end_point.second);
    // return 0;
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
    search_cell->h = Heuristic_Funtion(Map, search_cell, start_point, end_point); // Heuristic_Funtion();

    priority_queue<Search_Cell *, vector<Search_Cell *>, CompareF> open_list;
    // vector<Search_Cell *> close_list;
    vector<vector<Search_Cell *>> parent = vector<vector<Search_Cell *>>(M, vector<Search_Cell *>(N, nullptr));
    parent[start_point.first][start_point.second] = search_cell;
    open_list.push(search_cell);
    Search_Cell *end_cell = nullptr;

    while(!open_list.empty())
    {
        // TODO: A*搜索过程实现
        times++;
        Search_Cell *current_cell = open_list.top();
        open_list.pop();
        //cout << current_cell->map_cell->i << " " << current_cell->map_cell->j << " " << current_cell->g << " " << current_cell->t << endl;
        // close_list.push_back(current_cell);
        if(current_cell->map_cell->type == 4)
        {
            // 到达终点
            end_cell = current_cell;
            break;
        }
        if(current_cell->map_cell->type == 2)
        {
            // 补给点
            current_cell->t = T;
        }
        if(current_cell->t == 0) 
        {
            continue;
        }
        if( current_cell->map_cell->i > 0 && \
            Map[current_cell->map_cell->i - 1][current_cell->map_cell->j].type != 1 && \
            parent[current_cell->map_cell->i - 1][current_cell->map_cell->j] == nullptr)// 可以向上搜索
        {
            Search_Cell *up_cell = new Search_Cell;
            up_cell->g = current_cell->g + 1;
            up_cell->map_cell = &Map[current_cell->map_cell->i - 1][current_cell->map_cell->j];
            up_cell->t = current_cell->t - 1;
            up_cell->action = 'U';
            up_cell->h = Heuristic_Funtion(Map, up_cell, start_point, end_point); // Heuristic_Funtion();
            open_list.push(up_cell);
            parent[up_cell->map_cell->i][up_cell->map_cell->j] = current_cell;
        }
        if(current_cell->map_cell->i < M - 1 && \
            Map[current_cell->map_cell->i + 1][current_cell->map_cell->j].type != 1 && \
            parent[current_cell->map_cell->i + 1][current_cell->map_cell->j] == nullptr)// 可以向下搜索
        {
            Search_Cell *down_cell = new Search_Cell;
            down_cell->g = current_cell->g + 1;
            down_cell->map_cell = &Map[current_cell->map_cell->i + 1][current_cell->map_cell->j];
            down_cell->t = current_cell->t - 1;
            down_cell->action = 'D';
            down_cell->h = Heuristic_Funtion(Map, down_cell, start_point, end_point); // Heuristic_Funtion();
            open_list.push(down_cell);
            parent[down_cell->map_cell->i][down_cell->map_cell->j] = current_cell;
        }
        if(current_cell->map_cell->j > 0 && \
            Map[current_cell->map_cell->i][current_cell->map_cell->j - 1].type != 1 && \
            parent[current_cell->map_cell->i][current_cell->map_cell->j - 1] == nullptr)// 可以向左搜索
        {
            Search_Cell *left_cell = new Search_Cell;
            left_cell->g = current_cell->g + 1;
            left_cell->map_cell = &Map[current_cell->map_cell->i][current_cell->map_cell->j - 1];
            left_cell->t = current_cell->t - 1;
            left_cell->action = 'L';
            left_cell->h = Heuristic_Funtion(Map, left_cell, start_point, end_point); // Heuristic_Funtion();
            open_list.push(left_cell);
            parent[left_cell->map_cell->i][left_cell->map_cell->j] = current_cell;
        }
        if(current_cell->map_cell->j < N - 1 && \
            Map[current_cell->map_cell->i][current_cell->map_cell->j + 1].type != 1 && \
            parent[current_cell->map_cell->i][current_cell->map_cell->j + 1] == nullptr)// 可以向右搜索
        {
            Search_Cell *right_cell = new Search_Cell;
            right_cell->g = current_cell->g + 1;
            right_cell->map_cell = &Map[current_cell->map_cell->i][current_cell->map_cell->j + 1];
            right_cell->t = current_cell->t - 1;
            right_cell->action = 'R';
            right_cell->h = Heuristic_Funtion(Map, right_cell, start_point, end_point); // Heuristic_Funtion();
            open_list.push(right_cell);
            parent[right_cell->map_cell->i][right_cell->map_cell->j] = current_cell;
        }
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
    
    while(end_cell->map_cell->type != 3) {
        way.push_back(end_cell->action);
        end_cell = parent[end_cell->map_cell->i][end_cell->map_cell->j];
    }
    reverse(way.begin(), way.end());

    // ------------------------------------------------------------------
    // 释放动态内存
    for(int i = 0; i < M; i++)
    {
        delete[] Map[i];
    }
    delete[] Map;
    while(!open_list.empty())
    {
        auto temp = open_list.top();
        delete[] temp;
        open_list.pop();
    }
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
    string input_base = "../input/input_";
    string output_base = "../output/output_";
    // input_0为讲义样例，此处不做测试
    for(int i = 1; i < 11; i++)
    {
        int step_nums = -1;
        string way = "";
        Astar_search(input_base + to_string(i) + ".txt", step_nums, way) << ' ';
        output(output_base + to_string(i) + ".txt", step_nums, way);
    }
    cout << endl;
    return 0;
}