#include<bits/stdc++.h>
using namespace std;

auto goal_state = tuple<int, int, int>(0, 0, 0);


auto next_states(tuple<int, int, int> state) {
    auto next_states = vector<tuple<int, int, int>>();
    auto [m, c, b] = state;
    if(b == 1) {// 船在这一边
        if(m-1 >= 0 && (m-1 >=c || m-1 == 0) && 3-m+1 >= 3-c) { // 运一个传教士到另一边
            auto next_state = tuple<int, int, int>(m-1, c, 0);
            next_states.push_back(next_state);
        }
        if(c-1 >= 0 && (m >= c-1 || m == 0) && (3-m >= 3-c+1 || 3-m == 0)) { // 运一个野人到另一边
            auto next_state = tuple<int, int, int>(m, c-1, 0);
            next_states.push_back(next_state);
        }
        if(m-1 >= 0 && c-1 >= 0 && (m-1 >= c-1 || m-1 == 0) && (3-m+1 >= 3-c+1 || 3-m+1 == 0)) { // 运一个传教士和一个野人到另一边
            auto next_state = tuple<int, int, int>(m-1, c-1, 0);
            next_states.push_back(next_state);
        }
        if(m-2 >= 0 && (m-2 >= c || m-2 == 0) && 3-m+2 >= 3-c) { // 运两个传教士到另一边
            auto next_state = tuple<int, int, int>(m-2, c, 0);
            next_states.push_back(next_state);
        }
        if(c-2 >= 0 && (m >= c-2 || m == 0) && (3-m >= 3-c+2 || 3-m == 0)) { // 运两个野人到另一边
            auto next_state = tuple<int, int, int>(m, c-2, 0);
            next_states.push_back(next_state);
        }
    }
    else {// 船在另一边
        if(m+1 <= 3 && m+1 >=c && (3-m-1 >= 3-c || 3-m-1 == 0)) { // 运一个传教士到这一边
            auto next_state = tuple<int, int, int>(m+1, c, 1);
            next_states.push_back(next_state);
        }
        if(c+1 <= 3 && (m >= c+1 || m == 0) && (3-m >= 3-c-1 || 3-m == 0)) { // 运一个野人到这一边
            auto next_state = tuple<int, int, int>(m, c+1, 1);
            next_states.push_back(next_state);
        }
        if(m+1 <= 3 && c+1 <= 3 && m+1 >= c+1 && (3-m-1 >= 3-c-1 || 3-m-1 == 0)) { // 运一个传教士和一个野人到这一边
            auto next_state = tuple<int, int, int>(m+1, c+1, 1);
            next_states.push_back(next_state);
        }
        if(m+2 <= 3 && m+2 >=c && (3-m-2 >= 3-c || 3-m-2 == 0)) { // 运两个传教士到这一边
            auto next_state = tuple<int, int, int>(m+2, c, 1);
            next_states.push_back(next_state);
        }
        if(c+2 <= 3 && (m >= c+2 || m == 0) && (3-m >= 3-c-2 || 3-m == 0)) { // 运两个野人到这一边
            auto next_state = tuple<int, int, int>(m, c+2, 1);
            next_states.push_back(next_state);
        }
    }
    return next_states;
}

void BFS_search(tuple<int, int, int> init_state, vector<tuple<int, int, int>> & path) {
    queue<tuple<int, int, int>> q;
    q.push(init_state);
    auto prev = map<tuple<int, int, int>, tuple<int, int, int>>();
    while (!q.empty()) {
        auto state = q.front();
        q.pop();
        if (state == goal_state) {
            break;
        }
        for (auto next_state : next_states(state)) {
            if (prev.find(next_state) != prev.end()) {
                continue;
            }
            prev[next_state] = state;
            q.push(next_state);
            // cout << "(" << get<0>(next_state) << ", " << get<1>(next_state) << ", " << get<2>(next_state) << ")" << endl;
        }
    }
    auto state = goal_state;
    while (state != init_state) {
        path.push_back(state);
        state = prev[state];
    }
    path.push_back(init_state);
    reverse(path.begin(), path.end());
    return;
}

int main() {
    auto state = tuple<int, int, int>(3, 3, 1);
    vector<tuple<int, int, int>> path;
    BFS_search(state, path);
    for (auto state : path) {
        auto [m, c, b] = state;
        cout << "(" << m << ", " << c << ", " << b << ")" << endl;
    }
}