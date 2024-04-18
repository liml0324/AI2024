#include <fstream>
#include "node.h"
#include <time.h>

using namespace ChineseChess;

//博弈树搜索，depth为搜索深度
int alphaBeta(GameTreeNode &node, int alpha, int beta, int depth, bool isMaximizer) {
    if (depth == 0) {
        return node.getEvaluationScore();
    }
    // 如果将死了，直接返回极值
    int Jiang_alive = node.haveJiang();
    if(Jiang_alive == 1) {
        return std::numeric_limits<int>::max()-1;
    }
    if(Jiang_alive == -1) {
        return std::numeric_limits<int>::min()+1;
    }
    //TODO alpha-beta剪枝过程
    if (isMaximizer) {
        int maxEval = std::numeric_limits<int>::min();
        auto board = node.getBoardClass();
        auto moves = board.getMoves(true);// 红方是max方
        for (int i = 0; i < moves.size(); i++) {
            auto new_node = node.updateBoard(moves[i]);
            int eval = alphaBeta(*new_node, alpha, beta, depth - 1, false);
            delete new_node;
            if(eval > maxEval){
                node.setBestMove(moves[i]);
            }
            maxEval = std::max(maxEval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha) {
                break;
            }
        }
        return maxEval;
    } else {
        int minEval = std::numeric_limits<int>::max();
        auto board = node.getBoardClass();
        auto moves = board.getMoves(false);// 黑方是min方
        for (int i = 0; i < moves.size(); i++) {
            auto new_node = node.updateBoard(moves[i]);
            int eval = alphaBeta(*new_node, alpha, beta, depth - 1, true);
            delete new_node;
            if(eval < minEval){
                node.setBestMove(moves[i]);
            }
            minEval = std::min(minEval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha) {
                break;
            }
        }
        return minEval;
    }
    return 0;
}

int main() {
    auto depth = 4;
    auto begin_time = clock();
    for(int x = 1; x <= 10; x++) {
        std::ifstream file("../input/"+std::to_string(x)+".txt");
        std::vector<std::vector<char>> board;

        std::string line;
        int n = 0;
        while (std::getline(file, line)) {
            std::vector<char> row;

            for (char ch : line) {
                row.push_back(ch);
            }

            board.push_back(row);
            n = n + 1;
            if (n >= 10) break;
        }

        file.close();
        GameTreeNode root(true, board, std::numeric_limits<int>::min());

        //TODO
        std::cout << "Case " << x << std::endl;
        std::cout << "best score: " << alphaBeta(root, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), depth, true) << std::endl;
        std::cout << "best move: " << root.getBestMove().init_x << " " << root.getBestMove().init_y << "->" << root.getBestMove().next_x << " " << root.getBestMove().next_y << std::endl;
    }
    auto end_time = clock();
    std::cout << "time: " << (double)(end_time - begin_time) / CLOCKS_PER_SEC << std::endl;
    


    //代码测试
    // ChessBoard _board = root.getBoardClass();
    // std::vector<std::vector<char>> cur_board = _board.getBoard();

    // for (int i = 0; i < cur_board.size(); i++) {
    //     for (int j = 0; j < cur_board[0].size(); j++) {
    //         std::cout << cur_board[i][j];
    //     }
    //     std::cout << std::endl;
    // }

    // std::vector<Move> red_moves = _board.getMoves(true);
    // std::vector<Move> black_moves = _board.getMoves(false);

    // for (int i = 0; i < red_moves.size(); i++) {
    //     std::cout << "init: " << red_moves[i].init_x <<  " " << red_moves[i].init_y << std::endl;
    //     std::cout << "next: " << red_moves[i].next_x <<  " " << red_moves[i].next_y << std::endl;
    //     std::cout << "score " << red_moves[i].score << std::endl;
    // }
    // for (int i = 0; i < black_moves.size(); i++) {
    //     std::cout << "init: " << black_moves[i].init_x <<  " " << black_moves[i].init_y << std::endl;
    //     std::cout << "next: " << black_moves[i].next_x <<  " " << black_moves[i].next_y << std::endl;
    //     std::cout << "score " << black_moves[i].score << std::endl;
    // }

    return 0;
}