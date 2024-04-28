#include <fstream>
#include "node.h"
#include <time.h>

using namespace ChineseChess;

int checkRepeatNode(GameTreeNode* node, std::vector<GameTreeNode*> history) {
    for(int i = 0; i < history.size(); i++) {
        auto n = history[i];
        if(node->getPieceChar() != n->getPieceChar()) {
            continue;
        }
        auto node_pieces = node->getBoardClass().getChessPiece();
        auto n_pieces = n->getBoardClass().getChessPiece();
        auto same = true;
        for(int j = 0; j < node_pieces.size(); j++) {
            if(node_pieces[j].name != n_pieces[j].name || node_pieces[j].color != n_pieces[j].color || node_pieces[j].init_x != n_pieces[j].init_x || node_pieces[j].init_y != n_pieces[j].init_y) {
                same = false;
                break;
            }
        }
        if(same) {
            return i;
        }
    }
    return -1;
}

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
    auto meet = node.checkJiangMeet();
    if(meet) {
        if(isMaximizer) {
            return std::numeric_limits<int>::max()-1;
        }
        else {
            return std::numeric_limits<int>::min()+1;
        }
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
                moves[i].score = eval;
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
                moves[i].score = eval;
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
    auto depth = 5;
    auto max_steps = 30;
    auto begin_time = clock();
    auto total_step = 0;
    for(int x = 1; x <= 10; x++) {
        std::cout << "Case " << x << std::endl;
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
        std::ofstream out("../output/output_"+std::to_string(x)+".txt");
        if(!out.is_open()) {
            std::cout << "open file failed" << std::endl;
            continue;
        }
        auto score = 0;
        auto node = &root;
        auto steps = 0;
        std::vector<GameTreeNode*> red_history;
        std::vector<GameTreeNode*> black_history;
        while(score < std::numeric_limits<int>::max()-1 && score > std::numeric_limits<int>::min()+1 && steps < max_steps) {
            // 红方行动
            auto repeat_num = checkRepeatNode(node, red_history);
            if(repeat_num != -1) {
                auto best_move = red_history[repeat_num]->getBestMove();
                if(best_move.score > 0) {
                    node->removeMove(true, best_move);
                }
            }
            alphaBeta(*node, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), depth, true);
            red_history.push_back(node);
            total_step++;
            auto board = node->getBoardClass().getBoard();
            auto best_move = node->getBestMove();
            auto piece_char = board[best_move.init_y][best_move.init_x];
            out << piece_char << " (" << best_move.init_x << "," << 9-best_move.init_y << ") (" << best_move.next_x << "," << 9-best_move.next_y << ")" << std::endl;
            steps++;
            auto next_node = node->updateBoard(best_move);
            score = next_node->getEvaluationScore();
            if(score >= std::numeric_limits<int>::max()-1 || score <= std::numeric_limits<int>::min()+1) {
                break;
            }
            // 黑方行动
            repeat_num = checkRepeatNode(next_node, black_history);
            if(repeat_num != -1) {
                auto best_move = black_history[repeat_num]->getBestMove();
                if(best_move.score < 0) {
                    next_node->removeMove(false, best_move);
                }
            }
            alphaBeta(*next_node, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), depth, false);
            black_history.push_back(next_node);
            total_step++;
            board = next_node->getBoardClass().getBoard();
            best_move = next_node->getBestMove();
            piece_char = board[best_move.init_y][best_move.init_x];
            out << piece_char << " (" << best_move.init_x << "," << 9-best_move.init_y << ") (" << best_move.next_x << "," << 9-best_move.next_y << ")" << std::endl;
            node = next_node->updateBoard(best_move);
            score = node->getEvaluationScore();
            steps++;
        }
        for(auto n : red_history) {
            if(n != &root)
                delete n;
        }
        for(auto n : black_history) {
            if(n != &root)
                delete n;
        }
        out.close();
    }
    auto end_time = clock();
    std::cout << "average step time: " << (double)(end_time - begin_time) / CLOCKS_PER_SEC / total_step << std::endl;
    return 0;
}