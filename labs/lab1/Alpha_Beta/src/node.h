#include <vector>
#include <map>
#include <limits>
#include <iostream>
#include <string>

namespace ChineseChess
{
    //棋力评估，这里的棋盘方向和输入棋盘方向不同，在使用时需要仔细
    //生成合法动作代码部分已经使用，经过测试是正确的，大家可以参考
    std::vector<std::vector<int>> JiangPosition = {
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {1, -8, -9, 0, 0, 0, 0, 0, 0, 0},
                                                {5, -8, -9, 0, 0, 0, 0, 0, 0, 0},
                                                {1, -8, -9, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                            };

    std::vector<std::vector<int>> ShiPosition = {
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 3, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                            };
        
    std::vector<std::vector<int>> XiangPosition = {
                                                {0, 0, -2, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 3, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, -2, 0, 0, 0, 0, 0, 0, 0},
                                            };

    std::vector<std::vector<int>> MaPosition = {
                                                {0, -3, 5, 4, 2, 2, 5, 4, 2, 2},
                                                {-3, 2, 4, 6, 10, 12, 20, 10, 8, 2},
                                                {2, 4, 6, 10, 13, 11, 12, 11, 15, 2},
                                                {0, 5, 7, 7, 14, 15, 19, 15, 9, 8},
                                                {2, -10, 4, 10, 15, 16, 12, 11, 6, 2},
                                                {0, 5, 7, 7, 14, 15, 19, 15, 9, 8},
                                                {2, 4, 6, 10, 13, 11, 12, 11, 15, 2},
                                                {-3, 2, 4, 6, 10, 12, 20, 10, 8, 2},
                                                {0, -3, 5, 4, 2, 2, 5, 4, 2, 2},
                                            };

    std::vector<std::vector<int>> PaoPosition = {
                                                {0, 0, 1, 0, -1, 0, 0, 1, 2, 4},
                                                {0, 1, 0, 0, 0, 0, 3, 1, 2, 4},
                                                {1, 2, 4, 0, 3, 0, 3, 0, 0, 0},
                                                {3, 2, 3, 0, 0, 0, 2, -5, -4, -5},
                                                {3, 2, 5, 0, 4, 4, 4, -4, -7, -6},
                                                {3, 2, 3, 0, 0, 0, 2, -5, -4, -5},
                                                {1, 2, 4, 0, 3, 0, 3, 0, 0, 0},
                                                {0, 1, 0, 0, 0, 0, 3, 1, 2, 4},
                                                {0, 0, 1, 0, -1, 0, 0, 1, 2, 4},
                                            };

    std::vector<std::vector<int>> JuPosition = {
                                                {-6, 5, -2, 4, 8, 8, 6, 6, 6, 6},
                                                {6, 8, 8, 9, 12, 11, 13, 8, 12, 8},
                                                {4, 6, 4, 4, 12, 11, 13, 7, 9, 7},
                                                {12, 12, 12, 12, 14, 14, 16, 14, 16, 13},
                                                {0, 0, 12, 14, 15, 15, 16, 16, 33, 14},
                                                {12, 12, 12, 12, 14, 14, 16, 14, 16, 13},
                                                {4, 6, 4, 4, 12, 11, 13, 7, 9, 7},
                                                {6, 8, 8, 9, 12, 11, 13, 8, 12, 8},
                                                {-6, 5, -2, 4, 8, 8, 6, 6, 6, 6},
                                            };

    std::vector<std::vector<int>> BingPosition = {
                                                {0, 0, 0, -2, 3, 10, 20, 20, 20, 0},
                                                {0, 0, 0, 0, 0, 18, 27, 30, 30, 0},
                                                {0, 0, 0, -2, 4, 22, 30, 45, 50, 0},
                                                {0, 0, 0, 0, 0, 35, 40, 55, 65, 2},
                                                {0, 0, 0, 6, 7, 40, 42, 55, 70, 4},
                                                {0, 0, 0, 0, 0, 35, 40, 55, 65, 2},
                                                {0, 0, 0, -2, 4, 22, 30, 45, 50, 0},
                                                {0, 0, 0, 0, 0, 18, 27, 30, 30, 0},
                                                {0, 0, 0, -2, 3, 10, 20, 20, 20, 0},
                                            };

    //棋子价值评估
    std::map<std::string, int> piece_values = {
                                            {"Jiang", 10000},
                                            {"Shi", 10},
                                            {"Xiang", 30},
                                            {"Ma", 300},
                                            {"Ju", 500},
                                            {"Pao", 300},
                                            {"Bing", 90}
                                        };

    //行期可能性评估，这里更多是对下一步动作的评估
    std::map<std::string, int> next_move_values = {
                                            {"Jiang", 15000},
                                            {"Ma", 100},
                                            {"Ju", 500},
                                            {"Pao", 100},
                                            {"Bing", -20}
                                        };
    
    //动作结构体，每个动作设置score，可以方便剪枝
    struct Move {
        int init_x;
        int init_y;
        int next_x;
        int next_y;
        int score;
    };

    // 定义棋盘上的棋子结构体
    struct ChessPiece {
        char name; // 棋子名称
        int init_x, init_y; // 棋子的坐标
        bool color;   //棋子阵营 true为红色、false为黑色
    };

    // 定义棋盘类
    class ChessBoard {
    private:
        int sizeX, sizeY;   //棋盘大小，固定
        std::vector<ChessPiece> pieces;   //棋盘上所有棋子
        std::vector<std::vector<char>> board;    //当前棋盘、二维数组表示
        std::vector<Move> red_moves;    //红方棋子的合法动作
        std::vector<Move> black_moves;   //黑方棋子的合法动作
    public:
        std::string getPieceName(char pieceChar) {
            switch (pieceChar) {
                case 'R':
                    return "Ju";
                case 'C':
                    return "Pao";
                case 'N':
                    return "Ma";
                case 'B':
                    return "Xiang";
                case 'A':
                    return "Shi";
                case 'K':
                    return "Jiang";
                case 'P':
                    return "Bing";
                case 'r':
                    return "Ju";
                case 'c':
                    return "Pao";
                case 'n':
                    return "Ma";
                case 'b':
                    return "Xiang";
                case 'a':
                    return "Shi";
                case 'k':
                    return "Jiang";
                case 'p':
                    return "Bing";
                default:
                    return "";
            }
        }

        // 初始化棋盘，提取棋盘上棋子，并生成所有合法动作
        void initializeBoard(const std::vector<std::vector<char>>& init_board) {
            board = init_board;
            sizeX = board.size();
            sizeY = board[0].size();

            for (int i = 0; i < sizeX; ++i) {
                for (int j = 0; j < sizeY; ++j) {
                    char pieceChar = board[i][j];
                    if (pieceChar == '.') continue;

                    ChessPiece piece;
                    // 这里做了一个转置，x是横坐标，y是纵坐标
                    piece.init_x = j;
                    piece.init_y = i;
                    piece.color = (pieceChar >= 'A' && pieceChar <= 'Z');
                    piece.name = pieceChar;
                    pieces.push_back(piece);

                    switch (pieceChar) {
                        case 'R':
                            generateJuMoves(j, i, piece.color);
                            break;
                        case 'C':
                            generatePaoMoves(j, i, piece.color);
                            break;
                        case 'N':
                            generateMaMoves(j, i, piece.color);
                            break;
                        case 'B':
                            generateXiangMoves(j, i, piece.color);
                            break;
                        case 'A':
                            generateShiMoves(j, i, piece.color);
                            break;
                        case 'K':
                            generateJiangMoves(j, i, piece.color);
                            break;
                        case 'P':
                            generateBingMoves(j, i, piece.color);
                            break;
                        case 'r':
                            generateJuMoves(j, i, piece.color);
                            break;
                        case 'c':
                            generatePaoMoves(j, i, piece.color);
                            break;
                        case 'n':
                            generateMaMoves(j, i, piece.color);
                            break;
                        case 'b':
                            generateXiangMoves(j, i, piece.color);
                            break;
                        case 'a':
                            generateShiMoves(j, i, piece.color);
                            break;
                        case 'k':
                            generateJiangMoves(j, i, piece.color);
                            break;
                        case 'p':
                            generateBingMoves(j, i, piece.color);
                            break;
                        default:
                            break;
                    }
                } 
            }
        }
        
        //生成车的合法动作
        void generateJuMoves(int x, int y, bool color) {
            //前后左右分别进行搜索，遇到棋子停止，不同阵营可以吃掉
            std::vector<Move> JuMoves;
            for(int i = x + 1; i < sizeY; i++) {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = i;
                cur_move.next_y = y;
                cur_move.score = 0;
                if (board[y][i] != '.') {
                    bool cur_color = (board[y][i] >= 'A' && board[y][i] <= 'Z');
                    if (cur_color != color) {
                        JuMoves.push_back(cur_move);
                    }
                    break;
                }
                JuMoves.push_back(cur_move);
            }

            for(int i = x - 1; i >= 0; i--) {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = i;
                cur_move.next_y = y;
                cur_move.score = 0;
                if (board[y][i] != '.') {
                    bool cur_color = (board[y][i] >= 'A' && board[y][i] <= 'Z');
                    if (cur_color != color) {
                        JuMoves.push_back(cur_move);
                    }
                    break;
                }
                JuMoves.push_back(cur_move);
            }

            for(int j = y + 1; j < sizeX; j++) {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = x;
                cur_move.next_y = j;
                cur_move.score = 0;
                if (board[j][x] != '.') {
                    bool cur_color = (board[j][x] >= 'A' && board[j][x] <= 'Z');
                    if (cur_color != color) {
                        JuMoves.push_back(cur_move);
                    }
                    break;
                }
                JuMoves.push_back(cur_move);
            }

            for(int j = y - 1; j >= 0; j--) {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = x;
                cur_move.next_y = j;
                cur_move.score = 0;
                if (board[j][x] != '.') {
                    bool cur_color = (board[j][x] >= 'A' && board[j][x] <= 'Z');
                    if (cur_color != color) {
                        JuMoves.push_back(cur_move);
                    }
                    break;
                }
                JuMoves.push_back(cur_move); 
            }
            for (int i = 0; i < JuMoves.size(); i++) {
                if(color) {
                    JuMoves[i].score = JuPosition[JuMoves[i].next_x][9 - JuMoves[i].next_y] - JuPosition[x][9 -y];
                    red_moves.push_back(JuMoves[i]);
                }
                else {
                    JuMoves[i].score = JuPosition[JuMoves[i].next_x][JuMoves[i].next_y] - JuPosition[x][y];
                    black_moves.push_back(JuMoves[i]);
                }
            }
        }

        //生成马的合法动作
        void generateMaMoves(int x, int y, bool color) {
            //便利所有可能动作，筛选
            std::vector<Move> MaMoves;
            int dx[] = {2, 1, -1, -2, -2, -1, 1, 2};
            int dy[] = {1, 2, 2, 1, -1, -2, -2, -1};
            //简化，不考虑拌马脚
            //TODO 可以实现拌马脚过程
            for(int i = 0; i < 8; i++) {
                Move cur_move;
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (nx < 0 || nx >= 9 || ny < 0 || ny >= 10) continue;
                // 处理拌马脚
                // 这里不用考虑数组越界，因为越界的情况已经在上面被处理了
                if (dx[i] == 2 && board[y][x + 1] != '.') continue;
                if (dx[i] == -2 && board[y][x - 1] != '.') continue;
                if (dy[i] == 2 && board[y + 1][x] != '.') continue;
                if (dy[i] == -2 && board[y - 1][x] != '.') continue;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = nx;
                cur_move.next_y = ny;
                cur_move.score = 0;
                if (board[ny][nx] != '.') {
                    //注意棋盘坐标系，这里nx、ny相反是正确的
                    bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                    if (cur_color != color) {
                        MaMoves.push_back(cur_move);
                    }
                    continue;
                }
                MaMoves.push_back(cur_move);
            }
            for (int i = 0; i < MaMoves.size(); i++) {
                if(color) {
                    MaMoves[i].score = MaPosition[MaMoves[i].next_x][9 - MaMoves[i].next_y] - MaPosition[x][9 - y];
                    red_moves.push_back(MaMoves[i]);
                }
                else {
                    MaMoves[i].score = MaPosition[MaMoves[i].next_x][MaMoves[i].next_y] - MaPosition[x][y];
                    black_moves.push_back(MaMoves[i]);
                }
            }
        }

        //生成炮的合法动作
        void generatePaoMoves(int x, int y, bool color) {
            //和车生成动作相似，需要考虑炮翻山吃子的情况
            std::vector<Move> PaoMoves;
            //TODO
            for(int i = x + 1; i < sizeY; i++) {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = i;
                cur_move.next_y = y;
                cur_move.score = 0;
                if (board[y][i] != '.') {
                    if(i == sizeY - 1) break;// 到达边界
                    i++;
                    while(i < sizeY && board[y][i] == '.') {// 接着往下找到第一个不为空的位置
                        i++;
                    }
                    if(i >= sizeY) break;// 到达边界
                    bool cur_color = (board[y][i] >= 'A' && board[y][i] <= 'Z');
                    if (cur_color != color) {// 颜色不一样可以吃子
                        cur_move.next_x = i;
                        PaoMoves.push_back(cur_move);
                    }
                    break;
                }
                PaoMoves.push_back(cur_move);
            }
            for(int i = x - 1; i >= 0; i--) {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = i;
                cur_move.next_y = y;
                cur_move.score = 0;
                if (board[y][i] != '.') {
                    if(i == 0) break;// 到达边界
                    i--;
                    while(i >= 0 && board[y][i] == '.') {// 接着往下找到第一个不为空的位置
                        i--;
                    }
                    if(i < 0) break;// 到达边界
                    bool cur_color = (board[y][i] >= 'A' && board[y][i] <= 'Z');
                    if (cur_color != color) {// 颜色不一样可以吃子
                        cur_move.next_x = i;
                        PaoMoves.push_back(cur_move);
                    }
                    break;
                }
                PaoMoves.push_back(cur_move);
            }
            for(int j = y + 1; j < sizeX; j++) {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = x;
                cur_move.next_y = j;
                cur_move.score = 0;
                if (board[j][x] != '.') {
                    if(j == sizeX - 1) break;// 到达边界
                    j++;
                    while(j < sizeX && board[j][x] == '.') {// 接着往下找到第一个不为空的位置
                        j++;
                    }
                    if(j >= sizeX) break;// 到达边界
                    bool cur_color = (board[j][x] >= 'A' && board[j][x] <= 'Z');
                    if (cur_color != color) {// 颜色不一样可以吃子
                        cur_move.next_y = j;
                        PaoMoves.push_back(cur_move);
                    }
                    break;
                }
                PaoMoves.push_back(cur_move);
            }
            for(int j = y - 1; j >= 0; j--) {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = x;
                cur_move.next_y = j;
                cur_move.score = 0;
                if (board[j][x] != '.') {
                    if(j == 0) break;// 到达边界
                    j--;
                    while(j >= 0 && board[j][x] == '.') {// 接着往下找到第一个不为空的位置
                        j--;
                    }
                    if(j < 0) break;// 到达边界
                    bool cur_color = (board[j][x] >= 'A' && board[j][x] <= 'Z');
                    if (cur_color != color) {// 颜色不一样可以吃子
                        cur_move.next_y = j;
                        PaoMoves.push_back(cur_move);
                    }
                    break;
                }
                PaoMoves.push_back(cur_move);
            }

            for (int i = 0; i < PaoMoves.size(); i++) {
                if(color) {
                    PaoMoves[i].score = PaoPosition[PaoMoves[i].next_x][9 - PaoMoves[i].next_y] - PaoPosition[x][9 -y];
                    red_moves.push_back(PaoMoves[i]);
                }
                else {
                    PaoMoves[i].score = PaoPosition[PaoMoves[i].next_x][PaoMoves[i].next_y] - PaoPosition[x][y];
                    black_moves.push_back(PaoMoves[i]);
                }
            }
        }

        //生成相的合法动作
        void generateXiangMoves(int x, int y, bool color) {
            std::vector<Move> XiangMoves;
            //TODO
            int dx[] = {2, 2, -2, -2};
            int dy[] = {2, -2, 2, -2};
            for(int i = 0; i < 4; i++) {
                Move cur_move;
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (nx < 0 || nx >= 9 || ny < 0 || ny >= 10) continue;
                // 处理过河未过河，红方在棋盘下面，所以红方的相纵坐标只能为5~9，黑方同理
                if (color) {
                    if (ny >= 5) continue;
                } else {
                    if (ny < 5) continue;
                }
                // 处理象眼
                int mid_x = (x + nx) / 2;
                int mid_y = (y + ny) / 2;
                if (board[mid_y][mid_x] != '.') continue;
                
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = nx;
                cur_move.next_y = ny;
                cur_move.score = 0;
                if (board[ny][nx] != '.') {
                    bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                    if (cur_color != color) {
                        XiangMoves.push_back(cur_move);
                    }
                    continue;
                }
                XiangMoves.push_back(cur_move);
            }

            for (int i = 0; i < XiangMoves.size(); i++) {
                if(color) {
                    XiangMoves[i].score = XiangPosition[XiangMoves[i].next_x][9 - XiangMoves[i].next_y] - XiangPosition[x][9 -y];
                    red_moves.push_back(XiangMoves[i]);
                }
                else {
                    XiangMoves[i].score = XiangPosition[XiangMoves[i].next_x][XiangMoves[i].next_y] - XiangPosition[x][y];
                    black_moves.push_back(XiangMoves[i]);
                }
            }
        }

        //生成士的合法动作
        void generateShiMoves(int x, int y, bool color) {
            std::vector<Move> ShiMoves;
            //TODO
            int dx[] = {1, 1, -1, -1};
            int dy[] = {1, -1, 1, -1};
            for(int i = 0; i < 4; i++) {
                Move cur_move;
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (nx < 3 || nx >= 6 || ny < 0 || ny >= 10) continue;
                if(color && ny < 7) continue;
                if(!color && ny > 2) continue;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = nx;
                cur_move.next_y = ny;
                cur_move.score = 0;
                if (board[ny][nx] != '.') {
                    bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                    if (cur_color != color) {
                        ShiMoves.push_back(cur_move);
                    }
                    continue;
                }
                ShiMoves.push_back(cur_move);
            }

            for (int i = 0; i < ShiMoves.size(); i++) {
                if(color) {
                    ShiMoves[i].score = ShiPosition[ShiMoves[i].next_x][9 - ShiMoves[i].next_y] - ShiPosition[x][9 -y];
                    red_moves.push_back(ShiMoves[i]);
                }
                else {
                    ShiMoves[i].score = ShiPosition[ShiMoves[i].next_x][ShiMoves[i].next_y] - ShiPosition[x][y];
                    black_moves.push_back(ShiMoves[i]);
                }
            }
        }

        //生成将的合法动作
        void generateJiangMoves(int x, int y, bool color) {
            std::vector<Move> JiangMoves;
            //TODO
            int dx[] = {1, 0, -1, 0};
            int dy[] = {0, 1, 0, -1};
            for(int i = 0; i < 4; i++) {
                Move cur_move;
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (nx < 3 || nx >= 6 || ny < 0 || ny >= 10) continue;
                if(color && ny < 7) continue;
                if(!color && ny > 2) continue;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = nx;
                cur_move.next_y = ny;
                cur_move.score = 0;
                if (board[ny][nx] != '.') {
                    bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                    if (cur_color != color) {
                        JiangMoves.push_back(cur_move);
                    }
                    continue;
                }
                JiangMoves.push_back(cur_move);
            }
            // std::pair<int, int> black_Jiang_pos;
            // for(int i = 3; i < 6; i++) {
            //     for(int j = 0; j < 3; j++) {
            //         if(board[j][i] == 'k')  black_Jiang_pos = {i, j};
            //     }
            // }
            // // 出于简单起见，如果将帅碰面，就直接飞过去吃掉
            // if(color && x == black_Jiang_pos.first) {
            //     bool block = false;
            //     for(int i = y-1; i > black_Jiang_pos.second; i--) {// 检查将帅之间是否有棋子
            //         if(board[i][x] != '.') {
            //             block = true;
            //             break;
            //         }
            //     }
            //     if(!block) {
            //         Move cur_move;
            //         cur_move.init_x = x;
            //         cur_move.init_y = y;
            //         cur_move.next_x = x;
            //         cur_move.next_y = black_Jiang_pos.second;
            //         cur_move.score = 0;
            //         JiangMoves.push_back(cur_move);
            //     }
            // }
            // std::pair<int, int> red_Jiang_pos;
            // for(int i = 3; i < 6; i++) {
            //     for(int j = 0; j < 3; j++) {
            //         if(board[j][i] == 'K')  red_Jiang_pos = {i, j};
            //     }
            // }
            // if(!color && x == red_Jiang_pos.first) {
            //     bool block = false;
            //     for(int i = y+1; i < red_Jiang_pos.second; i++) {// 检查将帅之间是否有棋子
            //         if(board[i][x] != '.') {
            //             block = true;
            //             break;
            //         }
            //     }
            //     if(!block) {
            //         Move cur_move;
            //         cur_move.init_x = x;
            //         cur_move.init_y = y;
            //         cur_move.next_x = x;
            //         cur_move.next_y = red_Jiang_pos.second;
            //         cur_move.score = 0;
            //         JiangMoves.push_back(cur_move);
            //     }
            // }

            for (int i = 0; i < JiangMoves.size(); i++) {
                if(color) {
                    JiangMoves[i].score = JiangPosition[JiangMoves[i].next_x][9 - JiangMoves[i].next_y] - JiangPosition[x][9 -y];
                    red_moves.push_back(JiangMoves[i]);
                }
                else {
                    JiangMoves[i].score = JiangPosition[JiangMoves[i].next_x][JiangMoves[i].next_y] - JiangPosition[x][y];
                    black_moves.push_back(JiangMoves[i]);
                }
            }
        }

        //生成兵的合法动作
        void generateBingMoves(int x, int y, bool color) {
            //需要分条件考虑，小兵在过楚河汉界之前只能前进，之后可以左右前
            std::vector<Move> BingMoves;
            //TODO
            if(color) {
                if(y >= 5) {// 没过河，只能前进
                    int nx = x;
                    int ny = y - 1;
                    if(ny >= 0) {
                        Move cur_move;
                        cur_move.init_x = x;
                        cur_move.init_y = y;
                        cur_move.next_x = nx;
                        cur_move.next_y = ny;
                        cur_move.score = 0;
                        if(board[ny][nx] == '.') {
                            BingMoves.push_back(cur_move);
                        }
                        else {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if(cur_color != color) {
                                BingMoves.push_back(cur_move);
                            }
                        }
                    }
                }
                else {// 过河了，可以左右前
                    int nx = x;
                    int ny = y - 1;
                    if(ny >= 0) {
                        Move cur_move;
                        cur_move.init_x = x;
                        cur_move.init_y = y;
                        cur_move.next_x = nx;
                        cur_move.next_y = ny;
                        cur_move.score = 0;
                        if(board[ny][nx] == '.') {
                            BingMoves.push_back(cur_move);
                        }
                        else {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if(cur_color != color) {
                                BingMoves.push_back(cur_move);
                            }
                        }
                    }
                    nx = x + 1;
                    ny = y;
                    if(nx < 9) {
                        Move cur_move;
                        cur_move.init_x = x;
                        cur_move.init_y = y;
                        cur_move.next_x = nx;
                        cur_move.next_y = ny;
                        cur_move.score = 0;
                        if(board[ny][nx] == '.') {
                            BingMoves.push_back(cur_move);
                        }
                        else {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if(cur_color != color) {
                                BingMoves.push_back(cur_move);
                            }
                        }
                    }
                    nx = x - 1;
                    ny = y;
                    if(nx >= 0) {
                        Move cur_move;
                        cur_move.init_x = x;
                        cur_move.init_y = y;
                        cur_move.next_x = nx;
                        cur_move.next_y = ny;
                        cur_move.score = 0;
                        if(board[ny][nx] == '.') {
                            BingMoves.push_back(cur_move);
                        }
                        else {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if(cur_color != color) {
                                BingMoves.push_back(cur_move);
                            }
                        }
                    }
                }
            }
            else {
                if(y <= 4) {// 没过河，只能前进
                    int nx = x;
                    int ny = y + 1;
                    Move cur_move;
                    cur_move.init_x = x;
                    cur_move.init_y = y;
                    cur_move.next_x = nx;
                    cur_move.next_y = ny;
                    cur_move.score = 0;
                    if(board[ny][nx] == '.') {
                        BingMoves.push_back(cur_move);
                    }
                    else {
                        bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                        if(cur_color != color) {
                            BingMoves.push_back(cur_move);
                        }
                    }
                }
                else {// 过河了，可以左右前
                    int nx = x;
                    int ny = y + 1;
                    if(ny < 10) {
                        Move cur_move;
                        cur_move.init_x = x;
                        cur_move.init_y = y;
                        cur_move.next_x = nx;
                        cur_move.next_y = ny;
                        cur_move.score = 0;
                        if(board[ny][nx] == '.') {
                            BingMoves.push_back(cur_move);
                        }
                        else {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if(cur_color != color) {
                                BingMoves.push_back(cur_move);
                            }
                        }
                    }
                    nx = x + 1;
                    ny = y;
                    if(nx < 9) {
                        Move cur_move;
                        cur_move.init_x = x;
                        cur_move.init_y = y;
                        cur_move.next_x = nx;
                        cur_move.next_y = ny;
                        cur_move.score = 0;
                        if(board[ny][nx] == '.') {
                            BingMoves.push_back(cur_move);
                        }
                        else {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if(cur_color != color) {
                                BingMoves.push_back(cur_move);
                            }
                        }
                    }
                    nx = x - 1;
                    ny = y;
                    if(nx >= 0) {
                        Move cur_move;
                        cur_move.init_x = x;
                        cur_move.init_y = y;
                        cur_move.next_x = nx;
                        cur_move.next_y = ny;
                        cur_move.score = 0;
                        if(board[ny][nx] == '.') {
                            BingMoves.push_back(cur_move);
                        }
                        else {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if(cur_color != color) {
                                BingMoves.push_back(cur_move);
                            }
                        }
                    }
                }
            }


            for (int i = 0; i < BingMoves.size(); i++) {
                if(color) {
                    BingMoves[i].score = BingPosition[BingMoves[i].next_x][9 - BingMoves[i].next_y] - BingPosition[x][9 -y];
                    red_moves.push_back(BingMoves[i]);
                }
                else {
                    BingMoves[i].score = BingPosition[BingMoves[i].next_x][BingMoves[i].next_y] - BingPosition[x][y];
                    black_moves.push_back(BingMoves[i]);
                }
            }
        }
    
        //终止判断
        bool judgeTermination() {
            //TODO
            return false;
        }

        //棋盘分数评估，根据当前棋盘进行棋子价值和棋力评估，max玩家减去min玩家分数
        int evaluateNode(bool now_color) {
            //TODO
            int red_score = 0;
            int black_score = 0;
            auto red_moves = getMoves(true);
            auto black_moves = getMoves(false);
            bool red_Jiang = false;
            bool black_Jiang = false;
            std::pair<int, int> red_Jiang_pos;
            std::pair<int, int> black_Jiang_pos;
            // 计算棋子价值和位置价值
            for(auto piece: pieces) {
                auto piece_name = getPieceName(piece.name);
                auto piece_char = piece.name;
                if(piece_char == 'k')    {
                    black_Jiang_pos = {piece.init_x, piece.init_y};
                    black_Jiang = true;
                }
                if(piece_char == 'K')    {
                    red_Jiang_pos = {piece.init_x, piece.init_y};
                    red_Jiang = true;
                }
                if(piece.color) {
                    red_score += piece_values[piece_name];
                    switch (piece_char) {
                        case 'R':
                            red_score += JuPosition[piece.init_x][9 - piece.init_y];
                            break;
                        case 'C':
                            red_score += PaoPosition[piece.init_x][9 - piece.init_y];
                            break;
                        case 'N':
                            red_score += MaPosition[piece.init_x][9 - piece.init_y];
                            break;
                        case 'B':
                            red_score += XiangPosition[piece.init_x][9 - piece.init_y];
                            break;
                        case 'A':
                            red_score += ShiPosition[piece.init_x][9 - piece.init_y];
                            break;
                        case 'K':
                            red_score += JiangPosition[piece.init_x][9 - piece.init_y];
                            break;
                        case 'P':
                            red_score += BingPosition[piece.init_x][9 - piece.init_y];
                            break;
                        default:
                            break;
                    }
                }
                else {
                    black_score += piece_values[piece_name];
                    switch (piece_char) {
                        case 'r':
                            black_score += JuPosition[piece.init_x][piece.init_y];
                            break;
                        case 'c':
                            black_score += PaoPosition[piece.init_x][piece.init_y];
                            break;
                        case 'n':
                            black_score += MaPosition[piece.init_x][piece.init_y];
                            break;
                        case 'b':
                            black_score += XiangPosition[piece.init_x][piece.init_y];
                            break;
                        case 'a':
                            black_score += ShiPosition[piece.init_x][piece.init_y];
                            break;
                        case 'k':
                            black_score += JiangPosition[piece.init_x][piece.init_y];
                            break;
                        case 'p':
                            black_score += BingPosition[piece.init_x][piece.init_y];
                            break;
                        default:
                            break;
                    }
                }
            }

            // for(int j = 0; j < 10; j++) {
            //     for(int i = 0; i < 9; i++) {
            //         if(board[j][i] == '.') continue;
            //         char pieceChar = board[j][i];
            //         bool color = (pieceChar >= 'A' && pieceChar <= 'Z');
            //         std::string piece_name = getPieceName(pieceChar);
            //         if(pieceChar == 'k')    {
            //             black_Jiang_pos = {i, j};
            //             black_Jiang = true;
            //         }
            //         if(pieceChar == 'K')    {
            //             red_Jiang_pos = {i, j};
            //             red_Jiang = true;
            //         }
            //         if(color) {
            //             red_score += piece_values[piece_name];
            //             switch (pieceChar) {
            //                 case 'R':
            //                     red_score += JuPosition[i][9 - j];
            //                     break;
            //                 case 'C':
            //                     red_score += PaoPosition[i][9 - j];
            //                     break;
            //                 case 'N':
            //                     red_score += MaPosition[i][9 - j];
            //                     break;
            //                 case 'B':
            //                     red_score += XiangPosition[i][9 - j];
            //                     break;
            //                 case 'A':
            //                     red_score += ShiPosition[i][9 - j];
            //                     break;
            //                 case 'K':
            //                     red_score += JiangPosition[i][9 - j];
            //                     red_Jiang = true;
            //                     break;
            //                 case 'P':
            //                     red_score += BingPosition[i][9 - j];
            //                     break;
            //                 default:
            //                     break;
            //             }
            //         }
            //         else {
            //             black_score += piece_values[piece_name];
            //             switch (pieceChar) {
            //                 case 'r':
            //                     black_score += JuPosition[i][j];
            //                     break;
            //                 case 'c':
            //                     black_score += PaoPosition[i][j];
            //                     break;
            //                 case 'n':
            //                     black_score += MaPosition[i][j];
            //                     break;
            //                 case 'b':
            //                     black_score += XiangPosition[i][j];
            //                     break;
            //                 case 'a':
            //                     black_score += ShiPosition[i][j];
            //                     break;
            //                 case 'k':
            //                     black_score += JiangPosition[i][j];
            //                     black_Jiang = true;
            //                     break;
            //                 case 'p':
            //                     black_score += BingPosition[i][j];
            //                     break;
            //                 default:
            //                     break;
            //             }
            //         }
                
            //     }
            // }
            if(!red_Jiang) return std::numeric_limits<int>::min()+1; // 如果红方帅被吃，黑胜
            if(!black_Jiang) return std::numeric_limits<int>::max()-1; // 如果黑方将被吃，红胜
            
            if(red_Jiang_pos.first == black_Jiang_pos.first) {
                auto block = false;
                for(int i = black_Jiang_pos.second+1; i < red_Jiang_pos.second; i++) {
                    if(board[i][red_Jiang_pos.first] != '.') {
                        block = true;
                        break;
                    }
                }
                if(!block) {
                    if(now_color) return std::numeric_limits<int>::max()-1; // 红方可以吃掉黑方帅，红胜
                    else return std::numeric_limits<int>::min()+1; // 黑方可以吃掉红方帅，黑胜
                }
            }
            std::vector<std::pair<int, int>> red_check;// 红方能够将军的位置
            std::vector<std::pair<int, int>> black_check;// 黑方能够将军的位置
            // 计算红方下一步吃子的收益
            for(int i = 0; i < red_moves.size(); i++) {
                if(board[red_moves[i].next_y][red_moves[i].next_x] == '.') continue;
                auto piece_name = getPieceName(board[red_moves[i].next_y][red_moves[i].next_x]);
                // 下一步红方动，且可以直接吃将，因此已经赢下棋局
                if(now_color && piece_name == "Jiang") {
                    // 这里少加一点是为了正确更新best_move
                    return std::numeric_limits<int>::max()-1;
                }
                // 否则计算红方下一步吃子的收益
                if(next_move_values.find(piece_name) == next_move_values.end()) continue;
                red_score += next_move_values[piece_name];
                // 对能够将军的棋子位置做特别记录
                if(piece_name == "Jiang") {
                    red_check.push_back(std::make_pair(red_moves[i].init_x, red_moves[i].init_y));
                }
            }
            // 计算黑方下一步吃子的收益
            for(int i = 0; i < black_moves.size(); i++) {
                if(board[black_moves[i].next_y][black_moves[i].next_x] == '.') continue;
                auto piece_name = getPieceName(board[black_moves[i].next_y][black_moves[i].next_x]);
                // 下一步黑方动，且可以直接吃将，因此已经赢下棋局
                if(!now_color && piece_name == "Jiang") {
                    return std::numeric_limits<int>::min()+1;
                }
                // 否则计算黑方下一步吃子的收益
                if(next_move_values.find(piece_name) == next_move_values.end()) continue;
                black_score += next_move_values[piece_name];
                // 对能够将军的棋子位置做特别记录
                if(piece_name == "Jiang") {
                    black_check.push_back(std::make_pair(black_moves[i].init_x, black_moves[i].init_y));
                }
            }
            // 如果现在是红方动，则需要检查是否可以威胁到黑方正在将军的棋子
            // 如果黑方正在将军的棋子只有一个，且能够将其吃掉，则实际上黑方这一将威胁不大
            // 否则红方必须转向防守，黑方优势较大
            // 这里只对将军的情况进行处理的原因是，对于其它棋子，能够将其吃掉可以看作是对它进攻能力的一种牵制，
            // 因此可以不作这么复杂的处理。而将帅本身进攻能力不强，能够将将帅吃掉代表的是对胜局的影响，必须谨慎赋予分数
            // if(now_color && black_check.size() == 1) {
            //     for(int i = 0; i < red_moves.size(); i++) {
            //         if(red_moves[i].next_x == black_check[0].first && red_moves[i].next_y == black_check[0].second) {
            //             black_score -= next_move_values["Jiang"];
            //             black_score += 500; // 威胁不大，但不是完全没有作用。这一数值可以调整
            //             break;
            //         }
            //     }
            // }
            // else if(!now_color && red_check.size() == 1) {  // 黑方动时同理
            //     for(int i = 0; i < black_moves.size(); i++) {
            //         if(black_moves[i].next_x == red_check[0].first && black_moves[i].next_y == red_check[0].second) {
            //             red_score -= next_move_values["Jiang"];
            //             red_score += 500; // 威胁不大，但不是完全没有作用。这一数值可以调整
            //             break;
            //         }
            //     }
            // }
            
            return red_score - black_score;
        }

        //测试接口
        std::vector<Move> getMoves(bool color) {
            if(color) return red_moves;
            return black_moves;
        }

        void setMoveScore(bool color, int i, int score) {
            if(i < 0)   return;
            if(color) {
                if(i < red_moves.size())
                    red_moves[i].score = score;
            }
            else {
                if(i < black_moves.size())
                    black_moves[i].score = score;
            }
        }

        void removeMove(bool color, int i) {
            if(i < 0)   return;
            if(color) {
                if(i < red_moves.size())
                    red_moves.erase(red_moves.begin() + i);
            }
            else {
                if(i < black_moves.size())
                    black_moves.erase(black_moves.begin() + i);
            }
        }
    
        std::vector<ChessPiece> getChessPiece() {
            return pieces;
        }
    
        std::vector<std::vector<char>> getBoard() {
            return board;
        }
    };

    // 定义博弈树节点类
    class GameTreeNode {
    private:
        bool color; // 当前玩家类型，true为红色方、false为黑色方
                    // 这里的color表示在当前状态下，下一步应该谁动
        ChessBoard board; // 当前棋盘状态
        Move best_move; // 当前节点的最佳动作
        std::vector<GameTreeNode*> children; // 子节点列表
        int evaluationScore; // 棋盘评估分数
        bool evaluated; // 是否已经评估过

    public:
        // 构造函数
        GameTreeNode(bool color, std::vector<std::vector<char>> &initBoard, int evaluationScore)
            : color(color), evaluationScore(evaluationScore) {
            board.initializeBoard(initBoard);
            // std::vector<Move> moves = board.getMoves(color);
            children.clear();
            // std::vector<std::vector<char>> cur_board = board.getBoard();
            evaluated = false;

            // 为合法动作创建子节点
            // for (int i = 0; i < moves.size(); i++) {
            //     GameTreeNode* child = updateBoard(cur_board, moves[i], color);
            //     children.push_back(child);
            // }
        }

        //根据当前棋盘和动作构建新棋盘（子节点）
        GameTreeNode* updateBoard(Move move) {
            //TODO
            GameTreeNode* test;
            auto cur_board = board.getBoard();
            std::vector<std::vector<char>> new_board = cur_board;
            new_board[move.next_y][move.next_x] = new_board[move.init_y][move.init_x];
            new_board[move.init_y][move.init_x] = '.';
            test = new GameTreeNode(!color, new_board, 0);
            // children.push_back(test);
            return test;
        }

        //返回节点评估分数
        int getEvaluationScore() {
            if(evaluated) return evaluationScore;
            evaluationScore = board.evaluateNode(color);
            return evaluationScore;
        }

        int haveJiang() {   // 检查当前情况将是否还存活
            bool red_Jiang = false;
            bool black_Jiang = false;
            auto cur_board = board.getBoard();
            for(int j = 0; j < 10; j++) {
                for(int i = 0; i < 9; i++) {
                    if(cur_board[j][i] == '.') continue;
                    char pieceChar = cur_board[j][i];
                    if(pieceChar == 'k')    black_Jiang = true;
                    if(pieceChar == 'K')    red_Jiang = true;
                }
            }
            if(red_Jiang) {
                if(black_Jiang) {
                    return 0;   // 都存活
                }
                return 1;   // 红方存活
            }
            if(black_Jiang) {
                return -1;  // 黑方存活
            }
            return 0;   // 都死亡（实际不可能出现）
        }

        // 检查将帅是否见面
        bool checkJiangMeet() {
            auto cur_board = board.getBoard();
            std::pair<int, int> red_Jiang_pos;
            std::pair<int, int> black_Jiang_pos;
            for(int i = 3; i < 6; i++) {
                for(int j = 0; j < 3; j++) {
                    if(cur_board[j][i] == 'k')  black_Jiang_pos = {i, j};
                }
            }
            for(int i = 3; i < 6; i++) {
                for(int j = 7; j < 10; j++) {
                    if(cur_board[j][i] == 'K')  red_Jiang_pos = {i, j};
                }
            }
            if(red_Jiang_pos.first == black_Jiang_pos.first) {
                auto block = false;
                for(int i = black_Jiang_pos.second+1; i < red_Jiang_pos.second; i++) {
                    if(cur_board[i][red_Jiang_pos.first] != '.') {
                        block = true;
                        break;
                    }
                }
                if(!block) {
                    return true;
                }
            }
            return false;
        }

        //返回棋盘类
        ChessBoard getBoardClass() {
            return board;
        }

        void setBestMove(Move move) {
            best_move = move;
        }

        Move getBestMove() {
            return best_move;
        }

        void removeMove(bool color, Move move) {
            auto moves = board.getMoves(color);
            for(int i = 0; i < moves.size(); i++) {
                if(moves[i].init_x == move.init_x && moves[i].init_y == move.init_y && moves[i].next_x == move.next_x && moves[i].next_y == move.next_y) {
                    board.removeMove(color, i);
                    break;
                }
            }
        }

        void removeBestMove() {
            auto moves = board.getMoves(color);
            for(int i = 0; i < moves.size(); i++) {
                if(moves[i].init_x == best_move.init_x && moves[i].init_y == best_move.init_y && moves[i].next_x == best_move.next_x && moves[i].next_y == best_move.next_y) {
                    board.removeMove(color, i);
                    break;
                }
            }
        }

        std::vector<char> getPieceChar() {
            auto pieces = board.getChessPiece();
            std::vector<char> pieceChar;
            for(auto piece: pieces) {
                pieceChar.push_back(piece.name);
            }
            return pieceChar;
        }
        
        ~GameTreeNode() {
            for (GameTreeNode* child : children) {
                delete child;
            }
        }

    };
    
}