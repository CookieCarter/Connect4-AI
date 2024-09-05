#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <windows.h>
//float in range -1 to 1
#define randDouble ((double)rand()/RAND_MAX*2.0-1.0)
//int from 0 to x-1
#define randInt(x) ((int)rand()%(x))
#define listSize(l) (sizeof(l)/sizeof(l[0]))
#define swap(a,b) a^=b; b^=a; a^=b

//game values

#define boardHeight 6
#define boardWidth 7
#define games (agents/2)

//network values

#define winners 10
#define children 1
#define agents (winners*children+winners) //should be even
#define inputs (boardHeight*boardWidth) //board
#define nodes 17 //sqrt(input layer nodes * output layer nodes) = 17.1464281995...
#define outputs boardWidth


//structs

struct agent {
    double weights[inputs*nodes+nodes*outputs];
};

struct game {
    int board[boardHeight][boardWidth];
    int p1;
    int p2;
    int winner;
    int loser;
};


//general functions

//randomizes array with double values between -1 and 1
void randArray(double array[], int len) {
    for (int i = 0; i < len; i++) {
        array[i] = randDouble;
    }
}

//return the index of the highest value in the array
int highestValue(double array[], int len, int board[boardHeight][boardWidth]) {
    double highest = -999999999999;
    int index = -1;
    for (int i = 0; i < len; i++) {
        //set value to highest unless board is full in that column
        highest = (array[i]>highest&&!board[0][i])?array[i],index=i:highest;
        // if (array[i] > highest) {
        //     highest = array[i];
        //     index = i;
        // }
    }
    return index;
}


//network functions

//evaluate the network and return the index of the highest output
void evaluateNetwork(double input[], double weights[], double output[]) {
    double hidden[nodes];
    RtlZeroMemory(hidden, nodes);
    RtlZeroMemory(output, outputs);

    for (int i = 0; i < nodes; i++) {
        for (int j = 0; j < inputs; j++) {
            hidden[i]+= input[j]*weights[i*inputs+j];
        }
        hidden[i] = hidden[i]>0?hidden[i]:0.01*hidden[i]; //leaky relu
    }

    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < nodes; j++) {
            output[i]+= hidden[j]*weights[i*nodes+j];
        }
        output[i] = output[i]>0?output[i]:0.01*output[i]; //leaky relu
    }

    // return highestValue(output, outputs);
}


//game functions

//turns board into input
void boardToInput(int board[boardHeight][boardWidth], double input[inputs]) {
    for (int i = 0; i < boardHeight; i++) {
        for (int j = 0; j < boardWidth; j++) {
            input[i*boardWidth+j] = board[i][j];
        }
    }
}

//returns y cord of next spot in column
int xToY(int board[boardHeight][boardWidth], int column) {
    for (int y = boardHeight-1; y >= 0; y--) {
        if (!board[y][column]) {
            return y;
        }
    }
    return -1; //column full
}

//return number of full columns
int fullColumns(int board[boardHeight][boardWidth]) {
    int fullColumns = 0;
    for (int i = 0; i < boardWidth; i++) {
        if (board[0][i]) {
            fullColumns++;
        }
    }
    return fullColumns;
}

//returns true if player has won
bool checkWin(int board[boardHeight][boardWidth], int player) {
    for (int i = 0; i < boardHeight; i++) {
        for (int j = 0; j < boardWidth; j++) {
            if (i<boardHeight-3) {
                //vertical
                if (board[i][j] == player && board[i+1][j] == player && board[i+2][j] == player && board[i+3][j] == player) {
                    return true;
                }
                if (j<boardWidth-3) {
                    //diagonals
                    if (board[i+3][j] == player && board[i+2][j+1] == player && board[i+1][j+2] == player && board[i][j+3] == player) {
                        return true;
                    }
                    if (board[i][j+3] == player && board[i+1][j+2] == player && board[i+2][j+1] == player && board[i+3][j] == player) {
                    return true;
                    }
                }
            }
            if (j<boardWidth-3) {
                //horizontal
                if (board[i][j] == player && board[i][j+1] == player && board[i][j+2] == player && board[i][j+3] == player) {
                    return true;
                }
            }
        }
    }
    return false;
}

void help(void) {
    printf("ai.exe [rounds] [-v]\n  -v  Displays debug information\n");
    exit(EINVAL);
}

int main(int argc, char const *argv[]) {
    srand(time(0)); //initialize random
    double tempOutput[outputs]; //output for storing results
    double tempInput[inputs]; //inputs for evaluating network
    int tempVar;

    unsigned int rounds;
    bool verbose = false;

    if (argc > 1) {if (!(rounds = atoi(argv[1]))) help();} else {help();}
    if (argc > 2) {
        if (argv[2][0]=='-'&&argv[2][1]=='v') {
            verbose = true;
        } else {
            help();
        }
    }

    //setup agents
    struct agent agentList[agents];
    for (int i = 0; i < agents; i++) {
        randArray(agentList[i].weights, listSize(agentList[i].weights));
    }

    //setup games
    struct game gameList[games];
    for (int i = 0; i < games; i++) {
        for (int j = 0; j < boardHeight; j++) RtlZeroMemory(gameList[i].board[j], boardWidth);
        gameList[i].p1 = i*2;
        gameList[i].p2 = i*2+1;
    }

    // while(true) { //When to stop?
    for (int round = 0; round < rounds; round++) {
        for (int i = 0; i < games; i++) for (int j = 0; j < boardHeight; j++) for (int k = 0; k < boardWidth; k++) gameList[i].board[j][k] = 0;
        printf("Round %i start\n", round);
        for (int i = 0; i < games; i++) {
            printf("Game %i start\n", i);
            while(true) {
                if (verbose) printf("game round start\n");
                boardToInput(gameList[i].board, tempInput);
                evaluateNetwork(tempInput, agentList[gameList[i].p1].weights, tempOutput);
                if (verbose) for (int j = 0; j < outputs; j++) {printf("%f\n",tempOutput[j]);}
                tempVar = highestValue(tempOutput, boardWidth, gameList[i].board);
                if (tempVar==-1&&verbose) {printf("Columns Full Error\n");goto fullColumn;}
                if (verbose) printf("%i\n",tempVar);
                if (xToY(gameList[i].board, tempVar) == -1) {
                    printf("Full Column Error");
                    exit(ERANGE);
                }
                gameList[i].board[xToY(gameList[i].board, tempVar)][tempVar] = 1;
                if (checkWin(gameList[i].board, 1)) {
                    gameList[i].winner = gameList[i].p1;
                    gameList[i].loser = gameList[i].p2;
                    break;
                }

                boardToInput(gameList[i].board, tempInput);
                evaluateNetwork(tempInput, agentList[gameList[i].p2].weights, tempOutput);
                if (verbose) for (int j = 0; j < outputs; j++) {printf("%f\n",tempOutput[j]);}
                tempVar = highestValue(tempOutput, boardWidth, gameList[i].board);
                if (tempVar==-1&&verbose) {printf("Columns Full Error\n");goto fullColumn;}
                if (verbose) printf("%i\n",tempVar);
                if (xToY(gameList[i].board, tempVar) == -1) {
                    printf("Full Column Error");
                    exit(ERANGE);
                }
                gameList[i].board[xToY(gameList[i].board, tempVar)][tempVar] = 2;
                if (checkWin(gameList[i].board, 2)) {
                    gameList[i].winner = gameList[i].p2;
                    gameList[i].loser = gameList[i].p1;
                    break;
                }

                fullColumn:
                if (fullColumns(gameList[i].board)==boardWidth) { //random in case of tie
                    gameList[i].winner = (tempVar = randInt(2))?gameList[i].p1:gameList[i].p2;
                    gameList[i].loser = (tempVar)?gameList[i].p2:gameList[i].p1;
                    break;
                }

                if (verbose) {
                    for (int j = 0; j < boardHeight; j++) {
                        for (int k = 0; k < boardWidth; k++) {
                            printf("%i",gameList[i].board[j][k]);
                        }
                        printf("\n");
                    }
                }
            }
        }
        printf("Start Sorting\n");
        for (int i = 0; i < games; i++) {
            if (i < games/2) {
                gameList[i].p1 = gameList[i*2].winner;
                gameList[i].p2 = gameList[i*2+1].winner;
            } else {
                gameList[i].p1 = gameList[i*2-games].winner;
                agentList[gameList[i].p1].weights[randInt(inputs*nodes+nodes*outputs)] = randDouble; //randomize winner's children
                gameList[i].p2 = gameList[i*2+1-games].winner;
                agentList[gameList[i].p2].weights[randInt(inputs*nodes+nodes*outputs)] = randDouble;
            }
        }
        printf("Game Complete\n");
    }

    printf("Finished Traning\n");
    FILE* fptr = fopen("weights.txt", "w");
    for (int i = 0; i < (inputs*nodes+nodes*outputs); i++) {
        fprintf(fptr, "%f,", agentList[gameList[0].p1].weights[i]);
    }
    fclose(fptr);

    return 0;
}

/* Add
When to stop? fitness, time, win average, etc.
Manual exit
*/