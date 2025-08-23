#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <math.h>

//float in range -1 to 1
#define randDouble ((double)rand()/RAND_MAX*2.0-1.0)
//int from 0 to x-1
#define randInt(x) ((int)rand()%(x))

//game values

#define boardHeight 6
#define boardWidth 7

//network values

#define winners 100
#define children 4
#define agents (winners*children+winners) //should be even
#define games (agents/2)
#define fightsPerAgent 3
#define inputs (boardHeight*boardWidth) //board
#define nodes 32 //sqrt(input layer nodes * output layer nodes) = 17.1464281995...
#define outputs boardWidth
#define totalWeights (inputs*nodes+nodes*outputs)
#define mutations ((int)(totalWeights*0.02)) //0.02 mutation rate (2% of weights)


//structs

struct agent {
    double weights[totalWeights];
    unsigned int score;
};

struct game {
    int board[boardHeight][boardWidth];
    int p1;
    int p2;
};


//general functions

void displayProgressBar(int progress, int total) {
    int barWidth = 20;
    float ratio = (float)progress / (float)total;
    int filledLength = (int)(barWidth * ratio);
    int unfilledLength = barWidth - filledLength;

    printf("\r[");
    for (int i = 0; i < filledLength; i++) {
        printf("#");
    }
    for (int i = 0; i < unfilledLength; i++) {
        printf(" ");
    }
    printf("] %.2f%%", ratio * 100);
    fflush(stdout);
}

//randomizes array with double values between -1 and 1
void randArray(double array[], int len) {
    for (int i = 0; i < len; i++) {
        array[i] = randDouble;
    }
}

void scaledRandArray(double array[], int len) {
    int numOfInputs;
    for (int i = 0; i < len; i++) {
        if (i < inputs * nodes) { // input to hidden layer
            numOfInputs = inputs;
        } else { // hidden to output layer
            numOfInputs = nodes;
        }
        // scale the random number by the sqrt of the number of inputs
        array[i] = randDouble * sqrt(1.0 / numOfInputs);
    }
}

void shuffleAgents(struct agent array[], int n) {
    for (int i = n - 1; i > 0; i--) {
        // Pick a random index from 0 to i
        int j = randInt(i + 1);

        // Swap element i with element j
        struct agent temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

int compareAgents(const void *a, const void *b) { //For qsort
    return (((struct agent *)b)->score - ((struct agent *)a)->score);
}

//return the index of the highest value in the array
int highestValue(double array[], int len, int board[boardHeight][boardWidth]) {
    double highest = -INFINITY;
    int index = -1;
    for (int i = 0; i < len; i++) {
        //set value to highest unless board is full in that column
        // highest = (array[i]>highest&&!board[0][i])?array[i],index=i:highest;
        if (array[i] > highest && !board[0][i]) {
            highest = array[i];
            index = i;
        }
    }
    return index;
}


//network functions

//evaluate the network
void evaluateNetwork(double input[], double weights[], double output[]) {
    double hidden[nodes];

    // input to hidden layer
    for (int i = 0; i < nodes; i++) {
        hidden[i] = 0;
        for (int j = 0; j < inputs; j++) {
            hidden[i] += input[j] * weights[i * inputs + j];
        }
        hidden[i] = tanh(hidden[i]);
    }

    // hidden to output layer
    for (int i = 0; i < outputs; i++) {
        output[i] = 0;
        for (int j = 0; j < nodes; j++) {
            output[i] += hidden[j] * weights[inputs * nodes + i * nodes + j];
        }
    }
}


//game functions

//turns board into input
void boardToInput(int board[boardHeight][boardWidth], double input[inputs], int currentPlayer) {
    for (int i = 0; i < boardHeight; i++) {
        for (int j = 0; j < boardWidth; j++) {
            if (board[i][j] == currentPlayer) {
                input[i*boardWidth+j] = 1;
            } else if (board[i][j] != 0) { // opponent's piece
                input[i*boardWidth+j] = -1;
            } else { // empty
                input[i*boardWidth+j] = 0;
            }
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
                    //diagonal down-right
                    if (board[i][j] == player && board[i+1][j+1] == player && board[i+2][j+2] == player && board[i+3][j+3] == player) {
                        return true;
                    }
                }
                if (j>=3) {
                    //diagonal down-left
                    if (board[i][j] == player && board[i+1][j-1] == player && board[i+2][j-2] == player && board[i+3][j-3] == player) {
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
    printf("ai.exe [rounds] [starting percent] [stop percent] [-v/-s]\n  starting percent  The percentage that the randomness starts at (0-100)\n  stop percent  The percent through the rounds that the randomness reaches 0 (0-100)\n  -v  Displays debug information\n  -s  Does not print anything to screen\n");
    exit(EINVAL);
}

int main(int argc, char const *argv[]) {
    srand(time(0)); //initialize random
    double tempOutput[outputs]; //output for storing results
    double tempInput[inputs]; //inputs for evaluating network
    int tempVar;

    unsigned int rounds, startPercent, stopPercent;
    bool verbose = false, silent = false;
    double percent;

    //Get user values
    if (argc > 3) {if (!(rounds = atoi(argv[1]))) help();} else {help();}
    if ((startPercent = atoi(argv[2]))<0||startPercent>100) help();
    if ((stopPercent = atoi(argv[3]))<0||stopPercent>100) help();
    if (argc > 4) {
        if (argv[4][0]=='-'&&argv[4][1]=='v') {
            verbose = true;
        } else if (argv[4][0]=='-'&&argv[4][1]=='s') {
            silent = true;
        } else {
            help();
        }
    }
    
    //setup agents
    struct agent* agentList = malloc(agents * sizeof(struct agent));
    if (!agentList) {
        fprintf(stderr, "Failed to allocate agentList\n");
        exit(ENOMEM);
    }
    for (int i = 0; i < agents; i++) {
        // randArray(agentList[i].weights, totalWeights);
        scaledRandArray(agentList[i].weights, totalWeights);
    }

    //setup games
    struct game gameState;

    //Start of rounds
    for (int round = 0; round < rounds; round++) {
        if (!verbose&&!silent) displayProgressBar(round, rounds);
        if (verbose) printf("Round %i start\n", round);
        percent = ((-(double)startPercent/((double)rounds*((double)stopPercent/(double)100)))*(double)round+startPercent); //Set percentage for being random

        for (int i = 0; i < agents; i++) {
            agentList[i].score = 0;
        }

        for (int gameRound = 0; gameRound < fightsPerAgent; gameRound++) {
            if (verbose) printf("Game round %i start\n", gameRound);

            //shuffle agents for new pairings
            shuffleAgents(agentList, agents);

            for (int i = 0; i < games; i++) {
                //Reset game state
                for (int j = 0; j < boardHeight; j++) memset(gameState.board[j], 0, boardWidth * sizeof(int));
                gameState.p1 = i*2;
                gameState.p2 = i*2+1;

                if (verbose) printf("Game %i start\n", i);
                while(true) { //Run until a win or tie happens
                    if (verbose) printf("Start of round of game\n");

                    if (fullColumns(gameState.board)==boardWidth) { //Check if all tiles are full (tie)
                        agentList[gameState.p1].score += 1;
                        agentList[gameState.p2].score += 1;
                        break;
                    }

                    for (int player = 1; player < 2; player++) {

                        if (randInt(101) >= percent) { //Decide to be random or not
                            boardToInput(gameState.board, tempInput, player);
                            evaluateNetwork(tempInput, agentList[gameState.p1].weights, tempOutput);
                            if (verbose) for (int j = 0; j < outputs; j++) {printf("%f\n",tempOutput[j]);}
                            tempVar = highestValue(tempOutput, boardWidth, gameState.board);
                        } else { //Random
                            tempVar = randInt(boardWidth);
                            while (xToY(gameState.board, tempVar) == -1) {
                                tempVar = ++tempVar%boardWidth;
                            }
                        }

                        if (verbose) printf("%i\n",tempVar);
                        if (xToY(gameState.board, tempVar) == -1) {
                            printf("Full Column Error");
                            exit(ERANGE);
                        }

                        gameState.board[xToY(gameState.board, tempVar)][tempVar] = player;
                        if (checkWin(gameState.board, player)) {
                            if (player == 1) {
                                agentList[gameState.p1].score += 3;
                            } else {
                                agentList[gameState.p2].score += 3;
                            }
                            break;
                        }

                    }

                    if (verbose) {
                        for (int j = 0; j < boardHeight; j++) {
                            for (int k = 0; k < boardWidth; k++) {
                                printf("%i",gameState.board[j][k]);
                            }
                            printf("\n");
                        }
                    }
                }
            }
        }
        if (verbose) printf("Start Sorting\n");

        //sort agents to get winners
        qsort(agentList, agents, sizeof(struct agent), compareAgents);

        // winners reproduce
        for (int i = winners; i < agents; i++) {
            // pick a random winner to be the parent
            int parentIndex = randInt(winners);
            
            // copy the parent's weights to the child
            memcpy(agentList[i].weights, agentList[parentIndex].weights, totalWeights * sizeof(double));

            
            for (int j = 0; j < mutations; j++) {
                int mutationIndex = randInt(totalWeights);
                
                // agentList[i].weights[mutationIndex] = randDouble;
                
                agentList[i].weights[mutationIndex] += randDouble * 0.2;

                if (agentList[i].weights[mutationIndex] > 4.0) {
                    agentList[i].weights[mutationIndex] = 4.0;
                } else if (agentList[i].weights[mutationIndex] < -4.0) {
                    agentList[i].weights[mutationIndex] = -4.0;
                }
            }
        }

        if (verbose) printf("Game Complete\n");
    }

    printf("\nFinished Traning\n");
    FILE* fptr = fopen("weights.txt", "w");
    if (fptr == NULL) {
        printf("Error opening file");
        exit(ENOENT);
    }
    for (int i = 0; i < (totalWeights); i++) {
        fprintf(fptr, "%f,", agentList[0].weights[i]);
    }
    fclose(fptr);

    return 0;
}

/* Add
When to stop? fitness, time, win average, etc.
Manual exit
*/