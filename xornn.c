#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "forwarddecs.h"

// xor stuff
static const double INPUT[4][2] = { {0.0, 0.0}, // f
                        {0.0, 1.0}, // t
                        {1.0, 0.0}, // t
                        {1.0, 1.0} }; // f

// truth table outputs of XOR - aligns with above inputs
static const double EXPECTED[4] = {0.0, 
                                1.0, 
                                1.0, 
                                0.0};

// architecture: 2 input neurons -> 2 hidden neurons -> 1 output neuron

// initialize neurons
static double g_w1[2][2]; // 2 source (input) neurons -> 2 destination (hidden layer) neurons
static double g_b1[2]; // 2 bias one per each neuron in the hidden layer
static double g_w2[2]; // connecting hidden to output, 2 hidden layer input neurons to 1 output.
static double g_b2; // output bias

// other vars
static double g_hidden[2]; // storage for our two outputs from the hidden layer
static double g_prediction; // nn prediction
static double g_deltaHidden[2]; // needed for backprop
static const double LEARNING_RATE = 0.5;

int main(void) {
    srand(time(NULL));
    initialize();

    int epochs = 500000; // 500k is good balance of reasonable and good prediction
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < 4; i++) {           // one pass over all 4 examples
            forwardProp(INPUT[i][0], INPUT[i][1]);
            backProp(i);
        }
        if (epoch % 1000 == 0) {                 // every 1000 epochs, report
            printf("epoch %d - loss: %.4f\n", epoch, computeLoss());
        }
    }

    printf("\ntraining done\npredictions:\n");
    for (int i = 0; i < 4; i++) {
        forwardProp(INPUT[i][0], INPUT[i][1]);
        printf("[%.0lf, %.0lf] -> %.3f (rounds to %d)\n", INPUT[i][0], INPUT[i][1], g_prediction, (int)(g_prediction + 0.5));
    }

    return 0;
}

/*
 * Squashes any real number into the range (0, 1) via the sigmoid curve.
 *
 * @param input  the raw pre-activation value
 * @return       the activated output in range (0, 1)
 */
double sigmoid(double input) { 
    return (1.0 / (1.0 + exp(-input))); // exact sigmoid formula is f(x) = 1 / 1 + e^(-x)
}

/*
* Initializes weights and bias'
*/
void initialize(void) {
    // init W1
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            g_w1[i][j] = ((double)rand()/ RAND_MAX) * 2.0 - 1.0; // formula to seed within bounds of -1 to 1, cast to a double to avoid integer division
        }
    }
    // init b1
    for(int i = 0; i < 2; i++) {
        g_b1[i] = ((double)rand()/ RAND_MAX) * 2.0 - 1.0; 
    }
    // init W2
    for(int i = 0; i < 2; i++) {
        g_w2[i] = ((double)rand()/ RAND_MAX) * 2.0 - 1.0; 
    }
    // init b2
    g_b2 = ((double)rand()/ RAND_MAX) * 2.0 - 1.0; 
}

/*
* Sends initial XOR combination (from INPUT) through the forward propigation layer
* @param double x The first operand of the XOR input
* @param double y The second operand of the XOR input
*/
void forwardProp(double x, double y) {

    double currInputs[2] = {x, y}; 
    // sends inputs -> hidden layer
    for(int j = 0; j < 2; j++) { // hidden neuron index
        double sum = 0.0;
        for(int i = 0; i < 2; i++) {
            sum += currInputs[i] * g_w1[i][j]; // dot product
        }
        g_hidden[j] = sigmoid(sum + g_b1[j]); // add bias and apply sigmoid function
    }
    // sends hidden layer -> output neuron
    double sum = 0.0; // fresh sum var
    for(int k = 0; k < 2; k++) {
        sum += g_hidden[k] * g_w2[k]; // dot product
    }
    g_prediction = sigmoid(sum + g_b2); // add output bias and squash into final prediction
}


/*
 * Computes average squared error across all 4 XOR combos
 * @return  average squared loss (0.0 = perfect, higher = worse)
 */
double computeLoss(void) {
    double errormargin = 0.0; // how far off we were from expected output
    for(int i = 0; i < 4; i++) {
        forwardProp(INPUT[i][0], INPUT[i][1]); //run forw prop for every combo
        double diff = (EXPECTED[i] - g_prediction);  // difference is our expected 1 or 0 - the prediction
        errormargin += diff * diff; 
        // why do we square? 1. removes all negative nums (cant have a neg error) 2. punishes large misses harder (a 50% miss is alot worse than a 10% miss)
        // 3. most importantly makes for an easy derivative in backprop
    }
    return errormargin / 4.0;
}

/*
 * Runs one backpropagation pass, computing gradients and updating all weights and biases.
 *
 * @param i  Index into INPUT / EXPECTED for the current training example
 */
void backProp(int i) {

    // dLoss is a derivative of the error margin with respect to the prediction
    // dPred is a derivative of the sigmoid function with respect to the raw pre-sigmoid pred value
    // deltaOut: the output neuron's error signal (its "blame") / how the loss responds
    // to the output's pre-sigmoid sum. we use the chain rule for derivatives to calculate this value
    // d(loss)/dx = d(loss)/d(pred) * d(pred)/dx
    // (combines nicely into dloss * dpred)
    

    double dLoss = 2.0 * (g_prediction - EXPECTED[i]); // how wrong as final answer? pos = guessed too high neg = guessed too low
    double dPred = g_prediction * (1.0 - g_prediction); // how sensitive the output was, if prediction was slightly diff, would it matter?
    double deltaOut = dLoss * dPred; // combine into one blame score

    // we share the blame backwards towards each of the hidden neurons.
    // this layer aims to solve: how does the loss change as the output of the pre sigmoid value of the neuron changes?
    // before simplification, this is a large/messy derivative
    // the basic formula is: 1. our blame score * 2. how much the output sum changes when hidden neuron output changes * 3. the sigmoids slope at hidden neuron x.
    //                 1          2               3       
    g_deltaHidden[0] = deltaOut * g_w2[0] * g_hidden[0] * (1.0 - g_hidden[0]);
    g_deltaHidden[1] = deltaOut * g_w2[1] * g_hidden[1] * (1.0 - g_hidden[1]);

    // nudge each weight against its gradient (the rate of change of the loss with respect to the weight)
    // use subtraction because the gradient points in the direction of error- we want to go the other direction
    // learning rate is the factor at which we nudge the weights. higher = faster learning but can overshoot lower = slower learning but more precise
    g_w2[0] -= LEARNING_RATE * (deltaOut * g_hidden[0]);   
    g_w2[1] -= LEARNING_RATE * (deltaOut * g_hidden[1]);  
    g_b2 -= LEARNING_RATE * (deltaOut);                  

    // same gradient descent step as above, just need loops since w1 is a 2x2 array
    for(int k = 0; k < 2; k++) {       // which input (1 or 2)
        for(int j = 0; j < 2; j++) {   // which hidden neuron (1 or 2)
            g_w1[k][j] -= LEARNING_RATE * g_deltaHidden[j] * INPUT[i][k];
        }
    }
    
    // adjust the hidden layer bias
    g_b1[0] -= LEARNING_RATE * g_deltaHidden[0];
    g_b1[1] -= LEARNING_RATE * g_deltaHidden[1];
}

