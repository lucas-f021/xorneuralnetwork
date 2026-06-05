#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

//forward decs
double sigmoid(double input);
void debugging(void);
void run(size_t runs);
void initialize(void);
void forwardProp(double x, double y);
double computeLoss(void);
void backProp(int i);

// xor stuff
double inputs[4][2] = { {0.0, 0.0}, // f
                        {0.0, 1.0}, // t
                        {1.0, 0.0}, // t
                        {1.0, 1.0} }; // f

double expected[4] = {0.0, 1.0, 1.0, 0.0};

// architecture: 2 input neurons -> 2 hidden neurons -> 1 output neuron

// weight storage
double W1[2][2]; // 2 source (input) neurons -> 2 destination (hidden layer) neurons
double b1[2]; // 2 bias one per each neuron in the hidden layer
double W2[2]; // connecting hidden to output, 2 hidden layer input neurons to 1 output.
double b2; // output bias

// other vars
double hidden[2]; // storage for our two outputs from the hidden layer
double prediction; // nn prediction
double deltaHidden[2]; // needed for backprop
const double learningRate = 0.5;


double sigmoid(double input) { // squashes any real num into range 0,1 (a nice clean s curve)
    return (1.0 / (1.0 + exp(-input))); // exact sigmoid formula is f(x) = 1 / 1 + e^(-x)
}

void initialize(void) {
    // init W1
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            W1[i][j] = ((double)rand()/ RAND_MAX) * 2.0 - 1.0; // formula to seed within bounds of -1 to 1, cast to a double to avoid integer division
        }
    }
    // init b1
    for(int i = 0; i < 2; i++) {
        b1[i] = ((double)rand()/ RAND_MAX) * 2.0 - 1.0; 
    }
    // init W2
    for(int i = 0; i < 2; i++) {
        W2[i] = ((double)rand()/ RAND_MAX) * 2.0 - 1.0; 
    }
    // init b2
    b2 = ((double)rand()/ RAND_MAX) * 2.0 - 1.0; 
}

void forwardProp(double x, double y) {

    double currInputs[2] = {x, y}; 
    // sends inputs -> hidden layer
    for(int j = 0; j < 2; j++) { // hidden neuron index
        double sum = 0.0;
        for(int i = 0; i < 2; i++) {
            sum += currInputs[i] * W1[i][j]; // dot product
        }
        hidden[j] = sigmoid(sum + b1[j]); // add bias and apply sigmoid function
    }
    // sends hidden layer -> output neuron
    double sum = 0.0; // fresh sum var
    for(int k = 0; k < 2; k++) {
        sum += hidden[k] * W2[k]; // dot product
    }
    prediction = sigmoid(sum + b2); // add output bias and squash into final prediction
}

double computeLoss(void) {
    double errormargin = 0.0; // how far off we were from expected output
    for(int i = 0; i < 4; i++) {
        forwardProp(inputs[i][0], inputs[i][1]); //run forw prop for every combo
        double diff = (expected[i] - prediction);  // difference is our expected 1 or 0 - the prediction
        errormargin += diff * diff; 
        // why do we square? 1. removes all negative nums (cant have a neg error) 2. punishes large misses harder (a 50% miss is alot worse than a 10% miss)
        // 3. most importantly makes for an easy derivative in backprop
    }
    return errormargin / 4.0;
}

void backProp(int i) {

    // dLoss is a derivative of the error margin with respect to the prediction
    // dPred is a derivative of the sigmoid function with respect to the raw pre-sigmoid pred value
    // deltaOut: the output neuron's error signal (its "blame") / how the loss responds
    // to the output's pre-sigmoid sum. we use the chain rule for derivatives to calculate this value
    // d(loss)/dx = d(loss)/d(pred) * d(pred)/dx
    // (combines nicely into dloss * dpred)
    

    double dLoss = 2.0 * (prediction - expected[i]); // how wrong as final answer? pos = guessed too high neg = guessed too low
    double dPred = prediction * (1.0 - prediction); // how sensitive the output was, if prediction was slightly diff, would it matter?
    double deltaOut = dLoss * dPred; // combine into one blame score

    // we share the blame backwards towards each of the hidden neurons.
    // this layer aims to solve: how does the loss change as the output of the pre sigmoid value of the neuron changes?
    // before simplification, this is a large/messy derivative
    // the basic formula is: 1. our blame score * 2. how much the output sum changes when hidden neuron output changes * 3. the sigmoids slope at hidden neuron x.
    //                 1          2               3       
    deltaHidden[0] = deltaOut * W2[0] * hidden[0] * (1.0 - hidden[0]);
    deltaHidden[1] = deltaOut * W2[1] * hidden[1] * (1.0 - hidden[1]);

    // nudge each weight against its gradient (the rate of change of the loss with respect to the weight)
    // use subtraction because the gradient points in the direction of error- we want to go the other direction
    // learning rate is the factor at which we nudge the weights. higher = faster learning but can overshoot lower = slower learning but more precise
    W2[0] -= learningRate * (deltaOut * hidden[0]);   
    W2[1] -= learningRate * (deltaOut * hidden[1]);  
    b2 -= learningRate * (deltaOut);                  

    // same gradient descent step as above, just need loops since w1 is a 2x2 array
    for(int k = 0; k < 2; k++) {       // which input (1 or 2)
        for(int j = 0; j < 2; j++) {   // which hidden neuron (1 or 2)
            W1[k][j] -= learningRate * deltaHidden[j] * inputs[i][k];
        }
    }
    
    // adjust the hidden layer bias
    b1[0] -= learningRate * deltaHidden[0];
    b1[1] -= learningRate * deltaHidden[1];
}

int main(void) {
    srand(time(NULL));
    initialize();

    int epochs = 500000; // 500k is good balance of reasonable and good prediction
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < 4; i++) {           // one pass over all 4 examples
            forwardProp(inputs[i][0], inputs[i][1]);
            backProp(i);
        }
        if (epoch % 1000 == 0) {                 // every 1000 epochs, report
            printf("epoch %d - loss: %.4f\n", epoch, computeLoss());
        }
    }

    printf("\ntraining done\npredictions:\n");
    for (int i = 0; i < 4; i++) {
        forwardProp(inputs[i][0], inputs[i][1]);
        printf("[%.0lf, %.0lf] -> %.3f (rounds to %d)\n", inputs[i][0], inputs[i][1], prediction, (int)(prediction + 0.5));
}

    return 0;
}
