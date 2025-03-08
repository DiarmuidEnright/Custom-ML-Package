#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    double *weights_input_hidden;
    double *weights_hidden_output;
    double learning_rate;
} NeuralNetwork;

NeuralNetwork* initialize_network(int input_size, int hidden_size, int output_size, double learning_rate);

void forward(NeuralNetwork *network, double *input, double *hidden, double *output);

#endif
