#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int param1;
    int param2;
    double score;
} GridSearchResult;

double evaluate_model(int param1, int param2) {
    return (double)rand() / RAND_MAX;
}

GridSearchResult grid_search(int param1_values[], int param2_values[], int param1_size, int param2_size) {
    GridSearchResult best_result;
    best_result.score = -1;

    for (int i = 0; i < param1_size; i++) {
        for (int j = 0; j < param2_size; j++) {
            double score = evaluate_model(param1_values[i], param2_values[j]);
            if (score > best_result.score) {
                best_result.param1 = param1_values[i];
                best_result.param2 = param2_values[j];
                best_result.score = score;
            }
        }
    }

    return best_result;
}
