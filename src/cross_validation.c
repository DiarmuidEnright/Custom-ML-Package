#include "cross_validation.h"
#include "model.h"
#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>

void cross_validation(Model *model, Dataset *data, int k) {
    int fold_size = data->n_samples / k;
    for (int i = 0; i < k; i++) {
        Dataset *train_data, *test_data;
        split_dataset(data->X, data->y, data->n_samples, (double)(fold_size * i), &train_data, &test_data);
        model->train(model, train_data->X, train_data->n_samples, data->n_features);
        double accuracy = model_evaluate(model, test_data->X, test_data->n_samples, data->n_features);
        printf("Fold %d: Accuracy = %.2f\n", i + 1, accuracy);
        free(train_data);
        free(test_data);
    }
}
