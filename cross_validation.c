#include "cross_validation.h"
#include <stdio.h>
#include <stdlib.h>

void cross_validation(Model *model, Dataset *data, int k) {
    int fold_size = data->num_samples / k;
    for (int i = 0; i < k; i++) {
        Dataset *train_data, *test_data;
        split_data(data, i * fold_size, (i + 1) * fold_size, &train_data, &test_data);
        
        model_train(model, train_data);

        double accuracy = model_evaluate(model, test_data);
        printf("Fold %d: Accuracy = %.2f\n", i + 1, accuracy);
    }
}
