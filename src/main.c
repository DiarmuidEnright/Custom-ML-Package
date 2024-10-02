#include "svm.h"
#include "knn.h"
#include "decision_tree.h"
#include "dataset.h"
#include "utils.h"
#include <stdio.h>

int main() {

    size_t n_samples, n_features;
    double **X = load_dataset("data.csv", &n_samples, &n_features);
    double *y = NULL;

    double **X_train, **X_test;
    double *y_train, *y_test;
    split_dataset(X, y, n_samples, 0.8, &X_train, &X_test, &y_train, &y_test);

    SVM *svm_model = svm_train(X_train, y_train, n_samples, n_features, 1.0, 0.001, 1000);
    double prediction = svm_predict(svm_model, X_test[0]);
    printf("SVM Prediction: %lf\n", prediction);
    svm_free(svm_model);

    // Other models (k-NN, Decision Tree)
    // Add more code for training and testing other models

    return 0;
}
