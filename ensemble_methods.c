#include "ensemble_methods.h"
#include <stdlib.h>

Model* bagging(Model **models, Dataset *data, int num_models) {
    Model *ensemble_model = create_model();
    for (int i = 0; i < num_models; i++) {
        Dataset *sample = bootstrap_sample(data);
        model_train(models[i], sample);
    }
    return ensemble_model;
}

Model* stacking(Model **models, Dataset *data, int num_models) {
    Model *meta_model = create_model();
    for (int i = 0; i < num_models; i++) {
        model_train(models[i], data);
    }
    model_train(meta_model, data);
    return meta_model;
}
