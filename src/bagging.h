#ifndef BAGGING_H
#define BAGGING_H

#include "model.h"
#include "dataset.h"

Model* bagging(Model **models, Dataset *data, int num_models);
void free_bagging_model(Model *model);

#endif
