#ifndef ENSEMBLE_METHODS_H
#define ENSEMBLE_METHODS_H

#include "model.h"
#include "dataset.h"

Model* bagging(Model **models, Dataset *data, int num_models);
Model* stacking(Model **models, Dataset *data, int num_models);

#endif
