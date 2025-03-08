#ifndef ENSEMBLE_METHODS_H
#define ENSEMBLE_METHODS_H

#include "model.h"
#include "dataset.h"

Model* stacking(Model **models, Dataset *dataset, int num_models);

#endif
