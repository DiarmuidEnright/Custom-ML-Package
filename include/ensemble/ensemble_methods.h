#ifndef ENSEMBLE_METHODS_H
#define ENSEMBLE_METHODS_H

#include "core/model.h"
#include "core/dataset.h"

Model* stacking(Model **models, Dataset *dataset, int num_models);

#endif
