#ifndef CROSS_VALIDATION_H
#define CROSS_VALIDATION_H

#include "core/model.h"
#include "core/dataset.h"

void cross_validation(Model *model, Dataset *dataset, int k);

#endif
