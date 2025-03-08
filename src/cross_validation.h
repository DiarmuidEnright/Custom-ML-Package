#ifndef CROSS_VALIDATION_H
#define CROSS_VALIDATION_H

#include "model.h"
#include "dataset.h"

void cross_validation(Model *model, Dataset *dataset, int k);

#endif
