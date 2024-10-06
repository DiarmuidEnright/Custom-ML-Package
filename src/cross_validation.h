#ifndef CROSS_VALIDATION_H
#define CROSS_VALIDATION_H

#include "dataset.h"
#include "model.h"

void cross_validation(Model *model, Dataset *data, int k);

#endif
