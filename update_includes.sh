#!/bin/bash

# Update include paths in source files
find src -name "*.c" -type f -exec sed -i '' \
    -e 's|"model.h"|"core/model.h"|g' \
    -e 's|"dataset.h"|"core/dataset.h"|g' \
    -e 's|"decision_tree.h"|"ml/decision_tree.h"|g' \
    -e 's|"knn.h"|"ml/knn.h"|g' \
    -e 's|"svm.h"|"ml/svm.h"|g' \
    -e 's|"gradient_boosting.h"|"ml/gradient_boosting.h"|g' \
    -e 's|"random_forest.h"|"ml/random_forest.h"|g' \
    -e 's|"matrix.h"|"utils/matrix.h"|g' \
    -e 's|"preprocess.h"|"utils/preprocess.h"|g' \
    -e 's|"utils.h"|"utils/utils.h"|g' \
    -e 's|"cross_validation.h"|"utils/cross_validation.h"|g' \
    -e 's|"bagging.h"|"ensemble/bagging.h"|g' \
    -e 's|"ensemble_methods.h"|"ensemble/ensemble_methods.h"|g' \
    {} +

# Update include paths in header files
find include -name "*.h" -type f -exec sed -i '' \
    -e 's|"model.h"|"core/model.h"|g' \
    -e 's|"dataset.h"|"core/dataset.h"|g' \
    -e 's|"decision_tree.h"|"ml/decision_tree.h"|g' \
    -e 's|"knn.h"|"ml/knn.h"|g' \
    -e 's|"svm.h"|"ml/svm.h"|g' \
    -e 's|"gradient_boosting.h"|"ml/gradient_boosting.h"|g' \
    -e 's|"random_forest.h"|"ml/random_forest.h"|g' \
    -e 's|"matrix.h"|"utils/matrix.h"|g' \
    -e 's|"preprocess.h"|"utils/preprocess.h"|g' \
    -e 's|"utils.h"|"utils/utils.h"|g' \
    -e 's|"cross_validation.h"|"utils/cross_validation.h"|g' \
    -e 's|"bagging.h"|"ensemble/bagging.h"|g' \
    -e 's|"ensemble_methods.h"|"ensemble/ensemble_methods.h"|g' \
    {} +
