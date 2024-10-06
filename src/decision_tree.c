#include "decision_tree.h"
#include "model.h"
#include <stdio.h>
#include <float.h>
#include <stdlib.h>

static double gini_index(double **X, double *y, size_t n_samples, size_t *left_indices, size_t left_size, size_t *right_indices, size_t right_size) {
    double gini = 0.0;
    double left_label_count[2] = {0, 0};
    double right_label_count[2] = {0, 0};

    for (size_t i = 0; i < left_size; i++) {
        left_label_count[(int)y[left_indices[i]]]++;
    }

    for (size_t i = 0; i < right_size; i++) {
        right_label_count[(int)y[right_indices[i]]]++;
    }

    double left_prob_0 = left_label_count[0] / left_size;
    double left_prob_1 = left_label_count[1] / left_size;
    double left_gini = 1.0 - (left_prob_0 * left_prob_0 + left_prob_1 * left_prob_1);

    double right_prob_0 = right_label_count[0] / right_size;
    double right_prob_1 = right_label_count[1] / right_size;
    double right_gini = 1.0 - (right_prob_0 * right_prob_0 + right_prob_1 * right_prob_1);

    gini = (left_size / (double)n_samples) * left_gini + (right_size / (double)n_samples) * right_gini;
    return gini;
}

static void split_dataset(double **X, double *y, size_t n_samples, size_t feature_idx, double threshold, size_t **left_indices, size_t *left_size, size_t **right_indices, size_t *right_size) {
    *left_indices = (size_t *)malloc(n_samples * sizeof(size_t));
    *right_indices = (size_t *)malloc(n_samples * sizeof(size_t));
    *left_size = 0;
    *right_size = 0;

    for (size_t i = 0; i < n_samples; i++) {
        if (X[i][feature_idx] < threshold) {
            (*left_indices)[(*left_size)++] = i;
        } else {
            (*right_indices)[(*right_size)++] = i;
        }
    }
}

static void find_best_split(double **X, double *y, size_t n_samples, size_t n_features, size_t *best_feature, double *best_threshold, double *best_gini) {
    *best_gini = DBL_MAX;

    for (size_t feature_idx = 0; feature_idx < n_features; feature_idx++) {
        for (size_t i = 0; i < n_samples; i++) {
            double threshold = X[i][feature_idx];

            size_t *left_indices, *right_indices;
            size_t left_size, right_size;

            split_dataset(X, y, n_samples, feature_idx, threshold, &left_indices, &left_size, &right_indices, &right_size);

            if (left_size > 0 && right_size > 0) {
                double gini = gini_index(X, y, n_samples, left_indices, left_size, right_indices, right_size);

                if (gini < *best_gini) {
                    *best_gini = gini;
                    *best_feature = feature_idx;
                    *best_threshold = threshold;
                }
            }

            free(left_indices);
            free(right_indices);
        }
    }
}

static TreeNode* create_leaf_node(double *y, size_t *indices, size_t n_samples) {
    TreeNode *leaf = (TreeNode *)malloc(sizeof(TreeNode));
    leaf->left = NULL;
    leaf->right = NULL;

    size_t class_counts[2] = {0, 0};
    for (size_t i = 0; i < n_samples; i++) {
        class_counts[(int)y[indices[i]]]++;
    }

    leaf->value = (class_counts[0] > class_counts[1]) ? 0 : 1;
    return leaf;
}

static TreeNode* build_tree(double **X, double *y, size_t n_samples, size_t n_features, size_t depth, size_t max_depth, size_t min_samples_split) {
    if (n_samples <= min_samples_split || depth >= max_depth) {
        size_t *indices = (size_t *)malloc(n_samples * sizeof(size_t));
        for (size_t i = 0; i < n_samples; i++) {
            indices[i] = i;
        }
        TreeNode *leaf = create_leaf_node(y, indices, n_samples);
        free(indices);
        return leaf;
    }

    size_t best_feature;
    double best_threshold;
    double best_gini;

    find_best_split(X, y, n_samples, n_features, &best_feature, &best_threshold, &best_gini);

    if (best_gini == DBL_MAX) {
        size_t *indices = (size_t *)malloc(n_samples * sizeof(size_t));
        for (size_t i = 0; i < n_samples; i++) {
            indices[i] = i;
        }
        TreeNode *leaf = create_leaf_node(y, indices, n_samples);
        free(indices);
        return leaf;
    }

    size_t *left_indices, *right_indices;
    size_t left_size, right_size;
    split_dataset(X, y, n_samples, best_feature, best_threshold, &left_indices, &left_size, &right_indices, &right_size);

    TreeNode *node = (TreeNode *)malloc(sizeof(TreeNode));
    node->feature_idx = best_feature;
    node->threshold = best_threshold;
    node->left = build_tree(X, y, left_size, n_features, depth + 1, max_depth, min_samples_split);
    node->right = build_tree(X, y, right_size, n_features, depth + 1, max_depth, min_samples_split);

    free(left_indices);
    free(right_indices);

    return node;
}

double decision_tree_predict(TreeNode *node, double *x) {
    if (node->left == NULL && node->right == NULL) {
        return node->value;
    }

    if (x[node->feature_idx] < node->threshold) {
        return decision_tree_predict(node->left, x);
    } else {
        return decision_tree_predict(node->right, x);
    }
}

static void free_tree(TreeNode *node);

void decision_tree_free(DecisionTree *tree) {
    free_tree(tree->root);
    free(tree);
}

static void free_tree(TreeNode *node) {
    if (node == NULL) return;

    free_tree(node->left);
    free_tree(node->right);
    free(node);
}

void decision_tree_train(Model *self, double **X, double *y, int n_samples, int n_features, size_t max_depth, size_t min_samples_split) {
    self->tree = (DecisionTree *)malloc(sizeof(DecisionTree));
    self->tree->root = build_tree(X, y, n_samples, n_features, 0, max_depth, min_samples_split);
}

int main() {
    int n_samples = 6;
    int n_features = 2;

    double **X = (double **)malloc(n_samples * sizeof(double *));
    for (int i = 0; i < n_samples; i++) {
        X[i] = (double *)malloc(n_features * sizeof(double));
    }

    X[0][0] = 2.0; X[0][1] = 3.0;
    X[1][0] = 1.0; X[1][1] = 1.0;
    X[2][0] = 4.0; X[2][1] = 5.0;
    X[3][0] = 3.0; X[3][1] = 2.0;
    X[4][0] = 5.0; X[4][1] = 4.0;
    X[5][0] = 6.0; X[5][1] = 5.0;

    double *residuals = (double *)malloc(n_samples * sizeof(double));
    residuals[0] = 0;
    residuals[1] = 0;
    residuals[2] = 1;
    residuals[3] = 0;
    residuals[4] = 1;
    residuals[5] = 1;

    Model *model = create_decision_tree();

    size_t max_depth = 10;
    size_t min_samples_split = 2;

    decision_tree_train(model, X, residuals, n_samples, n_features, max_depth, min_samples_split);

    decision_tree_free(model->tree);
    free(model);
    for (int i = 0; i < n_samples; i++) {
        free(X[i]);
    }
    free(X);
    free(residuals);

    return 0;
}
