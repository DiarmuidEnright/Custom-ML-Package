CC = gcc
CFLAGS = -Wall -I../include

# Directories
BUILD_DIR = ../build
BIN_DIR = ../bin

# Source files in current directory
SRC_SRCS = main.c grid_search.c logical_regression.c neural_network.c pca.c

# Source files in subdirectories
CORE_SRCS = ../core/model.c ../core/dataset.c
ML_SRCS = ../ml/decision_tree.c ../ml/gradient_boosting.c ../ml/knn.c ../ml/random_forest.c ../ml/svm.c
UTILS_SRCS = ../utils/cross_validation.c ../utils/matrix.c ../utils/preprocess.c ../utils/utils.c
ENSEMBLE_SRCS = ../ensemble/bagging.c ../ensemble/ensemble_methods.c

SRCS = $(SRC_SRCS) $(CORE_SRCS) $(ML_SRCS) $(UTILS_SRCS) $(ENSEMBLE_SRCS)

# Generate object file names without stripping directory names
OBJS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRC_SRCS)) \
       $(patsubst ../core/%.c,$(BUILD_DIR)/core/%.o,$(CORE_SRCS)) \
       $(patsubst ../ml/%.c,$(BUILD_DIR)/ml/%.o,$(ML_SRCS)) \
       $(patsubst ../utils/%.c,$(BUILD_DIR)/utils/%.o,$(UTILS_SRCS)) \
       $(patsubst ../ensemble/%.c,$(BUILD_DIR)/ensemble/%.o,$(ENSEMBLE_SRCS))

# Create directories
$(shell mkdir -p $(BUILD_DIR) $(BIN_DIR) $(BUILD_DIR)/core $(BUILD_DIR)/ml $(BUILD_DIR)/utils $(BUILD_DIR)/ensemble)

TARGET = $(BIN_DIR)/program

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking..."
	@$(CC) -o $@ $^ -lm

$(BUILD_DIR)/%.o: %.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/core/%.o: ../core/%.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/ml/%.o: ../ml/%.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/utils/%.o: ../utils/%.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/ensemble/%.o: ../ensemble/%.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean
