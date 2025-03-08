CC = gcc
CFLAGS = -Wall -Iinclude

SRC = main.c pca.c grid_search.c dataset.c decision_tree.c \
      gradient_boosting.c knn.c logical_regression.c \
      random_forest.c svm.c preprocess.c utils.c \
      bagging.c cross_validation.c model.c ensemble_methods.c

OBJ = $(SRC:.c=.o)
TARGET = program

all: $(TARGET)

$(TARGET): $(OBJ)
	@echo "Linking..."
	$(CC) -o $(TARGET) $(OBJ) -lm

%.o: %.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@echo "Cleaning up..."
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
