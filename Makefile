CC = gcc
CFLAGS = -Wall -lm

SRC = main.c pca.c grid_search.c dataset.c decision_tree.c gradient_boosting.c knn.c logical_regression.c random_forest.c svm.c preprocess.c utils.c
OBJ = $(SRC:.c=.o)
TARGET = program

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean