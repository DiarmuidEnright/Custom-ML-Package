CC = gcc
CFLAGS = -Wall -Werror -g

SRC = src/svm.c src/knn.c src/decision_tree.c src/dataset.c src/utils.c src/main.c
OBJ = $(SRC:.c=.o)

ml_lib: $(OBJ)
	$(CC) $(CFLAGS) -o ml_lib $(OBJ)

clean:
	rm -f ml_lib $(OBJ)
