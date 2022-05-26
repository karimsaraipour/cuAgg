
CPP = clang++
CPP_FLAGS = -std=c++11
EXEC = main

.PHONY: all
all: main

main: main.cpp 
	$(CPP) $(CPP_FLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f $(EXEC)
