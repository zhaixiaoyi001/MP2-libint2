# Makefile for Hartree-Fock program
# Uses Libint2 and Eigen libraries

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp -Wall
CXXFLAGS_DBG = -std=c++11 -g -march=native -Wall
CPPFLAGS = -I. -I/home/zhaixy/libint/libint2/include -I/usr/include/eigen3

# Libraries
LDFLAGS = -fopenmp
LDLIBS = -L/home/zhaixy/libint/libint2/lib -lint2 -lpthread

# Source files
SRCS = main.cpp utils.cpp integrals.cpp linalg.cpp globals.cpp
OBJS = $(SRCS:.cpp=.o)
DEPS = hartree_fock.h

# Target executable
TARGET = hartree-fock++

# Default target
all: $(TARGET)

# Release build
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Debug build
debug: CXXFLAGS = $(CXXFLAGS_DBG)
debug: LDFLAGS = 
debug: $(TARGET)

# Compile source files
%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean debug

# Dependencies
main.o: hartree_fock.h
utils.o: hartree_fock.h
integrals.o: hartree_fock.h
linalg.o: hartree_fock.h
globals.o: hartree_fock.h
