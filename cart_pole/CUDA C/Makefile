# Makefile

# Try to find nvcc in PATH, otherwise use the hardcoded path
NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
    NVCC = /cs/local/lib/pkg/cudatoolkit/bin/nvcc
endif

# Add the architecture flag
CUFLAGS   := -O3 -g -arch=sm_86

# Define directories
SRC_DIR   := src
TEST_DIR  := tests
BUILD_DIR := build

# Source files: all .cu files in src, excluding any test files (if any)
SRCS := $(wildcard $(SRC_DIR)/*.cu)
SRCS := $(filter-out $(SRC_DIR)/tests.cu, $(SRCS))
# Test source files now live in the tests directory
TEST_SRCS := $(wildcard $(TEST_DIR)/*.cu)

# Object and dependency files for src and tests
OBJS      := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRCS))
TEST_OBJS := $(patsubst $(TEST_DIR)/%.cu,$(BUILD_DIR)/%.o,$(TEST_SRCS))

DEPS      := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.d,$(SRCS)) \
             $(patsubst $(TEST_DIR)/%.cu,$(BUILD_DIR)/%.d,$(TEST_SRCS))

# Output program names (main executable from src files, test executable from tests)
PROGS     := main
TEST_PROGS := tests

.PHONY: all clean test
all: $(BUILD_DIR) $(PROGS)

# Create the build directory if it does not exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Link the main program (using only objects built from src)
main: $(OBJS)
	$(NVCC) $(CUFLAGS) $(OBJS) -o run -lcurand
	@rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d

# Build the test executable (link tests objects with non-main objects from src)
test: $(TEST_OBJS) $(filter-out $(BUILD_DIR)/main.o, $(OBJS))
	$(NVCC) $(CUFLAGS) $(TEST_OBJS) $(filter-out $(BUILD_DIR)/main.o, $(OBJS)) -o run_tests -lcurand
	@rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d

# Clean up build directory
clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d run run_tests

# Compilation rule for src .cu files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) -c $(CUFLAGS) -o $@ $<

# Compilation rule for tests .cu files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) -c $(CUFLAGS) -o $@ $<

# Dependency generation rule for src files
$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) -M $(CUFLAGS) $< > $@

# Dependency generation rule for tests files
$(BUILD_DIR)/%.d: $(TEST_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) -M $(CUFLAGS) $< > $@

# Include the generated dependency files
-include $(DEPS)
