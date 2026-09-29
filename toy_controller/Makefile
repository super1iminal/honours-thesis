# Try to find nvcc in PATH, otherwise use the hardcoded path
NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
    NVCC = /cs/local/lib/pkg/cudatoolkit/bin/nvcc
endif

# Add the architecture flag
CUFLAGS   := -O3 -g -arch=sm_86

# Define directories
SRC_DIR   := src
BUILD_DIR := build

# Source files are now in the src directory
SRCS      := $(SRC_DIR)/main.cu $(SRC_DIR)/utility.cu $(SRC_DIR)/simulation.cu $(SRC_DIR)/bisection.cu $(SRC_DIR)/logger.cu $(SRC_DIR)/gradient.cu $(SRC_DIR)/regions.cu
OBJS      := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRCS))
DEPS      := $(SRCS:.cu=.d)
OBJ_FILES := $(addprefix $(BUILD_DIR)/, $(notdir $(OBJS)))
DEP_FILES := $(addprefix $(BUILD_DIR)/, $(notdir $(DEPS)))

# Output program name
PROGS     := run

.PHONY: all clean
all: $(BUILD_DIR) $(PROGS)

# Create the build directory if it does not exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Link the program
$(PROGS): $(OBJ_FILES)
	$(NVCC) $(CUFLAGS) $(OBJ_FILES) -o $@ -lcurand

# Clean up build directory
clean:
	rm -rf $(BUILD_DIR) $(PROGS)

# Compilation rule for .o files, placing them in the build directory
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) -c $(CUFLAGS) -o $@ $<

# Dependency generation rule, placing the .d files in the build directory
$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cu
	$(NVCC) -M $(CUFLAGS) $< > $@

# Include the generated dependency files
-include $(DEP_FILES)
