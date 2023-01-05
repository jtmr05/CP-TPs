# ----------//----------
# This is a C makefile.|
# ----------//----------



# whether or not to use precompiled headers (comment to disable)
#USE_PCH		:= 1



#directories
SRC_DIR 		:= src
INC_DIR 		:= include
OBJ_DIR 		:= obj
BIN_DIR 		:= bin



#files
SRC_FILES	 	:= $(shell find $(SRC_DIR) -name "*.cu")
OBJ_FILES		:= $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_FILES))
INC_FILES		:= $(shell find $(INC_DIR) -name "*.hpp")
PCH_INC_FILES 	:=

BIN		 		:= $(BIN_DIR)/k_means



#compiler
CC 				:= nvcc

#language standard to be used
STD				:= c++17

#compiler flags
CFLAGS 			:= -I$(INC_DIR) -O3 -std=$(STD) -DNDEBUG -Wno-deprecated-gpu-targets

#linker flags (e.g. -L/path/to/lib)
LDFLAGS			:=

#linker libraries (e.g. -lm)
LDLIBS			:=



ifdef USE_PCH

	ifeq (g++, $(CXX))
		PCH_EXT 	:= gch
	else
		PCH_EXT 	:= pch
	endif

	PCH_INC_FILES 	+= $(patsubst %,%.$(PCH_EXT),$(INC_FILES))

endif



#make default goal (using make with no specified recipe)
.DEFAULT_GOAL := all

all: $(INFO_OPT_FILE) $(BIN)

build: clean all

$(BIN): $(PCH_INC_FILES) $(OBJ_FILES)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(filter %.o,$^) -o $@ $(LDFLAGS) $(LDLIBS)

#generate each object file according to the corresponding source file
#create directories as needed
#https://www.gnu.org/software/make/manual/make.html#Static-Pattern
$(OBJ_FILES): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(PCH_INC_FILES): $(INC_DIR)/%.$(PCH_EXT) : $(INC_DIR)/%
	$(CC) $(CFLAGS) -x c-header $< -o $@

.PHONY: $(INFO_OPT_FILE)
$(INFO_OPT_FILE):
	@rm -f $@

#'clean' doesn't represent an actual file generating recipe
.PHONY: clean
clean:
	-rm -rf $(OBJ_DIR)
	-rm -rf $(BIN_DIR)
	-rm -f perf.data times.txt $(INFO_OPT_FILE)
ifdef PCH_INC_FILES
	-rm -rf $(PCH_INC_FILES)
endif



# number of threads to be used
#THREADS 		:= 4
#
#runseq: $(BIN)
#	./$^ 10000000 ${CP_CLUSTERS}
#
#runpar: $(BIN)
#	./$^ 10000000 ${CP_CLUSTERS} $(THREADS)
