CXX = nvcc

CXXFLAGS	:= --compiler-options=-Wextra,-Wall,-O2,-Wno-unused-result,-Wno-unused-parameter
CXXFLAGS_DEBUG	:= -DDEBUG -g -G --compiler-options=-Wextra,-Wall,-Wno-unused-result,-Wno-unused-parameter 


LFLAGS = #-lcuda -lcudart
OUTPUT	:= output

SRC		:= src
INCLUDE	:= include
LIB		:= lib

MAIN	:= main
SOURCEDIRS	:= $(shell find $(SRC) -type d)
INCLUDEDIRS	:= $(shell find $(INCLUDE) -type d)
LIBDIRS		:= $(shell find $(LIB) -type d)
RM = rm -f
MD	:= mkdir -p
OBJECTS := main.cu
OUTPUTMAIN := $(OUTPUT)/run


all: $(OUTPUT) $(MAIN)

debug:
	$(CXX) src/$(OBJECTS) $(CXXFLAGS_DEBUG) -I $(INCLUDEDIRS) -o $(OUTPUTMAIN)  $(LFLAGS) $(LIBS)


$(OUTPUT):
	$(MD) $(OUTPUT)

$(MAIN):
	$(CXX) src/$(OBJECTS) $(CXXFLAGS) -I $(INCLUDEDIRS) -o $(OUTPUTMAIN)  $(LFLAGS) $(LIBS)

clean:
	$(RM) $(OUTPUTMAIN)
	$(RM) $(call FIXPATH,$(OBJECTS))
