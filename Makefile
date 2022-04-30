CXX = nvcc

CXXFLAGS	:= -DNEDUBG --compiler-options=-Wextra,-Wall,-O3,-Wno-unused-result,-Wno-unused-parameter


LFLAGS = -lcuda -lcudart
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


$(OUTPUT):
	$(MD) $(OUTPUT)

$(MAIN):
	$(CXX) src/$(OBJECTS) $(CXXFLAGS) -I $(INCLUDEDIRS) -o $(OUTPUTMAIN)  $(LFLAGS) $(LIBS)

clean:
	$(RM) $(OUTPUTMAIN)
	$(RM) $(call FIXPATH,$(OBJECTS))
