CXX = nvcc

CXXFLAGS	:= --shared -c --compiler-options=-Wextra,-Wall,-O3,-Wno-unused-result,-Wno-unused-parameter,-fPIC
CXXFLAGS_DEBUG	:= --shared -c -DDEBUG -g -G --compiler-options=-Wextra,-Wall,-Wno-unused-result,-Wno-unused-parameter,-fPIC 


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
OBJECTS := $(shell find $(SRC) -type f -name "*.cu" | sed "s/.*\///" | cut -f 1 -d '.')


all: $(OUTPUT) $(MAIN)

debug:
	$(CXX) src/$(OBJECTS).cu $(CXXFLAGS_DEBUG) -I $(INCLUDEDIRS) -o lib/lib$(OBJECTS).so  $(LFLAGS) $(LIBS)


$(OUTPUT):
	$(MD) $(OUTPUT)

$(MAIN):
	$(CXX) src/$(OBJECTS).cu $(CXXFLAGS) -I $(INCLUDEDIRS) -o lib/lib$(OBJECTS).so $(LFLAGS) $(LIBS)

clean:
	$(RM) $(OUTPUTMAIN)
	$(RM) $(call FIXPATH,$(OBJECTS))
