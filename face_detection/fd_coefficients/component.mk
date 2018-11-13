#Component makefile

COMPONENT_ADD_INCLUDEDIRS := .
COMPONENT_SRCDIRS := .

#Call: $(eval $(call CompileNeuralNetCoefficients,directory,nn_name,flags))
define CompileNeuralNetCoefficients
COMPONENT_OBJS += $2.o
COMPONENT_EXTRA_CLEAN += $$(COMPILING_COMPONENT_PATH)/$2.c $$(COMPILING_COMPONENT_PATH)/$2.h

ifeq ($(findstring Linux, $(shell uname)), Linux)
	MKMODEL_EXE := mkmodel_linux
else ifeq ($(findstring MINGW, $(shell uname)), MINGW)
	MKMODEL_EXE := mkmodel_windows
else ifeq ($(findstring Darwin, $(shell uname)), Darwin)
	MKMODEL_EXE := mkmodel_macos
endif

$$(COMPONENT_PATH)/./$2.c: $$(COMPONENT_PATH)/$1/ $$(MKMODEL_PATH)/$$(MKMODEL_EXE) ../include/sdkconfig.h
	echo "Running mkmodel for $2, flags \"$3 $4\""
	$$(MKMODEL_PATH)/$$(MKMODEL_EXE) $$(COMPONENT_PATH)/$1 $$(COMPONENT_PATH)/$2.c $$(COMPONENT_PATH)/$2.h $2 $3 $4

endef

MKMODEL_PATH := $(COMPONENT_PATH)/../../lib

$(eval $(call CompileNeuralNetCoefficients,pnet/model,pnet_model,-no-quantized,-3d))
$(eval $(call CompileNeuralNetCoefficients,rnet/model,rnet_model,-no-quantized,-3d))
$(eval $(call CompileNeuralNetCoefficients,onet/model,onet_model,-no-quantized,-3d))
