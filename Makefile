PROJECT_NAME := esp_face

MODULE_PATH := $(abspath $(shell pwd))

EXTRA_COMPONENT_DIRS += $(MODULE_PATH)/lib
EXTRA_COMPONENT_DIRS += $(MODULE_PATH)/image_util
EXTRA_COMPONENT_DIRS += $(MODULE_PATH)/face_detection/fd_coefficients
EXTRA_COMPONENT_DIRS += $(MODULE_PATH)/face_detection/mtmn 
EXTRA_COMPONENT_DIRS += $(MODULE_PATH)/face_recognition/fr_coefficients
EXTRA_COMPONENT_DIRS += $(MODULE_PATH)/face_recognition/frmn

include $(IDF_PATH)/make/project.mk

