LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := adofai_editor_zygisk
LOCAL_SRC_FILES := module.cpp
LOCAL_C_INCLUDES := $(ZYGISK_API_DIR)
LOCAL_CPPFLAGS := -std=c++17 -fno-exceptions -fno-rtti -Wall -Wextra
LOCAL_LDLIBS := -llog -ldl -lstdc++
include $(BUILD_SHARED_LIBRARY)
