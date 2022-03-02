QT -= gui

CONFIG += c++17 console
CONFIG -= app_bundle

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        main.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

unix:!macx: LIBS += -L$$PWD/../build-Smart_Fashion-Desktop_Qt_5_15_1_GCC_64bit-Release/App/plugins/LP_Plugin_Garment_Manipulation/externs \
   -lc10 \
   -lc10_cuda \
   -ltorch \
   -ltorch_cpu \
   -ltorch_cuda \
   -ltorch_cuda_cu \
   -ltorch_cuda_cpp \
   -lprotobuf \
   -lprotobuf-lite \
   -lprotoc

INCLUDEPATH += $$PWD/../../Gym_CPP-main
DEPENDPATH += $$PWD/../../Gym_CPP-main

INCLUDEPATH += /usr/include/eigen3
DEPENDPATH += /usr/include/eigen3

INCLUDEPATH += $$PWD/../../OpenAI_Gym/CPPGym
DEPENDPATH += $$PWD/../../OpenAI_Gym/CPPGym

INCLUDEPATH += $$PWD/../../libtorch/include
DEPENDPATH += $$PWD/../../libtorch/include

INCLUDEPATH += $$PWD/../../libtorch/include/torch/csrc/api/include
DEPENDPATH += $$PWD/../../libtorch/include/torch/csrc/api/include

INCLUDEPATH += $$PWD/../../tensorboard_logger-master/include
DEPENDPATH += $$PWD/../../tensorboard_logger-master/include

INCLUDEPATH += $$PWD/../../tensorboard_logger-master/build
DEPENDPATH += $$PWD/../../tensorboard_logger-master/build

unix:!macx: LIBS += -L$$PWD/../../tensorboard_logger-master/build \
   -ltensorboard_logger
