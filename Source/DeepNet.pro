TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

HEADERS += \
    timer.h \
    matrix.h \
    linalg.h \
    data.h \
    rand_utils.h \
    nnetwork.h \
    operations.h \
    evaluation.h

OTHER_FILES += ./kernel.cu
CUDA_SOURCES += ./kernel.cu
CUDA_SDK = "/usr/local/cuda-10.1/"
CUDA_DIR = "/usr/local/cuda-10.1/"

SYSTEM_NAME = x64
SYSTEM_TYPE = 64
CUDA_ARCH = sm_75
NVCC_OPTIONS = --use_fast_math
INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib/
CUDA_OBJECTS_DIR = ./
CUDA_LIBS = -lcuda -lcudart
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
LIBS += -L$$CUDA_DIR/lib64 -lcuda -lcudart
LIBS += -pthread -fopenmp

CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

# MPI Settings
QMAKE_CXX = mpicxx
QMAKE_CXX_RELEASE = $$QMAKE_CXX
QMAKE_CXX_DEBUG = $$QMAKE_CXX
QMAKE_LINK = $$QMAKE_CXX
QMAKE_CC = mpicc

QMAKE_CFLAGS += $$system(mpicc --showme:compile)
QMAKE_LFLAGS += $$system(mpicxx --showme:link)
QMAKE_CXXFLAGS_RELEASE += $$system(mpicxx --showme:compile) -DMPICH_IGNORE_CXX_SEEK
QMAKE_CXXFLAGS += -std=c++17 -pthread -fopenmp -O3 $$system(mpicxx --showme:compile) -DMPICH_IGNORE_CXX_SEEK

DISTFILES += \
    matrix.txt \
    kernel.cu
