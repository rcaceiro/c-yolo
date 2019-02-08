CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
INCLUDE(CheckLanguage)
PROJECT(yolo)

SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_C_STANDARD 11)
SET(CMAKE_CUDA_STANDARD 11)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -DOPENCV")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -DOPENCV")

IF(NOT DEFINED NOT_COMPILE_CUDA)
 CHECK_LANGUAGE(CUDA)
 IF(CMAKE_CUDA_COMPILER)
  ENABLE_LANGUAGE(CUDA)

  SET(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
  SET(CUDA_VERBOSE_BUILD ON)
  SET(CUDA_64_BIT_DEVICE_CODE ON)

  SET(CUDA_LIBS cudnn curand cublas)

  SET(GENCODE "-arch=sm_50 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75")
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${GENCODE} --compiler-options \"-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC -DOPENCV -DGPU -DCUDNN")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGPU -DCUDNN")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU -DCUDNN")

  INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
 ENDIF()
ENDIF()

IF(CMAKE_BUILD_TYPE MATCHES "Debug")
 SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
 SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
 SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g\"")
ELSE(CMAKE_BUILD_TYPE NOT MATCHES "Debug")
 SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
 SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast")
 SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Ofast\"")
ENDIF()

FIND_PACKAGE(OpenCV 3.0.0 REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

IF(NOT DEFINED NOT_COMPILE_OMP)
 FIND_PACKAGE(OpenMP)
 IF(OPENMP_C_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
 ENDIF()
 IF(OPENMP_CXX_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
 ENDIF()
ENDIF()

INCLUDE_DIRECTORIES(./include)
INCLUDE_DIRECTORIES(./src)
INCLUDE_DIRECTORIES(./darknet/include)

SET(DARKNET_C_SOURCES ./darknet/src/gemm.c ./darknet/src/utils.c ./darknet/src/cuda.c ./darknet/src/deconvolutional_layer.c ./darknet/src/convolutional_layer.c ./darknet/src/list.c ./darknet/src/image.c ./darknet/src/activations.c ./darknet/src/im2col.c ./darknet/src/col2im.c ./darknet/src/blas.c ./darknet/src/crop_layer.c ./darknet/src/dropout_layer.c ./darknet/src/maxpool_layer.c ./darknet/src/softmax_layer.c ./darknet/src/data.c ./darknet/src/matrix.c ./darknet/src/network.c ./darknet/src/connected_layer.c ./darknet/src/cost_layer.c ./darknet/src/parser.c ./darknet/src/option_list.c ./darknet/src/detection_layer.c ./darknet/src/route_layer.c ./darknet/src/upsample_layer.c ./darknet/src/box.c ./darknet/src/normalization_layer.c ./darknet/src/avgpool_layer.c ./darknet/src/layer.c ./darknet/src/local_layer.c ./darknet/src/shortcut_layer.c ./darknet/src/logistic_layer.c ./darknet/src/activation_layer.c ./darknet/src/rnn_layer.c ./darknet/src/gru_layer.c ./darknet/src/crnn_layer.c ./darknet/src/demo.c ./darknet/src/batchnorm_layer.c ./darknet/src/region_layer.c ./darknet/src/reorg_layer.c ./darknet/src/tree.c ./darknet/src/lstm_layer.c ./darknet/src/l2norm_layer.c ./darknet/src/yolo_layer.c ./darknet/src/iseg_layer.c)
FILE(GLOB_RECURSE DARKNET_CXX_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "./darknet/src/*.cpp")
IF(NOT DEFINED NOT_COMPILE_CUDA AND CMAKE_CUDA_COMPILER)
 FILE(GLOB_RECURSE DARKNET_CU_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "./darknet/src/*.cu")
 INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
ENDIF()

ADD_LIBRARY(yolo SHARED src/yolo.cpp src/yolo_error.c ${DARKNET_C_SOURCES} ${DARKNET_CXX_SOURCES} ${DARKNET_CU_SOURCES})
TARGET_LINK_DIRECTORIES(yolo PRIVATE ./src ./darknet/include)
TARGET_LINK_LIBRARIES(yolo PRIVATE opencv_core opencv_highgui opencv_videoio opencv_video ${CUDA_LIBS})
SET_TARGET_PROPERTIES(yolo PROPERTIES VERSION 1.0.0)

ADD_EXECUTABLE(yolo_test src/main.c)
TARGET_LINK_LIBRARIES(yolo_test yolo)