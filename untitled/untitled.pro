#-------------------------------------------------
#
# Project created by QtCreator 2016-10-15T09:25:35
#
#-------------------------------------------------
INCLUDEPATH+=D:\QTANDOPENCV\Opencv\install\include\opencv\
                    D:\QTANDOPENCV\Opencv\install\include\opencv2\
                   D:\QTANDOPENCV\Opencv\install\include
LIBS+=D:\QTANDOPENCV\Opencv\install\lib\libopencv_*.a


QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = untitled
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    file.cpp \
    imagebox.cpp \
    global.cpp \
    dockwidget.cpp \
    stoast.cpp \
    dialog.cpp \
    knn.cpp \
    data.cpp

HEADERS  += mainwindow.h \
    file.h \
    imagebox.h \
    global.h \
    dockwidget.h \
    kwindow.h \
    stoast.h \
    dialog.h \
    knn.h \
    data.h

FORMS    += mainwindow.ui \
    dialog.ui \
    knn.ui

RESOURCES += \
    resource.qrc
