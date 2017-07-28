#ifndef GLOBAL_H
#define GLOBAL_H

#include <QPoint>
#include <vector>

#include "data.h"

extern std::vector<Data> ttar;

class Global
{
public:
    Global();
    ~Global();

    static int drag_position;
    static bool real_time;
    static bool ispress;
    static int cf_property;  //分类的属性，当选中knn分类方法时为1，当mindis时为2，当svm时为3
    static int label;
};

#endif // GLOBAL_H
