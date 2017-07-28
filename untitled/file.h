#ifndef FILE_H
#define FILE_H

#include <vector>
#include <QString>
#include <QPixmap>
#include <QFileInfo>
#include <QStringList>
#include <QMessageBox>

#include "global.h"

class File
{
public:
    File();
    ~File();

    void conn(QStringList filenamelist);//push图片路径和图片名称到vector
    bool isEmpty();                     //判断vector中poxmapList是否为空
    void save(QString filePath, int i);//保存当前选中的图片
    bool save_as(QString filePath, int i);//另存为当前选中的图片
    int match(QString s);//查找图片名为s的图片并返回下标
    void del(int position);

    std::vector<QString> filename; //存储图片名称
    std::vector<QString> filenameList;//存储图片路径
    std::vector<QPixmap> pixmapList; //存储pixmap格式的可读图片
};

#endif // FILE_H
