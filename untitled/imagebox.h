#ifndef IMAGEBOX_H
#define IMAGEBOX_H

#include <QMenu>
#include <QVector>
#include <QAction>
#include <QWidget>
#include <QPixmap>
#include <QString>
#include <QCursor>
#include <QPainter>
#include <QToolTip>
#include <iostream>
#include <QMessageBox>
#include <QPaintEvent>
#include <QWheelEvent>
#include <QFileDialog>
#include <QResizeEvent>
#include <QContextMenuEvent>

#include "data.h"
#include "global.h"
#include "stoast.h"


class ImageBox : public QWidget
{
    Q_OBJECT

public:
    explicit ImageBox(QWidget *parent = 0);
    ~ImageBox();

    //在控件中显示图像
    void setPixmap(const QPixmap& pixmap);

    // 对图像的简单操作
    void showOriginalSize();
    void showSuitableSize();
    void zoomIn();
    void zoomOut();
    void clockWise();
    void anticlockWise();

    //对右键菜单操作
    void actionRefresh_tragged();
    void action_16_label();
    void action_32_label();
    void action_non_label();
    void contextMenuEvent(QContextMenuEvent *event);

signals:
    void sendData(QPoint);
    void require();

private slots:
    void receive(int a);
    void rrefresh();

public:
    QPixmap m_pixmap;

    //缩放
    double m_scale;
    double m_percentage;
    int m_suitableWidth;
    int m_suitableHeight;

    void ariseScale(int rate);

    //移动
    QPoint m_pressPoint;
    double m_originX;
    double m_originY;
    double m_basicX;
    double m_basicY;

    //右键菜单
    QMenu *menu;
    QAction *refresh;
    QAction *label_16;
    QAction *label_32;
    QAction *label_non;

    //固定的当前图片的左上角图标
    double CC_x, CC_y;
    double CC_width, CC_height;

    int cc;

protected:
    void paintEvent(QPaintEvent *event);
    void wheelEvent(QWheelEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);

private:
    bool isInPixmap(QPoint pos);

};

#endif // IMAGEBOX_H
