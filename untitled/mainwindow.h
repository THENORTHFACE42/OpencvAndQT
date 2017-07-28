#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMenu>
#include <QLabel>
#include <vector>
#include <QString>
#include <QPixmap>
#include <QCursor>
#include <QStringList>
#include <QFileDialog>
#include <QMainWindow>
#include <QMessageBox>
#include <QTextStream>
#include <QProgressBar>
#include <QSystemTrayIcon>

#include "knn.h"
#include "file.h"
#include "dialog.h"
#include "global.h"
#include "imagebox.h"


//=======opencv头文件包含==========
#include <opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

//===============================
#include <iostream>

using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    QMenu* m_projectMenu;//针对项目右键弹出的菜单

    QMenu* m_itemMenu; //针对图片右键弹出的菜单

    QAction* delProject;//当选中第一个项目时不能删除项目，设置该按钮不显示

    QAction* clear1;//当项目为空的时候不显示该按钮

    QStandardItem *knn_item;

    Mat K_means(const Mat &src,int k,int t,int Count[]);

    void ISOData(int state, Mat image,int newexpClusters ,int newthetaN , int newmaxIts, int newcombL , double newthetaS , double newthetaC, QString str, QString filePath );

    Mat KNN_two(string root,Mat trainingDataMat,Mat labelsMat);

    Mat KNN(string root,Mat trainingDataMat,Mat labelsMat,vector<int> &info);

    Mat nearest_return(string root,Mat trainingData,Mat labelMat);

    Mat Svmclass(string root,Mat trainingDataMat,Mat labelsMat,vector<int> &info);

    Mat Svmclass_return(string root,Mat trainingDataMat,Mat labelsMat);

    Mat nearest(string root,Mat trainingData,Mat labelMat,vector<int> &info);


private slots:
    void on_actionOpen_triggered();

    void on_actionSave_triggered();

    void on_actionSave_As_triggered();

    void on_actionCursor_triggered();

    void on_actionHands_triggered();

    void on_actionChoose_position_triggered();

    void on_actionZoom_In_triggered();

    void on_actionZoom_Out_triggered();

    void on_actionNormal_Size_triggered();

    void on_actionWise_clock_triggered();

    void on_actionWise_anticlock_triggered();

    void showNormal();

    void OnlineTreeDoubleClick(const QModelIndex &index); //双击获取图片的名称

    void showmenu(const QPoint &pos);//显示右键菜单函数

    void remove();//右键删除图片函数

    void fold();//右键折叠全部函数

    void clear();//右键清空项目函数

    void openshow();//右键打开按钮并显示图片函数

    void NewProject();//右键添加新项目函数

    void DelProject();//右键删除项目函数

    void Add();//右键项目添加图片函数

    void Image_Add();//右键图片添加图片

    void on_actionK_means_triggered();

    void on_actionKNN_triggered();

    void on_actionISODATA_triggered();

    void receiveData(QString data1,QString data2,QString data3,QString data4, QString data5, QString data6);

    void receivearray(int a);//接收knn传入的像素点数组

    void choose_position_changed();//接收knn传入的值

    void choose_position_nonchanged();

    void on_actionMindisf_triggered();

    void on_actionSVM_triggered();



private:
    Ui::MainWindow *ui;

    QPixmap currentPixmap;  //获取左边TreeView选取的图片信息
    QStringList filenameList;
    File file;

    //任务栏托盘菜单
    QSystemTrayIcon *myTrayIcon;
    QMenu *myMenu;
    QAction *restoreWinAction;
    QAction *quitAction;
    void createMenu();

    //底部菜单
    QLabel *sizeLabel;
    QLabel *infoLabel;
    Dialog *dlg;
    QProgressBar *progress;

    //s1---ss4为ISODATA参数传回值
    QString s1;
    QString s2;
    QString s3;
    QString s4;
    QString s5;
    QString s6;
    knn *k;
    QString src_image;
    int model_item_count;

signals:
    void refresh();

protected:
    //void mouseMoveEvent(QMouse *event);
    //void wheelEvent(QWheelEvent *event);
};

#endif // MAINWINDOW_H
