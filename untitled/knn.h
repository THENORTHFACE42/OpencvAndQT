#ifndef KNN_H
#define KNN_H

#include <QMenu>
#include <QDialog>
#include <QTreeView>
#include <QStandardItemModel>

#include "global.h"
#include "imagebox.h"


namespace Ui {
class knn;
}

class knn : public QDialog
{
    Q_OBJECT

public:
    explicit knn(QWidget *parent = 0);
    ~knn();
    QMenu* m_projectMenu;//针对类别右键弹出的菜单
    QMenu* m_itemMenu; //针对像素点右键弹出的菜单
    QAction* delProject;//当选中第一个项目时不能删除项目，设置该按钮不显示
    QAction* clear1;//当项目为空的时候不显示该按钮
    QTreeView *treeview;
    QStandardItemModel *model;
    ImageBox *ib;

    int count;

private slots:
     void showmenu(const QPoint &pos);//右键菜单显示

     void NewProject();//右键添加类别函数

     void DelProject();//右键删除项目函数

     void Add();  //右键添加像素点

     void reciveData(QPoint);//接收imagebox所选取的像素点

     void remove();//右键移除像素点

     void fold();//折叠节点

     void on_buttonBox_accepted();

     void clear();//右键清空像素点函数

     //void on_buttonBox_destroyed();

     void on_buttonBox_rejected();

     void getrequire();

private:
    Ui::knn *ui;

signals:
    void sendData(int x);
    void refresh();
    void sendarray(int a);
    void choose_position_changed();
    void choose_position_nonchanged();
};

#endif // KNN_H
