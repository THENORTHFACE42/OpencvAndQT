#include "dockwidget.h"

#include<QLabel>
#include<QTreeview>
#include<QModelIndex>
#include<QStandardItem>

//======================================
#define ROLE_MARK Qt::UserRole + 1
#define ROLE_MARK_FOLDER Qt::UserRole + 2

//====ROLE_MARK用于区分根节点和文件夹节点====
#define MARK_PROJECT 1  //这是总项目标记
#define MARK_FOLDER 2   //这是文件夹标记
#define MARK_FOLDER_H 1

DockWidget::DockWidget()
{

}

DockWidget::~DockWidget()
{

}

DockWidget::DockWidget(QWidget *parent):QDockWidget(parent){

}

void DockWidget::addtreeview()      //Dockwidget添加treeview函数
{
    treeview=new QTreeView(this);   //新建一个treeview
    this->setWidget(treeview);      //让treeview填充DockWidget，让treeview属于dockwidget的一部分
    model = new QStandardItemModel();
    model->setHorizontalHeaderLabels(QStringList()<<QStringLiteral("Classfication beta 0.0.0"));    //添加一个Headerlabel
    root = new QStandardItem(QIcon(":/pic/icon/file"),QStringLiteral("默认项目"));                   //image的目录
    root->setData(MARK_PROJECT,ROLE_MARK);                                                          //首先它是项目中目录
    root->setData(MARK_FOLDER,ROLE_MARK_FOLDER);                                                    //其次它属于文件夹
    model->appendRow(root);
    treeview->setModel(model);
}

QString DockWidget::getImageName(QString s) //截取字符串函数，用于截取图片路径
{
    QFileInfo fileInfo;
    fileInfo = QFileInfo(s);

    return fileInfo.fileName();
}

void DockWidget::appendrow(QStandardItem* folder)   //在root下面添加一个item
{
    folder->setData(MARK_FOLDER,ROLE_MARK);         //首先它是文件夹
    folder->setData(MARK_FOLDER,ROLE_MARK_FOLDER);  //其次它属于文件夹的子项目
    root->appendRow(folder);
    treeview->setModel(model);
    treeview->expandAll();
}

