#ifndef DOCKWIDGET_H
#define DOCKWIDGET_H

#include <QLabel>
#include <QMenu>
#include <QString>
#include <QAction>
#include <QTreeView>
#include <QFileInfo>
#include <QDockWidget>
#include <QStandardItem>

class DockWidget : public QDockWidget
{
    Q_OBJECT

public:
     explicit DockWidget(QWidget *parent=0);
    DockWidget();
    ~DockWidget();

    void addtreeview(); //该函数为在DockWidget 添加treeview
    QString getImageName(QString s); //截取图片名函数
    void appendrow(QStandardItem* folder); //添加图片名到treeview
    QTreeView *treeview;
    QStandardItemModel* model;
    QStandardItem* root;
    int count=0;

private:
     QMenu* m_projectMenu;
     QMenu* m_itemMenu;

private slots:
     //void on_treeView_customContextMenuRequested(const QPoint &pos);

};

#endif // DOCKWIDGET_H
