#include "knn.h"
#include "ui_knn.h"

#include <vector>
#include <QTreeView>
#include <QMessageBox>
#include <QStandardItemModel>

#define ROLE_MARK Qt::UserRole + 1
#define ROLE_MARK_FOLDER Qt::UserRole + 2

//=============ROLE_MARK用于区分根节点和文件夹节点==============
#define MARK_PROJECT 1  //这是总项目标记
#define MARK_FOLDER 2   //这是文件夹标记
#define MARK_FOLDER_H 1


knn::knn(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::knn)
{
    count = 0;
    ui->setupUi(this);

    setWindowTitle("分类信息存储器");
    setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowIcon(QIcon(":/pic/main/main"));

    //初始化treeview
    treeview=new QTreeView(this);
    ui->widget->layout()->addWidget(treeview);

    model = new QStandardItemModel();
    model->setHorizontalHeaderLabels(QStringList()<<QStringLiteral("Classfication beta 0.0.0"));    //添加一个Headerlabel
    QStandardItem *root = new QStandardItem(QStringLiteral("类别1"));                                //iamge的目录
    root->setData(MARK_PROJECT,ROLE_MARK);                                                          //首先它是项目中目录
    root->setData(MARK_FOLDER,ROLE_MARK_FOLDER);                                                    //其次它属于文件夹
    model->appendRow(root);
    treeview->setModel(model);

    //设置treeview属性为自定义菜单
    treeview->setContextMenuPolicy(Qt::CustomContextMenu);

    //右键菜单槽函数连接
    this->connect(treeview,SIGNAL(customContextMenuRequested(QPoint)),
                  this,SLOT(showmenu(QPoint)));

//============================添加两个不同的菜单============================//
     m_projectMenu = new QMenu(this);
     m_itemMenu = new QMenu(this);

     //项目右键添加像素点事件
     QAction* newpic  = new QAction(QStringLiteral("添加样本像素点"),this);
     m_projectMenu->addAction(newpic);
     this->connect(newpic,SIGNAL(triggered(bool)),this,SLOT(Add()));

     //清空类别中像素点事件
     clear1 = new QAction(QStringLiteral("清空该类别数据"),this);
     m_projectMenu->addAction(clear1);
     this->connect(clear1,SIGNAL(triggered(bool)),this,SLOT(clear()));

     //类别右键添加类别事件
     QAction* NewProject = new QAction(QStringLiteral("添加类别"),this);
     m_projectMenu->addAction(NewProject);
     this->connect(NewProject,SIGNAL(triggered(bool)),this,SLOT(NewProject()));

     //类别右键删除类别事件
     delProject = new QAction(QStringLiteral("删除该类别"),this);
     m_projectMenu->addAction(delProject);
     this->connect(delProject,SIGNAL(triggered(bool)),this,SLOT(DelProject()));

//==================以上时针对类别右键菜单，完成。。。。。。========================
     //移除像素点按钮
     QAction *remove = new QAction(QStringLiteral("移除"),this);
     m_itemMenu->addAction(remove);
     this->connect(remove,SIGNAL(triggered(bool)),this,SLOT(remove()));

     //折叠treeview全部子项按钮
     QAction *fold = new QAction(QStringLiteral("折叠全部"),this);
     m_itemMenu->addAction(fold);
     this->connect(fold,SIGNAL(triggered(bool)),SLOT(fold()));
}

knn::~knn()
{
    int countofmodel=model->rowCount();
    //model->removeRows(1,countofmodel);
    for(int i=0;i<countofmodel-1;i++)
    {
        model->removeRow(1);
    }
    count=0;
    QStandardItem *itemone=new QStandardItem;
    QModelIndex index=model->index(0,0);
    itemone=model->itemFromIndex(index);
    int countofpoint=itemone->rowCount();
    itemone->removeRows(0,countofpoint);
    delete ui;
}

//显示右键菜单函数实现
void knn::showmenu(const QPoint &pos)
{
    QStandardItemModel* model = static_cast<QStandardItemModel*>(treeview->model());
    QModelIndex currentIndex = treeview->currentIndex();
    QModelIndex index = treeview->indexAt(pos);
    QStandardItem* currentItem = model->itemFromIndex(currentIndex);
    QVariant var = index.data(ROLE_MARK);

    if(var.isValid())
    {
        if(MARK_PROJECT == var.toInt())             //判断右键点击的目标属性
        {
            int count=currentItem->rowCount();

            if(count==0)
            {
                clear1->setEnabled(false);
            }
            else
            {
                clear1->setEnabled(true);
            }
            if(currentIndex.row()==0)
            {
                delProject->setEnabled(false);
            }
            else
            {
                delProject->setEnabled(true);
            }

            m_projectMenu->exec(QCursor::pos());    //弹出右键菜单，菜单位置为光标位置

        }
        else if(MARK_FOLDER == var.toInt())
        {
            m_itemMenu->exec(QCursor::pos());
        }
    }
}

//右键添加类别函数实现
void knn::NewProject()
{
    QStandardItem *newproject=new QStandardItem(QStringLiteral("类别%1").arg(count+2));
    newproject->setData(MARK_PROJECT,ROLE_MARK);            //标识
    newproject->setData(MARK_FOLDER,ROLE_MARK_FOLDER);      //标识
    model->appendRow(newproject);
    count++;
}

//右键删除类别函数实现
void knn::DelProject()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::warning(this, "", "将删除该项目包含的所有子项目\n 是否继续？", QMessageBox::Ok | QMessageBox::No, QMessageBox::No);
    if(reply == QMessageBox::Ok){
        QModelIndex currentIndex = treeview->currentIndex();
        QStandardItem* currentItem = model->itemFromIndex(currentIndex);
        model->removeRow(currentItem->row());
    }
    else{
        //do nothing.
    }
}

//clear清空类别函数实现
void knn::clear()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::warning(this, "", "将删除该项目包含的所有子项目\n 是否继续？", QMessageBox::Ok | QMessageBox::No, QMessageBox::No);
    if(reply == QMessageBox::Ok){
        QModelIndex currentIndex = treeview->currentIndex();
        QStandardItem* currentItem = model->itemFromIndex(currentIndex);
        int count=currentItem->rowCount();

        for(int j=0;j<count;j++)
        {
            QStandardItem *child=currentItem->child(j,0);
            Data p;
            p.info = currentIndex.row()+1;
            QString str = child->index().data().toString();
            QString x="";
            QString y="";
            for(int i=0;i<str.length();i++)
            {
                if(str[i]==',')
                {
                    break;
                }
                else
                {
                    x= x + str[i];
                }
            }
            for(int j=str.length()-1;j>0;j--)
            {
                if(str[j]==',')
                {
                    break;
                }
                else
                {
                    y=  str[j]+y;
                }
            }
            p.point.setX(x.toInt());
            p.point.setY(y.toInt());

            Data temp;
            for(int i = 0; i < ttar.size(); i++)
            {
                temp = ttar[i];
                if(p.info == temp.info &&
                    p.point == temp.point){
                    ttar.erase(ttar.begin() + i);
                    emit refresh();
                }
            }
        }
        currentItem->removeRows(0,count);
    }
    else{
        //Do nothing.
    }
}

//右键添加像素点函数实现
void knn::Add()
{
    Global::ispress=true;
    Global::drag_position = 2;

    QModelIndex currentIndex = treeview->currentIndex();

    emit choose_position_changed();
    emit sendData(currentIndex.row());                  //传当前类别到imagebox
}
//接收imagebox所选取的像素点并添加到类别
void knn::reciveData(QPoint point)
{
    //QMessageBox::information(NULL,NULL,"f",NULL);
    QString str=QString::number(point.x())+","+QString::number(point.y());
    QStandardItemModel* model = static_cast<QStandardItemModel*>(treeview->model());
    QModelIndex currentIndex = treeview->currentIndex();
    QStandardItem* currentItem = model->itemFromIndex(currentIndex);
    if(currentItem->data(ROLE_MARK)!=currentItem->data(ROLE_MARK_FOLDER))
    {
        QStandardItem *root = new QStandardItem(str);   //iamge的目录
        root->setData(MARK_FOLDER,ROLE_MARK);           //首先它是文件夹
        root->setData(MARK_FOLDER,ROLE_MARK_FOLDER);    //其次它属于文件夹的子项目
        currentItem->appendRow(root);
        treeview->expand(currentIndex);
    }
}

//右键移除像素点函数实现
void knn::remove()
{
    QStandardItemModel* model = static_cast<QStandardItemModel*>(treeview->model());
    QModelIndex currentIndex = treeview->currentIndex();
    QStandardItem* currentItem = model->itemFromIndex(currentIndex);
    QString str = treeview->currentIndex().data().toString();
    Data p;
    if(currentItem->data(ROLE_MARK)!=currentItem->data(ROLE_MARK_FOLDER))
    {
         p.info = currentIndex.row()+1;
    }
    else
    {
         p.info=currentItem->parent()->index().row()+1;
    }

    QString x="";
    QString y="";
    for(int i=0;i<str.length();i++)
    {
        if(str[i]==',')
        {
            break;
        }
        else
        {
            x= x + str[i];
        }
    }
    for(int j=str.length()-1;j>0;j--)
    {
        if(str[j]==',')
        {
            break;
        }
        else
        {
            y=  str[j]+y;
        }
    }
    p.point.setX(x.toInt());
    p.point.setY(y.toInt());

    Data temp;
    for(int i = 0; i < ttar.size(); i++){
        temp = ttar[i];
        if(p.info == temp.info &&
                p.point == temp.point){
            ttar.erase(ttar.begin() + i);
            emit refresh();
        }
    }
    //file.del(file.match(str));

    //从TreeView中删除节点
    QStandardItem *parent_Item=currentItem->parent();
    parent_Item->removeRow(currentItem->row());
}

//右键折叠全部
void knn::fold()
{
   treeview->collapseAll();
}

//点击确定按钮事件
void knn::on_buttonBox_accepted()
{
    int countofmodel=model->rowCount();
    //model->removeRows(1,countofmodel);
    for(int i=0;i<countofmodel-1;i++)
    {
        model->removeRow(1);
    }
    count=0;
    QStandardItem *itemone=new QStandardItem;
    QModelIndex index=model->index(0,0);
    itemone=model->itemFromIndex(index);
    int countofpoint=itemone->rowCount();
    itemone->removeRows(0,countofpoint);
    //ttar.clear();
    emit choose_position_nonchanged();
    //QMessageBox::information(NULL,NULL,QString::number(countofmodel),NULL);
    emit sendarray(countofmodel);
}


void knn::on_buttonBox_rejected()
{

    int countofmodel=model->rowCount();
    //model->removeRows(1,countofmodel);
    for(int i=0;i<countofmodel-1;i++)
    {
        model->removeRow(1);
    }
    count=0;
    QStandardItem *itemone=new QStandardItem;
    QModelIndex index=model->index(0,0);
    itemone=model->itemFromIndex(index);
    int countofpoint=itemone->rowCount();
    itemone->removeRows(0,countofpoint);
    ttar.clear();
    emit choose_position_nonchanged();
}

void knn::getrequire(){
    QModelIndex currentIndex = treeview->currentIndex();
    emit sendData(currentIndex.row());
}
