#include "mainwindow.h"
#include "ui_mainwindow.h"

#include<QDebug>
#include<QInputDialog>
#include<QProgressDialog>

#include "knn.h"

using namespace std;
using namespace cv;
const double scale = 0.5;

//=============ROLE_MARK是一个Qt的角色，用于区分项目和图片两种角色==============
#define ROLE_MARK Qt::UserRole + 1
#define ROLE_MARK_FOLDER Qt::UserRole + 2

//=============MARK用于区分项目和图片==============
#define MARK_PROJECT 1
#define MARK_FOLDER 2

//=============初始聚类数=========================
#define iniClusters 5  //初始类聚的个数

//=================ISODATA=======================
//定义6个使用的参数
struct Args
{
    int expClusters;   //期望得到的聚类数
    int thetaN;        //聚类中最少样本数
    int maxIts;        //最大迭代次数
    int combL;         //每次迭代允许合并的最大聚类对数
    double thetaS;     //标准偏差参数
    double thetaC;     //合并参数
}args;
struct covarry{          //最大似然分类的结构体
    Mat covar; //协方差
    Mat means;  //特征均值
    double hcovar;  //行列式
    Mat zmatrix;    //协方差的逆
    int label;      //标签

    //计算概率
    void calculatep(Mat matData)
    {
        //calcCovarMatrix(data, covar, means, CV_COVAR_NORMAL | CV_COVAR_ROWS);
        calcCovarMatrix(matData,covar,means,CV_COVAR_NORMAL |CV_COVAR_ROWS);
        hcovar=determinant(covar);     //协方差行列式的值
        zmatrix=covar.inv();
        //cvInvert( covar, zmatrix,0 );    //求协方差的逆矩阵并存入zmatrix中
    }
};
//需要合并的两个类聚的信息，包括两个类聚的id和距离
struct MergeInfo
{
    int u, v;
    double d;    //类聚u中心与类聚v中心的距离
};

//定义比较函数
bool cmp(MergeInfo a, MergeInfo b)
{
    return a.d < b.d;
}

//计算两点之间距离
double dist(float a, float b)
{
    return sqrt((a-b)*(a-b));
}

//声明每一类中的相关变量，并求类的均值及标准差
struct Cluster
{
    int nSamples;          //样本点的个数
    double avgDist;        //样本点到样本中心的平均距离
    float center;          //样本中心
    float sigma;           //样本与中心的标准差
    vector<float> data;  //聚类的数据
    vector<int> label;

    //计算该聚类的中心，即该类的均值
    void calMean()
    {
        assert(nSamples == data.size());
        for(int i = 0; i < nSamples; i++)
        {
            center += data.at(i);
        }
        center/= nSamples;
    }

    //计算该类样本点到该聚类中心得平均距离
    void calDist()
    {
        avgDist = 0;
        for(int i = 0; i < nSamples; i++)
            avgDist += dist((data.at(i)), center);
        avgDist /= nSamples;
    }

    //计算样本与中心的标准差
    void calStErr()
    {
        assert(nSamples == data.size());
        double attr1 = 0;
        for(int i = 0; i < nSamples; i++)
        {
            attr1 += (data.at(i) - center) * (data.at(i) - center);
        }
        sigma = sqrt(attr1 / nSamples);
    }
};

//设置参数的值
void setArgs(int newexpClusters = 5,int newthetaN = 3, int newmaxIts = 3, int newcombL = 10, double newthetaS = 3, double newthetaC = 0.01)
{
    args.expClusters = newexpClusters;   //期望聚类的数目
    args.thetaN = newthetaN;       //每个类别中最少的样本数，少于次数则去掉该类别
    args.maxIts = newmaxIts;       //迭代运算的次数
    args.combL = newcombL;       //在一次合并操作中可以合并的类别的最多对数
    args.thetaS = newthetaS;        //样本中最大标准差，若大于它则分裂
    args.thetaC = newthetaC;    //两个类别中心间的最小距离，若小于它合并
}

//寻找点t距离最近的类的中心对应的id
int FindIdx(vector<Cluster> &c, float &t)
{
    int nClusters = c.size();
    assert(nClusters >= 1);
    double ans = dist(c.at(0).center, t);
    int idx = 0;
    for(int i = 1; i < nClusters; i++)
    {
        double tmp = dist(c.at(i).center, t);
        if(ans > tmp)
        {
            idx = i;
            ans = tmp;
        }
    }
    return idx;
}

//二分法寻找距离刚好小于thetaC的两个类聚的index
int FindPos(MergeInfo *info, int n, double thetaC)
{
    int l = 0;
    int r = n - 1;
    while(l <= r)
    {
        int mid = (l + r) >> 1;
        if(info[mid].d < thetaC)
        {
            l = mid + 1;
            if(l < n && info[l].d >= thetaC)
                return mid;
        }
        else
        {
            r = mid - 1;
            if(r >= 0 && info[r].d < thetaC)
                return r;
        }
    }
    if(info[n - 1].d < thetaC)
    {
        return n - 1;
    }
    else
    {
        return -1;
    }
}

//标签处理
void Getlabel(const vector<Cluster> c,vector<int>&labels)
{
    int st=0;
    int n = c.size();
    for(int i = 0 ; i < c.size(); i++)
    {
        for(int j = 0; j < c.at(i).label.size(); j++)
        {
            st=(int)c.at(i).label.at(j);
            labels[st]=i+1;
        }

    }
}
//=================ISODATA结束======================

void MainWindow::ISOData(int state, Mat image,int newexpClusters ,int newthetaN , int newmaxIts, int newcombL , double newthetaS , double newthetaC, QString str, QString filePath )
{
    Scalar colorTab[] =     //10个颜色(染色),即最多可分10个类别
    {
        Scalar(0, 0, 255),
        Scalar(0, 255, 0),
        Scalar(255, 100, 100),
        Scalar(255, 0, 255),
        Scalar(0, 255, 255),
        Scalar(255, 0, 0),
        Scalar(255, 255, 0),
        Scalar(255, 0, 100),
        Scalar(100, 100, 100),
        Scalar(50, 125, 125),
    };
    int n=0;

    cvtColor(image,image,CV_BGR2GRAY);
    int rows=image.rows;
    int cols=image.cols;
    //计算图片大小
    n=rows*cols;
    //imagedata用于保存所有的灰度像素
    float idata;
    vector<float>imagedata;
    for (int i = 0; i < rows;++i)
    {
        for (int j = 0; j < cols;++j)
        {
            idata=image.at<uchar>(i,j);
            imagedata.push_back(idata);
        }
    }

    //初始化参数
    setArgs(newexpClusters, newthetaN, newmaxIts, newcombL, newthetaS, newthetaC);
    vector<int>labels(n,0);
    //分类
    //cout << "ISOData is processing......." << endl;
    vector<Cluster> c;              //每个类聚的数据
    const double split = 0.5;       //分裂常数(0,1]
    int nClusters = iniClusters;    //初始化类聚个数

    //初始化nClusters个类，设置相关数据
    for(int i = 0; i < nClusters; i++)
    {
        Cluster t;
        t.center = imagedata[i];
        t.nSamples = 0;
        t.avgDist = 0;
        c.push_back(t);
    }

    int iter = 0;
    bool isLess = false;            //标志是否有类的数目低于thetaN
    progress->setValue(36);
    while(1)
    {
        //先清空每一个聚类
        for(int i = 0; i < nClusters; i++)
        {
            c.at(i).nSamples = 0;
            c.at(i).data.clear();
            c.at(i).label.clear();    //标签
        }

        //将所有样本划分到距离类聚中心最近的类中
        float f=0;
        for(int i = 0; i < n; i++)
        {
            f=imagedata[i];
            int idx = FindIdx(c, f);
            c.at(idx).data.push_back(imagedata[i]);
            c.at(idx).label.push_back(i);   //标签
            c.at(idx).nSamples++;
        }

        int k = 0;                   //记录样本数目低于thetaN的类的index
        for(int i = 0; i < nClusters; i++)
        {
            if(c.at(i).data.size() < args.thetaN)
            {
                isLess = true;       //说明样本数过少，该类应该删除
                k = i;
                break;
            }
        }

        //如果有类的样本数目小于thetaN
        if(isLess)
        {
            nClusters--;
            Cluster t = c.at(k);
            vector<Cluster>::iterator pos = c.begin() + k;
            c.erase(pos);
            assert(nClusters == c.size());
            for(int i = 0; i < t.data.size(); i++)
            {
                int idx = FindIdx(c, (t.data.at(i)));
                c.at(idx).data.push_back(t.data.at(i));
                c.at(idx).label.push_back(t.label.at(i)); //标签
                c.at(idx).nSamples++;
            }
            isLess = false;
        }

        //重新计算均值和样本到类聚中心的平均距离
        for(int i = 0; i < nClusters; i++)
        {
            c.at(i).calMean();
            c.at(i).calDist();
        }

        //计算总的平均距离
        double totalAvgDist = 0;
        for(int i = 0; i < nClusters; i++)
            totalAvgDist += c.at(i).avgDist * c.at(i).nSamples;
        totalAvgDist /= n;

        if(iter >= args.maxIts) break;

        //分裂操作
        if(nClusters <= args.expClusters / 2)
        {
            vector<double> maxsigma;
            for(int i = 0; i < nClusters; i++)
            {
                //计算该类的标准偏差
                c.at(i).calStErr();
                //计算该类标准差的最大分量(?????)
                double mt = c.at(i).sigma;
                maxsigma.push_back(mt);
            }
            for(int i = 0; i < nClusters; i++)
            {
                if(maxsigma.at(i) > args.thetaS)
                {
                    if((c.at(i).avgDist > totalAvgDist && c.at(i).nSamples > 2 * (args.thetaN + 1)) || (nClusters < args.expClusters / 2))
                    {
                        nClusters++;
                        Cluster newCtr;     //新的聚类中心
                        //获取新的中心
                        newCtr.center = c.at(i).center - split * c.at(i).sigma;
                        c.push_back(newCtr);
                        //改变老的中心
                        c.at(i).center = c.at(i).center + split * c.at(i).sigma;
                          break;
                    }
                }
            }
        }

        //合并操作
        if(nClusters >= 2 * args.expClusters || (iter & 1) == 0)
        {
            int size = nClusters * (nClusters - 1);
            //需要合并的聚类个数
            int cnt = 0;
            MergeInfo *info = new MergeInfo[size];
            for(int i = 0; i < nClusters; i++)
            {
                for(int j = i + 1; j < nClusters; j++)
                {
                    info[cnt].u = i;
                    info[cnt].v = j;
                    info[cnt].d = dist(c.at(i).center, c.at(j).center);
                    cnt++;
                }
            }
            //进行排序
            sort(info, info + cnt, cmp);
            //找出info数组中距离刚好小于thetaC的index，那么index更小的更应该合并
            int iPos = FindPos(info, cnt, args.thetaC);

            //用于指示该位置的样本点是否已经合并
            bool *flag = new bool[nClusters];
            memset(flag, false, sizeof(bool) * nClusters);
            //用于标记该位置的样本点是否已经合并删除
            bool *del = new bool[nClusters];
            memset(del, false, sizeof(bool) * nClusters);
            //记录合并的次数
            int nTimes = 0;

            for(int i = 0; i <= iPos; i++)
            {
                int u = info[i].u;
                int v = info[i].v;
                //确保同一个类聚只合并一次
                if(!flag[u] && !flag[v])
                {
                    nTimes++;
                    //如果一次迭代中合并对数多于combL，则停止合并
                    if(nTimes > args.combL) break;
                    //将数目少的样本合并到数目多的样本中
                    if(c.at(u).nSamples < c.at(v).nSamples)
                    {
                        del[u] = true;
                        Cluster t = c.at(u);
                        assert(t.nSamples == t.data.size());
                        for(int j = 0; j < t.nSamples; j++)
                        {
                            c.at(v).data.push_back(t.data.at(j));
                            c.at(v).label.push_back(t.label.at(j));  //标签
                        }
                        c.at(v).center = c.at(v).center * c.at(v).nSamples + t.nSamples * t.center;
                        c.at(v).nSamples += t.nSamples;
                        c.at(v).center /= c.at(v).nSamples;
                     }
                    else
                    {
                        del[v] = true;
                        Cluster t = c.at(v);
                        assert(t.nSamples == t.data.size());
                        for(int j = 0; j < t.nSamples; j++)
                        {
                            c.at(u).data.push_back(t.data.at(j));
                            c.at(u).label.push_back(t.label.at(j));  //标签
                        }
                        c.at(u).center = c.at(u).center * c.at(u).nSamples + t.nSamples * t.center;
                        c.at(u).nSamples += t.nSamples;
                        c.at(u).center /= c.at(u).nSamples;
                    }
                }
            }

            //删除合并后的聚类
            vector<Cluster>::iterator id = c.begin();
            for(int i = 0; i < nClusters; i++)
            {
                if(del[i])
                    id = c.erase(id);
                else
                    id++;
            }

            //合并多少次就删除多少个
            nClusters -= nTimes;
            assert(nClusters == c.size());
            delete[] info;
            delete[] flag;
            delete[] del;
            info = NULL;
            flag = NULL;
            del = NULL;
        }

        if(iter >= args.maxIts) break;
        iter++;
    }
    progress->setValue(98);
    assert(nClusters == c.size());
    Getlabel(c,labels);

    //分完类后处理
    vector<float>labelnumber(10,0);    //记录每一类中的像素点数目
    Mat clusteredMat(image.rows,image.cols,CV_32FC3);
    //标记像素点的类别，颜色区分
    for (int i = 0; i < rows;++i)
    {
        for (int j = 0; j < cols;++j)
        {
            labelnumber[labels[i*cols+j]]++;
            circle(clusteredMat, Point(j,i), 1, colorTab[labels[i*cols + j]]);        //标记像素点的类别，颜色区分
        }
    }
    QString info;   //此处的info是收集的分类信息
    //输出每一类中的像素点数目
    for(int i=0; i<newexpClusters;i++)
    {
        //if(!labelnumber[i]){
        //    break;
        //}
        //cout<<labelnumber[i]<<endl;
        //else{
        //    info+="第"+QString::number(i)+"类像素点个数:  "+QString::number(labelnumber[i])+"\n";
        //}
        info+="第"+QString::number(i + 1)+"类像素点个数:  "+QString::number(labelnumber[i])+"\n";
    }
    if(state == 1){//处理单张图片
        progress->setValue(100);
        QMessageBox::about(NULL,str + "分类结果信息",info);

        str = str + " @ ISODATA";

        namedWindow(str.toStdString(), CV_WINDOW_FREERATIO);
        imshow(str.toStdString(),clusteredMat);
    }
    else{
        str = filePath + "/iso_" + str;

        //imwrite(str.toStdString(), clusteredMat);

        //此处开始写文件
        str = str + ".txt";
        QFile _f(str);
        if(!_f.open(QIODevice::ReadWrite | QIODevice::Text))
        {
           QMessageBox::warning(this,"","错误代码 0006",QMessageBox::Yes);
        }
        QTextStream _in(&_f);
        _in<<info;
        _f.close();
    }
    sizeLabel->setText("处理完成");

    progress->setVisible(false);
    progress->setValue(0);
    infoLabel->setVisible(true);
    //waitKey(0);
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //此处为左侧边导航栏
    ui->dockWidget->addtreeview();      //初始化treeview
    ui->dockWidget->update();           //初始化后更新

    //new一个dialog类并将类中的值传回
    dlg = new Dialog;
    this->connect(dlg,SIGNAL(sendData(QString,QString,QString,QString,QString,QString)),this,SLOT(receiveData(QString,QString,QString,QString,QString,QString)));

    //knn槽函数连接
    k = new knn();
    this->connect(k,SIGNAL(sendData(int)),ui->widget,SLOT(receive(int)));
    this->connect(ui->widget,SIGNAL(sendData(QPoint)),k,SLOT(reciveData(QPoint)));
    this->connect(k,SIGNAL(sendarray(int)),this,SLOT(receivearray(int)));
    //删除的点实时刷新到画布
    this->connect(k, SIGNAL(refresh()), ui->widget, SLOT(rrefresh()));
    this->connect(k, SIGNAL(choose_position_changed()), this, SLOT(choose_position_changed()));
    this->connect(k, SIGNAL(choose_position_nonchanged()), this, SLOT(choose_position_nonchanged()));
    this->connect(ui->widget, SIGNAL(require()), k, SLOT(getrequire()));

    //UI刷新连接
    this->connect(this, SIGNAL(refresh()), ui->widget, SLOT(rrefresh()));

    //双击槽函数信号连接
    this->connect(ui->dockWidget->treeview,SIGNAL(doubleClicked(QModelIndex)),this,SLOT(OnlineTreeDoubleClick(QModelIndex)));

    //设置treeview属性为自定义菜单
    ui->dockWidget->treeview->setContextMenuPolicy(Qt::CustomContextMenu);

    //右键菜单槽函数连接
    this->connect(ui->dockWidget->treeview,SIGNAL(customContextMenuRequested(QPoint)),this,SLOT(showmenu(QPoint)));

//============================添加两个不同的菜单============================//
    m_projectMenu = new QMenu(this);
    m_itemMenu = new QMenu(this);

    //项目右键添加图片事件
    QAction* newpic  = new QAction(QStringLiteral("添加新图片"),this);
    m_projectMenu->addAction(newpic);
    this->connect(newpic,SIGNAL(triggered(bool)),this,SLOT(Add()));
    clear1 = new QAction(QStringLiteral("清空该项目"),this);
    m_projectMenu->addAction(clear1);
    this->connect(clear1,SIGNAL(triggered(bool)),SLOT(clear()));

    //添加新项目事件
    QAction* NewProject = new QAction(QStringLiteral("添加新项目"),this);
    m_projectMenu->addAction(NewProject);
    this->connect(NewProject,SIGNAL(triggered(bool)),this,SLOT(NewProject()));

    //删除项目事件
    delProject = new QAction(QStringLiteral("删除该项目"),this);
    m_projectMenu->addAction(delProject);
    this->connect(delProject,SIGNAL(triggered(bool)),this,SLOT(DelProject()));

//==================以上时针对项目右键菜单，完成。。。。。。========================
    //打开图片按钮
    QAction *open = new QAction(QStringLiteral("打开"),this);
    m_itemMenu->addAction(open);
    this->connect(open,SIGNAL(triggered(bool)),this,SLOT(openshow()));      //此处修改为打开多个文件

    //添加图片按钮
    QAction *add = new QAction(QStringLiteral("添加"),this);
    m_itemMenu->addAction(add);
    this->connect(add,SIGNAL(triggered(bool)),this,SLOT(Image_Add()));      //此处修改为打开多个文件


     //移除图片按钮
    QAction *remove = new QAction(QStringLiteral("移除"),this);
    m_itemMenu->addAction(remove);
    this->connect(remove,SIGNAL(triggered(bool)),this,SLOT(remove()));      //连接remove事件，此处需将该信号连接到删除File类中的数据


     //折叠treeview全部子项按钮
    QAction *fold = new QAction(QStringLiteral("折叠全部"),this);
    m_itemMenu->addAction(fold);
    this->connect(fold,SIGNAL(triggered(bool)),SLOT(fold()));

    //底部状态栏
    sizeLabel = new QLabel("");
    infoLabel = new QLabel("Classfication beta 0.0.0");
    progress = new QProgressBar();

    //progress
    statusBar()->addPermanentWidget(progress);
    progress->setVisible(false);

    //sizelabel
    sizeLabel->setMinimumSize(sizeLabel->sizeHint());
    sizeLabel->setAlignment(Qt::AlignCenter);
    statusBar()->addWidget(sizeLabel);

    //infolabel
    infoLabel->setMinimumSize(infoLabel->sizeHint());
    infoLabel->setAlignment(Qt::AlignCenter);
    statusBar()->addPermanentWidget(infoLabel);

    statusBar()->setStyleSheet(QString("QStatusBar::item{border: 0px}"));   // 设置不显示label的边框
    statusBar()->setSizeGripEnabled(true);                                  //设置是否显示右边的大小控制点

    setWindowTitle("Classfication beta 0.0.0");
    setWindowIcon(QIcon(":/pic/main/main"));

    //任务栏托盘菜单
    createMenu();
    if(!QSystemTrayIcon::isSystemTrayAvailable()){
        return;
    }
    myTrayIcon = new QSystemTrayIcon(this);
    myTrayIcon->setIcon(QIcon(":/pic/main/main"));
    myTrayIcon->setToolTip("Classfication beta 0.0.0");
    myTrayIcon->setContextMenu(myMenu);
    myTrayIcon->show();

    //设置图像操作按钮
    switch(Global::drag_position){
    case 0 :
        ui->actionCursor->setChecked(true);
        break;
    case 1:
        ui->actionHands->setChecked(true);
        break;
    }
    ui->actionChoose_position->setVisible(false);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionOpen_triggered()
{
    QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
    QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
    QStandardItem* currentItem = model->itemFromIndex(currentIndex);

    QFileDialog::Options options;
        //options |= QFileDialog::DontUseNativeDialog;

    QString selectedFilter;
    //filenameList用来存储读进的多个图片路径
    filenameList = QFileDialog::getOpenFileNames(
                    this, tr("导入文件"),
                    "C:/Users/Public/Pictures",
                    tr("ImageFiles(*.jpg *.png *.bmp)"),
                    &selectedFilter,
                    options);
    if(!filenameList.empty()){
        file.conn(filenameList);//将图片路径push到vector数组filenameList中
        for(int i = 0; i < filenameList.count(); i++){
            QString s=ui->dockWidget->getImageName(filenameList[i]);        //调用Dockwidget类的函数截取图片名

//======在此处可将图片名push,图片名为s=============================
            QStandardItem *image=new QStandardItem(s);                      //新建一个item，并以图片名称来命名
            image->setIcon(QIcon(filenameList[i]));                         //设置图片预览图标
            image->setData(MARK_FOLDER,ROLE_MARK);                          //首先它是文件夹
            image->setData(MARK_FOLDER,ROLE_MARK_FOLDER);                   //其次它属于文件夹的子项目
            currentItem->appendRow(image);
            //ui->dockWidget->appendrow(image);
            ui->dockWidget->treeview->expandAll();
            ui->dockWidget->update();
        }
    }
}

void MainWindow::on_actionSave_triggered()
{
    QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
    QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
    QStandardItem* currentItem = model->itemFromIndex(currentIndex);
    if(file.isEmpty()){
        QMessageBox::warning(this, "", "错误代码 0000", QMessageBox::Ok);
    }
    else{
        if(!currentIndex.isValid())
        {
            QMessageBox::warning(this, "", "错误代码 0000", QMessageBox::Ok);
        }

        else if(currentItem->data(ROLE_MARK)!=currentItem->data(ROLE_MARK_FOLDER))
        {
            int count=currentItem->rowCount();
            for(int i=0;i<count;i++)
            {
               int j=file.match(currentItem->child(i,0)->index().data().toString());
               file.save("", j);
            }

             QMessageBox::information(this, "", "保存成功", QMessageBox::Ok);
        }
        else
        {
            int j=file.match(ui->dockWidget->treeview->currentIndex().data().toString());
            file.save("",j);
            QMessageBox::information(this, "", "保存成功", QMessageBox::Ok);
        }
    }
}

void MainWindow::on_actionSave_As_triggered()
{
    QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
    QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
    QStandardItem* currentItem = model->itemFromIndex(currentIndex);
    if(file.isEmpty()){
        QMessageBox::warning(this, "", "错误代码 0000", QMessageBox::Ok);
    }
    else{
        if(!currentIndex.isValid())
        {
            QMessageBox::warning(this, "", "错误代码 0000", QMessageBox::Ok);
        }

        else if(currentItem->data(ROLE_MARK)!=currentItem->data(ROLE_MARK_FOLDER))
        {
            bool s=true;
            QString filePath = QFileDialog::getExistingDirectory(this, tr("选择保存位置"), "C:/Users/Public/Pictures");
            int count=currentItem->rowCount();
            for(int i=0;i<count;i++)
            {
               int j=file.match(currentItem->child(i,0)->index().data().toString());
               if(!file.save_as(filePath, j)){
                   s=false;
               }
            }
             if(s)
             {
                 QMessageBox::information(this, "", "保存成功", QMessageBox::Ok);
             }
             else
             {
                 QMessageBox::information(this, "", "未知错误", QMessageBox::Ok);
             }
        }
        else
        {
            QString filePath = QFileDialog::getExistingDirectory(this, tr("选择保存位置"), "C:/Users/Public/Pictures");
            int j=file.match(ui->dockWidget->treeview->currentIndex().data().toString());
            if(file.save_as(filePath, j)){
                QMessageBox::information(this, "", "保存成功", QMessageBox::Ok);
            }
            else{
                QMessageBox::information(this, "", "未知错误", QMessageBox::Ok);
            }
        }



    }
}

void MainWindow::on_actionCursor_triggered()
{
    ui->actionChoose_position->setChecked(false);
    ui->actionHands->setChecked(false);
    Global::drag_position = 0;
}

void MainWindow::on_actionHands_triggered()
{
    ui->actionChoose_position->setChecked(false);
    ui->actionCursor->setChecked(false);
    Global::drag_position = 1;
}

void MainWindow::on_actionChoose_position_triggered()
{
    ui->actionCursor->setChecked(false);
    ui->actionHands->setChecked(false);
    Global::drag_position = 2;
}

void MainWindow::on_actionZoom_In_triggered()
{
    ui->widget->zoomIn();
}

void MainWindow::on_actionZoom_Out_triggered()
{
    ui->widget->zoomOut();
}

void MainWindow::on_actionNormal_Size_triggered()
{
    if(currentPixmap.isNull()){
        QMessageBox::warning(this, "", "错误代码 0002", QMessageBox::Ok);
    }
    else{
        ui->widget->setPixmap(currentPixmap);
    }
}

void MainWindow::on_actionWise_clock_triggered()
{
    ui->widget->clockWise();
}

void MainWindow::on_actionWise_anticlock_triggered()
{
    ui->widget->anticlockWise();
}

//任务栏托盘菜单
void MainWindow::createMenu(){
    restoreWinAction = new QAction("打开主窗口", this);
    quitAction = new QAction("退出", this);

    connect(restoreWinAction, SIGNAL(triggered()), this, SLOT(showNormal()));
    connect(quitAction, SIGNAL(triggered()), qApp, SLOT(quit()));

    myMenu = new QMenu((QWidget*)QApplication::desktop());
    myMenu->addAction(restoreWinAction);
    myMenu->addSeparator();
    myMenu->addAction(quitAction);
}

void MainWindow::showNormal(){
    this->showMaximized();
}

void QWidget::changeEvent(QEvent *e){
    Q_UNUSED(e);

    if((e->type() == QEvent::WindowStateChange) && this->isMinimized()){
        this->hide();
    }
}

//鼠标双击函数
void MainWindow::OnlineTreeDoubleClick(const QModelIndex &index)
{
    QString str;
    QString name;
                //info;
    if(index.column() == 0)
    {
        name = index.data().toString();
        //info = index.sibling(index.row(),1).data().toString();
    }
    else
    {
        name = index.sibling(index.row(),0).data().toString();
        //info = index.data().toString();
    }
    str=name;
    int i = file.match(str);//查找图片名称对应的index
    if(i != -1){
       ui->widget->setPixmap(file.pixmapList[i]);
       currentPixmap = file.pixmapList[i];

       sizeLabel->setText("图片尺寸：  " + QString::number(ui->widget->m_pixmap.width())
                          + " x " + QString::number(ui->widget->m_pixmap.height()));
       infoLabel->setText("图片路径：  " + file.filenameList[i]);
    }
}

//showmenu菜单显示实现函数
void MainWindow::showmenu(const QPoint &pos)
{
        QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
        QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
        QModelIndex index = ui->dockWidget->treeview->indexAt(pos);
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

//remove删除节点函数
void MainWindow::remove()
{
    QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
    QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
    QStandardItem* currentItem = model->itemFromIndex(currentIndex);
    //ui->dockWidget->root->removeRow(currentItem->row());

    //从File中删除节点
    QString str = ui->dockWidget->treeview->currentIndex().data().toString();
    file.del(file.match(str));

    //从TreeView中删除节点
    QStandardItem *parent_Item=currentItem->parent();
    parent_Item->removeRow(currentItem->row());
}

//fold折叠全部函数实现
void MainWindow::fold()
{
    ui->dockWidget->treeview->collapseAll();
}

//clear清空项目函数实现
void MainWindow::clear()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::warning(this, "", "将删除该项目包含的所有子项目\n 是否继续？", QMessageBox::Ok | QMessageBox::No, QMessageBox::No);
    if(reply == QMessageBox::Ok){
        QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
        QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
        QStandardItem* currentItem = model->itemFromIndex(currentIndex);

        //从File中获取文件
        int count=currentItem->rowCount();
        //ui->dockWidget->root->removeRows(0,count);
        currentItem->removeRows(0,count);
    }
    else{
        //Do nothing.
    }
}

void MainWindow::openshow()
{
    QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
    QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
    //右键打开之后获取当前点击的位置的图片名
    QString src_pic=currentIndex.sibling(currentIndex.row(),0).data().toString();
    int i = file.match(src_pic);//查找图片名称对应的index
    if(i != -1){
       ui->widget->setPixmap(file.pixmapList[i]);
    }
}

//新建项目函数实现
void MainWindow::NewProject()
{
    int count=ui->dockWidget->count;
    QStandardItem *newproject=new QStandardItem(QIcon(":/pic/icon/file"),QStringLiteral("项目%1").arg(count+2));
    newproject->setData(MARK_PROJECT,ROLE_MARK);        //标识
    newproject->setData(MARK_FOLDER,ROLE_MARK_FOLDER);  //标识
    ui->dockWidget->model->appendRow(newproject);
    ui->dockWidget->count++;

}

//删除项目函数实现
void MainWindow::DelProject()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::warning(this, "", "将删除该项目包含的所有子项目\n 是否继续？", QMessageBox::Ok | QMessageBox::No, QMessageBox::No);
    if(reply == QMessageBox::Ok){
        QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
        QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
        QStandardItem* currentItem = model->itemFromIndex(currentIndex);
        ui->dockWidget->model->removeRow(currentItem->row());
    }
    else{
        //do nothing.
    }
}

//右键项目添加图片函数实现
void MainWindow::Add()
{
    QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
    QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
    QStandardItem* currentItem = model->itemFromIndex(currentIndex);
    QFileDialog::Options options;
        //options |= QFileDialog::DontUseNativeDialog;

    QString selectedFilter;
    filenameList = QFileDialog::getOpenFileNames(
                    this, tr("导入文件"),
                    "C:/Users/Public/Pictures",
                    tr("ImageFiles(*.jpg *.png *.bmp)"),
                    &selectedFilter,
                    options);
    if(!filenameList.empty()){
        file.conn(filenameList);
        for(int i = 0; i < filenameList.count(); i++){
            QString s=ui->dockWidget->getImageName(filenameList[i]);    //调用Dockwidget类的函数截取图片名
//======在此处可将图片名push,图片名为s=============================
            QStandardItem *image=new QStandardItem(s);                  //新建一个item，并以图片名称来命名
            image->setIcon(QIcon(filenameList[i]));                     //设置图片预览图标
            image->setData(MARK_FOLDER,ROLE_MARK);                      //首先它是文件夹
            image->setData(MARK_FOLDER,ROLE_MARK_FOLDER);               //其次它属于文件夹的子项目
            currentItem->appendRow(image);
            //ui->dockWidget->appendrow(image);
            ui->dockWidget->treeview->expand(currentIndex);
            ui->dockWidget->update();
        }
    }
}

//右键图片添加
void MainWindow::Image_Add()
{
    QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
    QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
    QStandardItem* currentItem = model->itemFromIndex(currentIndex);
    QStandardItem* parent_Item=currentItem->parent();
    QFileDialog::Options options;
        //options |= QFileDialog::DontUseNativeDialog;

    QString selectedFilter;
    filenameList = QFileDialog::getOpenFileNames(
                    this, tr("导入文件"),
                    "C:/Users/Public/Pictures",
                    tr("ImageFiles(*.jpg *.png *.bmp)"),
                    &selectedFilter,
                    options);
    if(!filenameList.empty()){
        file.conn(filenameList);
        for(int i = 0; i < filenameList.count(); i++){
            QString s=ui->dockWidget->getImageName(filenameList[i]);    //调用Dockwidget类的函数截取图片名
//======在此处可将图片名push,图片名为s=============================
            QStandardItem *image=new QStandardItem(s);                  //新建一个item，并以图片名称来命名
            image->setIcon(QIcon(filenameList[i]));                     //设置图片预览图标
            image->setData(MARK_FOLDER,ROLE_MARK);                      //首先它是文件夹
            image->setData(MARK_FOLDER,ROLE_MARK_FOLDER);               //其次它属于文件夹的子项目
            //currentItem->appendRow(image);
            parent_Item->appendRow(image);
            ui->dockWidget->treeview->expandAll();
            ui->dockWidget->update();
        }
    }
}

Mat MainWindow::K_means(const Mat &src,int k,int t,int Count[])
{
    Scalar colorTab[] =                                                 //10个颜色(染色),即最多可分10个类别
    {
       Scalar(0, 0, 255),
       Scalar(0, 255, 0),
       Scalar(255, 100, 100),
       Scalar(255, 0, 255),
       Scalar(0, 255, 255),
       Scalar(255, 0, 0),
       Scalar(255, 255, 0),
       Scalar(255, 0, 100),
       Scalar(100, 100, 100),
       Scalar(50, 125, 125),
    };
    Mat labels;    //标签
    //转为灰度图
    Mat image;
   // if (image.channels() != 1)
        cvtColor(src, image, COLOR_BGR2GRAY);

    progress->setValue(24);

    int rows = image.rows;               //获取图像横向大小
    int cols = image.cols;               //获取图像纵向大小

    //保存聚类后的图片
    Mat clusteredMat(rows, cols, CV_8UC3);
    clusteredMat.setTo(Scalar::all(0));

    Mat pixels(rows*cols, 1, CV_32FC1); //pixels用于保存所有的灰度像素

    for (int i = 0; i < rows;++i)
    {
        const uchar *idata = image.ptr<uchar>(i);
        float *pdata = pixels.ptr<float>(0);
        for (int j = 0; j < cols;++j)
        {
            pdata[i*cols + j] = idata[j];
        }
    }
    progress->setValue(36);
/*..............................参数说明...................................................................*
* .............................样本点（传入）...............................................................*
* .............................类别数（传入）...............................................................*
* .............................标签........................................................................*
* .............................TermCriteria代收敛准则（MAX_ITER最大迭代次数，EPS最高精度）......................*
* .............................尝试的次数（传入）............................................................*
* .............................聚类中心的选取方式............................................................*
* KMEANS_RANDOM_CENTERS 随机选取，KMEANS_PP_CENTERS使用Arthur提供的算法,KMEANS_USE_INITIAL_LABELS使用初始标签..*/

     kmeans(pixels, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 0), t, KMEANS_PP_CENTERS);

     progress->setValue(85);
     //标记像素点的类别，颜色区分
     for (int i = 0; i < rows;++i)
     {
         for (int j = 0; j < cols;++j)
         {
             Count[labels.at<int>(i*cols+j)]++;
             circle(clusteredMat, Point(j,i), 1, colorTab[labels.at<int>(i*cols + j)]);        //标记像素点的类别，颜色区分
         }
     }
     return clusteredMat;
    progress->setValue(99);
}

IplImage* ResizeImage(IplImage *src)        //重新调整图片的大小
{
    IplImage* dsc = cvCreateImage(cvSize(src->width*scale, src->height*scale),
        src->depth, src->nChannels);
    cvResize(src, dsc, CV_INTER_LINEAR);    //使用默认方法，双线性插值
    return dsc;

}

//k-means分类
void MainWindow::on_actionK_means_triggered()
{
    QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
    QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
    QStandardItem* currentItem = model->itemFromIndex(currentIndex);

    if(!currentIndex.isValid())
    {
        QMessageBox::warning(this, "", "错误代码 0004", QMessageBox::Ok);
    }
    //k_means批量处理
    //在批量处理的环节中，因为不会在屏幕上显示分类处理的信息，所以采用写文件的方式
    else if(currentItem->data(ROLE_MARK)!=currentItem->data(ROLE_MARK_FOLDER))
    {
//============弹出输入框并获取返回值==========
        bool ok;
        int k;
        int value = QInputDialog::getInt(this,tr("K值"), //设置标题
        tr("请输入类别数 [0~10]"),100,0,10,1,&ok);
        if(ok)
        {
            infoLabel->setVisible(false);
            sizeLabel->setText("");
            progress->setVisible(true);
            progress->setValue(0);
            k=value;                                    //获取返回值
        }
        else
        {
            return;
        }

//===================================
        //查找当前 选择的图片节点的路径，QString型的
        int count=currentItem->rowCount();
        QString str, info;
        int Count[10];
        Mat org, colorResults;

        QString filePath = QFileDialog::getExistingDirectory(this, tr("选择保存位置"), "C:/Users/Public/Pictures");

        if(filePath.isEmpty()){
            QMessageBox::information(this, "", "错误代码 0005", QMessageBox::Ok);
        }
        else{
            for(int i=0;i<count;i++)
            {
                QString sttr = "共有 " + QString::number(count) +
                        " 张图片，当前正在处理 " + QString::number(i+1) +
                        "/" + QString::number(count) + "  ……";
                sizeLabel->setText(sttr);
                qApp->processEvents();

               int j=file.match(currentItem->child(i,0)->index().data().toString());
               str=file.filenameList[j];

               //读入图片并分类
               Count[10]={0};
               org=imread(str.toStdString());
               progress->setValue(12);

               info="";
               if(!org.empty())
               {
                  colorResults=K_means(org,k,5,Count);

                  for(int i=0;i<k;i++)
                  {
                      info+="第"+QString::number(i+1)+"类像素点个数:  "+QString::number(Count[i])+"\n";
                  }

                  progress->setValue(100);
                   if(!colorResults.empty())
                   {
                       //弹出窗口现实分类结果信息
                        //QMessageBox::about(this,file.filename[file.match(currentItem->child(i,0)->index().data().toString())] + tr("分类结果信息"),info);
                        //waitKey(0);
                        //设置鼠标随意拖动窗口改变大小并显示结果
                        str = file.filename[file.match(currentItem->child(i,0)->index().data().toString())];
                        str = filePath + "/km_" + str;
                        if(imwrite(str.toStdString(), colorResults))
                        {
                            //QMessageBox::information(NULL, NULL, NULL, NULL);
                        }
                        //此处初始化写文件需要的函数
                        str = str + ".txt";
                        QFile _f(str);

                        if(!_f.open(QIODevice::ReadWrite | QIODevice::Text))
                        {
                           QMessageBox::warning(this,"","错误代码 0006",QMessageBox::Ok);

                        }
                        QTextStream _in(&_f);

                        //此处开始写分类信息到文件
                        _in<<info;
                        _f.close();
                   }
               }
               progress->setValue(0);
               if(i == count - 1){
                   progress->setValue(100);
                   progress->setVisible(false);
                   infoLabel->setVisible(true);
                   sizeLabel->setText("处理完成");
               }
            }
            QMessageBox::information(this,NULL,"分类结束",QMessageBox::Ok);
        }
    }
    //k_means单张图片处理
    else
    {
//============弹出输入框并获取返回值==========
        bool ok;
        int k;
        int value = QInputDialog::getInt(this,tr("K值"), //设置标题
        tr("请输入类别数 [0~10]"),100,0,10,1,&ok);
        if(ok)
        {
            infoLabel->setVisible(false);
            sizeLabel->setText("共有 1 张图片，当前正在处理 1/1 ……");
            progress->setVisible(true);
            progress->setValue(0);
            k=value;  //获取返回值
        }
        else
        {
            return;
        }

        int j=file.match(ui->dockWidget->treeview->currentIndex().data().toString());
        QString str=file.filenameList[j];

        //读入图片并分类
        int Count[10]={0};
        Mat org=imread(str.toStdString());
        progress->setValue(12);

        QString info="";
        if(!org.empty())
        {
           Mat colorResults=K_means(org,k,5,Count);

           for(int i=0;i<k;i++)
           {
               info+="第"+QString::number(i+1)+"类像素点个数:  "+QString::number(Count[i])+"\n";
           }

           progress->setValue(100);
           progress->setVisible(false);
           sizeLabel->setText("处理完成");
           progress->setValue(0);

            if(!colorResults.empty())
            {
                //弹出窗口现实分类结果信息
                 QMessageBox::about(this,file.filename[file.match(ui->dockWidget->treeview->currentIndex().data().toString())] + tr("分类结果信息"),info);

                 //设置鼠标随意拖动窗口改变大小并显示结果
                 QString str = file.filename[file.match(ui->dockWidget->treeview->currentIndex().data().toString())] + " @ K-means";


                 namedWindow(str.toStdString(), CV_WINDOW_FREERATIO);
                 imshow(str.toStdString(),colorResults);
            }
        }
    }
}

void MainWindow::on_actionKNN_triggered()
{
    Global::drag_position = 2;
    Global::cf_property=1;//控制分类属性，选择knn分类时属性值为1,选择minddis的属性值为2，选择SVM的属性值为3
    ui->actionChoose_position->setVisible(true);
    ui->actionChoose_position->setVisible(true);
    ui->actionChoose_position->setChecked(true);
    ui->actionCursor->setChecked(false);
    ui->actionHands->setChecked(false);

    ttar.clear();
    //此处初始化分类样本选择器
    k->show();
}

//ISODATA按钮事件函数实现
void MainWindow::on_actionISODATA_triggered()
{
        dlg->show();                        //弹出输入参数的dialog
}

//接受数据并分类
void MainWindow::receiveData(QString data1, QString data2, QString data3, QString data4, QString data5, QString data6)
{
    s1=data1;
    s2=data2;
    s3=data3;
    s4=data4;
    s5=data5;
    s6=data6;
    int newexpClusters=s1.toInt();
    int newthetaN=s2.toInt();
    int newmaxIts=s3.toInt();
    int newcombL=s4.toInt();
    double newthetaS=s5.toDouble();
    double newthetaC=s6.toDouble();

    QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
    QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
    QStandardItem* currentItem = model->itemFromIndex(currentIndex);

    if(!currentIndex.isValid())
    {
        QMessageBox::warning(this, "", "错误代码 0004", QMessageBox::Ok);
    }
    //IOSDATA批量分类处理
    else if(currentItem->data(ROLE_MARK)!=currentItem->data(ROLE_MARK_FOLDER))
    {


        //查找当前 选择的图片节点的路径，QString型的
        int count=currentItem->rowCount();
        QString filePath = QFileDialog::getExistingDirectory(this, tr("选择保存位置"), "C:/Users/Public/Pictures");
        if(filePath.isEmpty())
        {
            QMessageBox::warning(this, "", "错误代码 0005", QMessageBox::Ok);
        }
        else
        {
            for(int i=0;i<count;i++)
            {
                //显示进度条并初始化
                infoLabel->setVisible(false);
                progress->setVisible(true);
                progress->setValue(0);
                QString sttr = "共有 " + QString::number(count) +
                        " 张图片，当前正在处理 " + QString::number(i+1) +
                        "/" + QString::number(count) + "  ……";
                sizeLabel->setText(sttr);
                qApp->processEvents();

                int j=file.match(currentItem->child(i,0)->index().data().toString());
                QString str=file.filenameList[j];
                //读入图片并分类
                Mat image=imread(str.toStdString());
                progress->setValue(24);
                ISOData(2, image, newexpClusters, newthetaN, newmaxIts, newcombL, newthetaS, newthetaC, currentItem->child(i,0)->index().data().toString(), filePath);
            }
            QMessageBox::information(this, "", "分类完成", QMessageBox::Ok);
        }
    }
    //ISODATA单张图片分类处理
    else
    {
        //显示进度条并初始化
        infoLabel->setVisible(false);
        sizeLabel->setText("共有 1 张图片，当前正在处理 1/1 ……");
        progress->setVisible(true);
        progress->setValue(12);

        int j=file.match(ui->dockWidget->treeview->currentIndex().data().toString());
        QString str=file.filenameList[j];

        //读入图片并分类
        Mat image;
        image=imread(str.toStdString());
        progress->setValue(24);//设置进度条进度
        if(!image.empty())
        {
            ISOData(1, image, newexpClusters, newthetaN, newmaxIts, newcombL, newthetaS, newthetaC, ui->dockWidget->treeview->currentIndex().data().toString(), "");

        }
        else
        {
            QMessageBox::warning(this, "", "错误代码 0004", QMessageBox::Ok);
        }
    }
}

//knn分类函数实现
Mat MainWindow::KNN(string root,Mat trainingDataMat,Mat labelsMat,vector<int> &info )
{
    float data[5][3]={{0, 0, 255},{0, 255, 255},{0, 255, 0},{255, 0, 0},{0, 0, 0}};
    Scalar colorTab[] =     //10个颜色(染色),即最多可分10个类别
    {
     Scalar(255,255,255),
     Scalar(0, 0, 255),
     Scalar(0, 255, 255),
     Scalar(0, 255, 0),
     Scalar(255, 0, 0),
     Scalar(0, 0, 0),
     Scalar(255, 0, 0),
     Scalar(255, 255, 0),
     Scalar(255, 0, 100),
     Scalar(100, 100, 100),
     Scalar(50, 125, 125),
  };
    vector<int>labels;
    vector<int> countdata(10);
    Mat image1=imread(root.data());
    Mat image,trainingDataMat1;
    cvtColor(image1,image,CV_BGR2GRAY);
    //cvtColor(trainingDataMat,trainingDataMat1,CV_BGR2GRAY);
    //imshow("image",image1);
    //cout<<trainingDataMat1<<endl<<endl;
    // 训练KNN——model
    CvKNearest knn;
    knn.train(trainingDataMat,labelsMat,Mat(),false,3);
    progress->setValue(80);
    //对图像进行预测分类
    Mat resultimg(image.rows,image.cols,CV_32FC3);
    Vec3b green (0,255,0),red (0,0,255),blue (255,0,0),black(0,0,0),white(255,255,255),five(0,139,139),six(47,79,79),seven(0,128,128),eight(64,224,208),nine(0,255,0),ten(107,142,35);
    for (int i = 0; i <image.rows;i++)
        {
            //const uchar *idata = image.ptr<uchar>(i);
            //float *pdata = sampleMat.ptr<float>(0);
            for (int j = 0; j <image.cols;j++)
            {
                Mat sampleMat=(Mat_<float>(1,1)<<image.at<uchar>(i,j));
                //pdata[0] = idata[j];
                //cout<<sampleMat.at<float>(0, 0)<<endl<<endl;
                if(sampleMat.at<float>(0, 0)==255)
                {
                    resultimg.at<Vec3f>(i,j)=white;
                }
                else
                {
                    float response = knn.find_nearest(sampleMat,3); ;
                    if(response==1)
                    {
                        resultimg.at<Vec3f>(i, j)=blue;
                        countdata[0]++;
                    }
                    else if(response==2)
                    {
                        resultimg.at<Vec3f>(i, j)=red;
                        countdata[1]++;
                    }
                    else if(response==3)
                    {
                         resultimg.at<Vec3f>(i, j)=green;
                         countdata[2]++;
                    }
                    else if(response==4)
                    {
                        resultimg.at<Vec3f>(i,j)=black;
                        countdata[3]++;
                    }
                    else if(response==5)
                    {
                       resultimg.at<Vec3f>(i,j)=five;
                       countdata[4]++;
                    }
                    else if(response==6)
                    {
                        resultimg.at<Vec3f>(i,j)=six;
                        countdata[5]++;
                    }
                    else if(response==7)
                    {
                        resultimg.at<Vec3f>(i,j)=seven;
                        countdata[6]++;
                    }
                    else if(response==8)
                    {
                        resultimg.at<Vec3f>(i,j)=eight;
                        countdata[7]++;
                    }
                    else if(response==9)
                    {
                        resultimg.at<Vec3f>(i,j)=nine;
                        countdata[8]++;
                    }
                    else if(response==10)
                    {
                        resultimg.at<Vec3f>(i,j)=ten;
                        countdata[9]++;
                    }
                }
           }
        }
    info=countdata;
    progress->setValue(95);
    return resultimg;
}


void MainWindow::choose_position_changed(){
    Global::drag_position == 2;
    ui->actionChoose_position->setChecked(true);
    ui->actionCursor->setChecked(false);
    ui->actionHands->setChecked(false);
}

void MainWindow::choose_position_nonchanged(){
    Global::drag_position = 1;
    Global::ispress = false;
    ui->actionChoose_position->setVisible(false);
    ui->actionCursor->setChecked(false);
    ui->actionHands->setChecked(true);
}

//获取每类样本的均值函数实现（最小距离分类）
float trainData(Mat trainingDataMat)
{
    float sum=0;
    for(int i=0;i<trainingDataMat.rows;i++)
        sum+=trainingDataMat.at<float>(i,0);
    float f=(sum*1.0/trainingDataMat.rows);
    return f;
}


//算两像素点距离（欧式距离） 最小距离分类
float Edist(float a,float b)
{
    float d=0.0;
    d=abs((a-b));
    return d;
}

//找最小距离类
float Find(Mat sampleMat,Mat trainingData,Mat labelMat)
{
    float samp=sampleMat.at<float>(0,0);
    float min,t;
    float ptr;
    min=ptr=0;
    ptr=trainingData.at<float>(0,0);
    min=Edist(samp,ptr);
    t=1;
    int i=1;
    float tmp;
    while(i<trainingData.rows)
    {
        ptr=trainingData.at<float>(i,0);
        tmp=Edist(samp,ptr);
        if(min>tmp)
        {
            min=tmp;
            t=i+1;
        }
        i++;
    }
    return t;
}

//最小距离分类方法
Mat MainWindow::nearest(string root,Mat trainingData,Mat labelMat,vector<int> &info)
{
    Mat image;
    Mat image1=imread(root.data());
    vector<int> countdata(10);
    cvtColor(image1,image,CV_BGR2GRAY);
    float a[4]={0};
    //Mat trainingData;
   // Mat labelMat;

   // GetTraindata(trainingData,labelMat);    //此处的tranningData labelMat直接通过数组获取每一类的均值和标签

    //对图像进行预测分类
    Mat resultimg(image.rows,image.cols,CV_32FC3);
    progress->setValue(48);
    Vec3b green (0,255,0),red (0,0,255),blue (255,0,0),black(0,0,0),white(255,255,255),five(0,139,139),six(47,79,79),seven(0,128,128),eight(64,224,208),nine(0,255,0),ten(107,142,35);

    for (int i = 0; i <image.rows;i++)
    {
        //const uchar *idata = image.ptr<uchar>(i);
        //float *pdata = sampleMat.ptr<float>(0);
        for (int j = 0; j <image.cols;j++)
        {
            Mat sampleMat=(Mat_<float>(1,1)<<image.at<uchar>(i,j));
            //pdata[0] = idata[j];
            //cout<<sampleMat.at<float>(0, 0)<<endl<<endl;
            if(sampleMat.at<float>(0, 0)==255)
            {
                resultimg.at<Vec3f>(i,j)=white;
            }
            else
            {
                float response =Find(sampleMat,trainingData,labelMat);

                if(response==1)
                {
                     resultimg.at<Vec3f>(i, j)=blue;
                     countdata[0]++;
                }
                else if(response==2)
                {
                      resultimg.at<Vec3f>(i, j)=red;
                      countdata[1]++;
                }
                else if(response==3)
                {
                      resultimg.at<Vec3f>(i, j)=green;
                      countdata[2]++;
                }
                 else if(response==4)
                {
                      resultimg.at<Vec3f>(i,j)=black;
                      countdata[3]++;
                }
                else if(response==5)
                {
                      resultimg.at<Vec3f>(i,j)=five;
                      countdata[4]++;
                }
                else if(response==6)
                {
                      resultimg.at<Vec3f>(i,j)=six;
                      countdata[5]++;
                }
                else if(response==7)
                {
                      resultimg.at<Vec3f>(i,j)=seven;
                      countdata[6]++;
                }
                else if(response==8)
                {
                      resultimg.at<Vec3f>(i,j)=eight;
                      countdata[7]++;
                }
                 else if(response==9)
                {
                      resultimg.at<Vec3f>(i,j)=nine;
                      countdata[8]++;
                }
                else if(response==10)
                {
                      resultimg.at<Vec3f>(i,j)=ten;
                      countdata[9]++;
                }
            }
       }
    }
    info=countdata;
    progress->setValue(95);
    return resultimg;

}


//最小距离分类方法无返回处理结果Mat
Mat MainWindow::nearest_return(string root,Mat trainingData,Mat labelMat)
{
    Mat image;
    Mat image1=imread(root.data());
    cvtColor(image1,image,CV_BGR2GRAY);
    float a[4]={0};
    //Mat trainingData;
   // Mat labelMat;

   // GetTraindata(trainingData,labelMat);    //此处的tranningData labelMat直接通过数组获取每一类的均值和标签

    //对图像进行预测分类
    Mat resultimg(image.rows,image.cols,CV_32FC3);
    progress->setValue(80);
    Vec3b green (0,255,0),red (0,0,255),blue (0,255,255),m(255,255,0),white(255,255,255);
    for (int i = 0; i <image.rows;i++)
    {
        //const uchar *idata = image.ptr<uchar>(i);
        //float *pdata = sampleMat.ptr<float>(0);
        for (int j = 0; j <image.cols;j++)
        {
            Mat sampleMat=(Mat_<float>(1,1)<<image.at<uchar>(i,j));
            //pdata[0] = idata[j];
            //cout<<sampleMat.at<float>(0, 0)<<endl<<endl;
            if(sampleMat.at<float>(0, 0)==255)
            {
                resultimg.at<Vec3f>(i,j)=white;
            }
            else
            {
                float response =Find(sampleMat,trainingData,labelMat);

                if(response==1)
                {
                    resultimg.at<Vec3f>(i, j)=blue;
                     a[0]++;

                }
                else if(response==2)
                {
                    resultimg.at<Vec3f>(i, j)=red;
                    a[1]++;
                }
                else if(response==3)
                {
                  resultimg.at<Vec3f>(i, j)=green;
                   a[2]++;
                }
                else if(response==4)
                {
                   resultimg.at<Vec3f>(i,j)=m;
                   a[3]++;
                }
            }
       }
    }
    progress->setValue(95);
    return resultimg;
}

//svm分类方法无返回值
Mat MainWindow::Svmclass(string root,Mat trainingDataMat,Mat labelsMat,vector<int> &info)
{
    vector<int> Countnum(4);
    Mat image1=imread(root.data());
    Mat image,trainingDataMat1;
    cvtColor(image1,image,CV_BGR2GRAY);
    //cvtColor(trainingDataMat,trainingDataMat1,CV_BGR2GRAY);
    //imshow("image",image1);
    //cout<<trainingDataMat1<<endl<<endl;
    //设置 SVM 参数
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC ;
    params.kernel_type = CvSVM::RBF;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // 训练model
    CvSVM SVM;
    SVM.train(trainingDataMat,labelsMat, Mat(), Mat(), params);
    //对图像进行预测分类
    Mat resultimg(image.rows,image.cols,CV_32FC3);
    progress->setValue(60);
    Vec3b green (0,255,0),red (0,0,255),blue (255,0,0),yellow(0,255,255),white(255,255,255);
    for (int i = 0; i <image.rows;i++)
    {
        //const uchar *idata = image.ptr<uchar>(i);
        //float *pdata = sampleMat.ptr<float>(0);
        for (int j = 0; j <image.cols;j++)
        {
            Mat sampleMat=(Mat_<float>(1,1)<<image.at<uchar>(i,j));
            //pdata[0] = idata[j];
            //cout<<sampleMat.at<float>(0, 0)<<endl<<endl;
            if(sampleMat.at<float>(0, 0)==255)
            {
                resultimg.at<Vec3f>(i,j)=white;
            }
            else
            {
                float response = SVM.predict(sampleMat);

                if(response==1)
                {
                    resultimg.at<Vec3f>(i, j)=red;
                    Countnum[(int)response-1]++;
                }
                else if(response==2)
                {
                    resultimg.at<Vec3f>(i, j)=yellow;
                    Countnum[(int)response-1]++;
                }
                else if(response==3)
                {
                    resultimg.at<Vec3f>(i, j)=green;
                    Countnum[(int)response-1]++;
                }
                else if(response==4)
                {
                    resultimg.at<Vec3f>(i,j)=blue;
                    Countnum[(int)response-1]++;
                }

            }
       }
    }
    info=Countnum;
    cout<<info[1]<<endl;
    progress->setValue(95);
    return resultimg;
}

//矩阵相乘

void selfmul(Mat as, Mat ab, Mat &result)
{
    Mat result1(as.rows,ab.cols,CV_8UC1);
    //cout<<as.at<uchar>(0,0)<<endl;
    double temp=0;
    for(int i=0; i<as.rows; i++)
        for(int j=0; j<ab.cols; j++)
        {
            temp=0;
            for(int k=0; k<=as.cols; k++)
            {
                temp=temp+as.at<uchar>(i,k)*ab.at<uchar>(k,j);

            }
            result1.at<uchar>(i,j)=temp;
            //cout<<temp<<endl;
            //cout<<result1.at<float>(i,j)<<"  ";
        }
    result1.copyTo(result);
    //cout<<result1<<endl;
}

//乘
int slmul(Mat as, Mat ab)
{
    int temp=0;
    for(int i=1; i<=as.cols; i++)
    {
        //cout<<as.at<uchar>(i)<<endl;
        temp=temp+as.at<uchar>(i)*ab.at<uchar>(i);
    }

    return temp;
}
void  GetTraindata(int n,vector<covarry>&pre,QString src_img)
{
    /*....样本进行处理....*/
    //对数据进行处理
       Mat image=imread(src_img.toStdString());      //以Mat格式读取图片
       //cvtColor(image,image,CV_BGR2GRAY);              //转化为灰度图
        for(int ii=0;ii<n;ii++)
        {
            covarry t;
            int count_row=0;
            for(int j=0;j<ttar.size();j++)
            {


                Data a=ttar[j];
                int info=a.info;
                if(info==ii+1)
                {
                    count_row++;
                }
            }
            int k=0;
            double data[count_row][3];
            for(int j=0;j<ttar.size();j++)
            {
                Data a=ttar[j];
                int info=a.info;
                if(info==ii+1)
                {
                     int x=a.point.x();
                     int y=a.point.y();
                     data[k][0]=image.at<Vec3b>(y,x)[0];
                     data[k][1]=image.at<Vec3b>(y,x)[1];
                     data[k][2]=image.at<Vec3b>(y,x)[2];
                     k++;
                 }
            }
            t.label=ii+1;
            Mat matData(count_row,3,CV_64FC1,data);
            for(int m=0;m<k;m++)
            {
                for(int n=0;n<3;n++)
                {
                    cout<<data[m][n]<<" ";
                }
            }
            cout<<endl;
            t.calculatep(matData);
            pre.push_back(t);
        }
}

//减
void sub(Mat sa,Mat sb,Mat &result)
{
    float sr[3]={0};
    float f;
    for(int i=0; i<3; i++)
    {
        sr[i]=sa.at<uchar>(0,0)-sb.at<uchar>(0,i);
    }
    Mat re(1,3,CV_8UC1,sr);
    //cout<<re<<endl;
    re.copyTo(result);
}

float predict(vector<covarry>pre, Mat sample)
{
    int maxlabel=0;
    double maxvalue=0;
    double value;
    Mat a;
    Mat matrix,submatrix;
    double b;
    sub(sample,pre.at(0).means,submatrix);
    selfmul(submatrix,pre.at(0).zmatrix,a);
    matrix=submatrix.t();
    b=slmul(a,matrix);
    //cout<<a<<endl;
    //预测样本属于本类的概率
    //log(1.0)-0.5*log(pre.at(0).hcovar)
    value=log(1.0)-0.5*log(pre.at(0).hcovar)-0.5*b;
    maxvalue=value;
    maxlabel=pre.at(0).label;
    for(int i=1; i<pre.size();i++)
    {
        sub(sample,pre.at(i).means,submatrix);
        selfmul(submatrix,pre.at(i).zmatrix,a);
        matrix=submatrix.t();
        b=slmul(a,matrix);
        //预测样本属于本类的概率
        //log(1.0)-0.5*log(pre.at(i).hcovar)
        value=log(1.0)-0.5*log(pre.at(i).hcovar)-0.5*b;
        //更新最大概率值及对应标签
        if(value>maxvalue)
        {
            maxvalue=value;
            maxlabel=pre.at(i).label;
        }
    }
    //cout<<maxlabel<<"   ";
    return maxlabel;
}


//分类顶部
//接受knn分类窗口的确定按钮
void MainWindow::receivearray(int a)
{
    model_item_count=a;
    //获取当前选中treeview的项
    QStandardItemModel* model = static_cast<QStandardItemModel*>(ui->dockWidget->treeview->model());
    QModelIndex currentIndex = ui->dockWidget->treeview->currentIndex();
    QStandardItem* currentItem = model->itemFromIndex(currentIndex);
    if(!ttar.empty())
    {
        //knn分类=================================
        if(Global::cf_property==1) //选择了knn分类
        {

             //knn批量分类处理=======================================================
            if(currentItem->data(ROLE_MARK)!=currentItem->data(ROLE_MARK_FOLDER))
            {
                QString str=currentIndex.data().toString();
                QString filePath = QFileDialog::getExistingDirectory(this, tr("选择保存位置"), "C:/Users/Public/Pictures");
                if(filePath.isEmpty()){
                    QMessageBox::information(this, "", "错误代码 0005", QMessageBox::Ok);
                }
                else
                {
                    int count_image=currentItem->rowCount();
                    for(int i=0;i<count_image;i++)
                    {
                        //显示进度条并初始化
                        infoLabel->setVisible(false);
                        progress->setVisible(true);
                        progress->setValue(0);
                        QString sttr = "共有 " + QString::number(count_image) +
                                " 张图片，当前正在处理 " + QString::number(i+1) +
                                "/" + QString::number(count_image) + "  ……";
                        sizeLabel->setText(sttr);
                        qApp->processEvents();


                        //获取当前处理的图片名称
                        QString str=currentIndex.child(i,0).data().toString();
                        //QMessageBox::information(NULL,NULL,str,NULL);


                        //获取当前处理图片的路径
                        int j=file.match(str);
                        src_image=file.filenameList[j];   //str为当前选中图片的路径
                        Mat image=imread(src_image.toStdString());//以Mat格式读取图片
                        progress->setValue(12);
                        cvtColor(image,image,CV_BGR2GRAY);
                        progress->setValue(14);

                        vector<float> v;                            //临时存储训练样本像素值

                        //根据坐标获取图像的像素值
                        for(int ii=0;ii<ttar.size();ii++)
                        {
                            Data a=ttar[ii];
                            int x=a.point.x();
                            int y=a.point.y();
                            float temp=image.at<uchar>(y,x);
                            v.push_back(temp);
                        }
                         progress->setValue(20);

                        //将vector数组赋值到b中
                        float *b=new float[v.size()];
                        for(int ii=0;ii<v.size();ii++)
                        {
                            b[ii]=v[ii];
                        }
                        //得到Mat格式的训练样本
                        Mat trainingDataMat(v.size(),1,CV_32FC1,b);
                        progress->setValue(30);

                        //获取训练样本标签
                        vector<float>label;
                        for(int ii=0;ii<ttar.size();ii++)//获取训练样本标签并其push到vector(label)
                        {
                            Data data;
                            data=ttar[ii];
                            float info=data.info;
                            label.push_back(info);
                        }

                        //将vector数组赋值给label1数组中
                        float label1[label.size()];
                        for(int ii=0;ii<label.size();ii++)
                        {
                            label1[ii]=label[ii];
                        }

                        //Mat类型的训练样本标签
                        Mat labelsMat(label.size(),1,CV_32FC1,label1);
                        progress->setValue(40);
                        //调用KNN分类方法
                        //我在这里
                        vector<int> info_class;
                        QString info;
                        Mat colorResults=KNN(src_image.toStdString(),trainingDataMat,labelsMat,info_class);
                        for(int ii=0;ii<model_item_count;ii++)
                        {
                            info+="第"+QString::number(ii+1)+"类像素点个数:  "+QString::number(info_class[i])+"\n";
                        }
                        QString strr = file.filename[file.match(currentItem->child(i,0)->index().data().toString())];
                        strr = filePath + "/knn_" + str;
                        //导出处理结果图片
                        imwrite(strr.toStdString(), colorResults);


                        //此处初始化写文件需要的函数
                        strr = strr + ".txt";
                         QFile _f(strr);

                        if(!_f.open(QIODevice::ReadWrite | QIODevice::Text))
                        {
                               QMessageBox::warning(this,"","错误代码 0006",QMessageBox::Ok);

                        }
                        QTextStream _in(&_f);

                        _in<<info;
                         _f.close();

                    }
                    ttar.clear();
                    emit refresh();
                    progress->setValue(100);
                    sizeLabel->setText("处理完成");
                    progress->setVisible(false);
                    progress->setValue(0);
                    infoLabel->setVisible(true);
                    QMessageBox::information(NULL,NULL,"分类完成",NULL);
                }

            }
            //knn单张图片分类处理============================================
            else
            {
                //添加进度条并初始化
                infoLabel->setVisible(false);
                sizeLabel->setText("共有 1 张图片，当前正在处理 1/1 ……");
                progress->setVisible(true);
                progress->setValue(0);
                QString str=currentIndex.data().toString();

                //获取当前选中图片的路径
                int j=file.match(str);
                src_image=file.filenameList[j];                 //str为当前选中图片的路径
                Mat image=imread(src_image.toStdString());      //以Mat格式读取图片
                progress->setValue(12);
                cvtColor(image,image,CV_BGR2GRAY);
                progress->setValue(14);

                vector<float> v;                            //临时存储训练样本像素值

                //根据坐标获取图像的像素值
                for(int i=0;i<ttar.size();i++)
                {
                    Data a=ttar[i];
                    int x=a.point.x();
                    int y=a.point.y();
                    float temp=image.at<uchar>(y,x);
                    v.push_back(temp);
                }
                progress->setValue(20);

                //将vector数组赋值到b中
                float *b=new float[v.size()];
                for(int i=0;i<v.size();i++)
                {
                    b[i]=v[i];
                }

                //得到Mat格式的训练样本
                Mat trainingDataMat(v.size(),1,CV_32FC1,b);
                progress->setValue(30);
                //获取训练样本标签
                vector<float>label;
                for(int i=0;i<ttar.size();i++)              //获取训练样本标签并其push到vector(label)
                {
                    Data data;
                    data=ttar[i];
                    float info=data.info;
                    label.push_back(info);
                }
                //将vector数组赋值给label1数组中
                float label1[label.size()];
                for(int i=0;i<label.size();i++)
                {
                    label1[i]=label[i];
                }
                //Mat类型的训练样本标签
                Mat labelsMat(label.size(),1,CV_32FC1,label1);
                //调用KNN分类方法
                //ttar.clear();
                emit refresh();
                progress->setValue(40);
                vector<int> info;
                Mat colorResults=KNN(src_image.toStdString(),trainingDataMat,labelsMat,info);
                QString info_class;
                for(int i=0;i<model_item_count;i++)
                {
                     info_class+="第"+QString::number(i+1)+"类像素点个数:  "+QString::number(info[i])+"\n";
                }

                emit refresh();
                sizeLabel->setText("处理完成");
                progress->setVisible(false);
                progress->setValue(0);
                infoLabel->setVisible(true);
                //弹出分类信息
                QMessageBox::about(this,file.filename[file.match(ui->dockWidget->treeview->currentIndex().data().toString())] + tr("分类结果信息"),info_class);

                ttar.clear();
                namedWindow(str.toStdString(), CV_WINDOW_FREERATIO);
                imshow(str.toStdString(),colorResults);
            }
        }
        //mindis分类============================================
        if(Global::cf_property==2) //选择了mindis分类
        {
            //mindis批量分类处理================================
            if(currentItem->data(ROLE_MARK)!=currentItem->data(ROLE_MARK_FOLDER))
            {
                QString filePath = QFileDialog::getExistingDirectory(this, tr("选择保存位置"), "C:/Users/Public/Pictures");
                if(filePath.isEmpty())
                {
                    QMessageBox::information(this, "", "错误代码 0005", QMessageBox::Ok);
                }
                else
                {
                    int count_image=currentItem->rowCount();
                    for(int p=0;p<count_image;p++)
                    {
                        //显示进度条并初始化
                        infoLabel->setVisible(false);
                        progress->setVisible(true);
                        progress->setValue(0);
                        QString sttr = "共有 " + QString::number(count_image) +
                                 " 张图片，当前正在处理 " + QString::number(p+1) +
                                        "/" + QString::number(count_image) + "  ……";
                        sizeLabel->setText(sttr);
                        qApp->processEvents();

                        //获取当前处理的图片名称
                        QString str=currentIndex.child(p,0).data().toString();
                        //获取当前处理图片的路径
                        int j=file.match(str);
                        src_image=file.filenameList[j];   //str为当前选中图片的路径
                        Mat image=imread(src_image.toStdString());//以Mat格式读取图片
                        progress->setValue(12);
                        cvtColor(image,image,CV_BGR2GRAY);
                        progress->setValue(24);
                        vector<float> v;    //存储每类样本均值
                        for(int i=0;i<model_item_count;i++)
                        {
                            vector<float> temp_i;//临时存储每一类的样本
                            for(int j=0;j<ttar.size();j++)
                            {

                                Data a=ttar[j];
                                int info=a.info;
                                if(info==i+1)
                                {
                                    int x=a.point.x();
                                    int y=a.point.y();
                                    float temp=image.at<uchar>(y,x);//根据坐标获取对应像素点的值
                                    temp_i.push_back(temp);
                                }
                            }
                            float temp_i_tomat[temp_i.size()];   //将temp_i数组的值赋给数组temp_i_tomat
                            for(int ii=0;ii<temp_i.size();ii++)
                            {
                                temp_i_tomat[ii]=temp_i[ii];
                            }

                            Mat trainingDataMat_i(temp_i.size(),1,CV_32FC1,temp_i_tomat);
                            float average_i=trainData(trainingDataMat_i);
                            v.push_back(average_i);
                        }
                        progress->setValue(36);
                        //QMessageBox::information(NULL,NULL,QString::number(v.size()),NULL);
                        float trainningData_tomat[v.size()];  //将v数组的值赋给数组trainningData_tomat
                        for(int i=0;i<v.size();i++)
                        {
                            trainningData_tomat[i]=v[i];
                        }
                        Mat  trainningData(v.size(),1,CV_32FC1,trainningData_tomat);//存储每类样本的均值
                        progress->setValue(48);
                        float label_mindis[v.size()];
                        for(int i=0;i<v.size();i++)
                        {
                            label_mindis[i]=i+1;
                        }
                        Mat trainningLabel(v.size(),1,CV_32FC1,label_mindis); //存储每类样本的标签
                        progress->setValue(60);
                        //调用mindis分类方法返回处理Mat
                        //回到这里
                        vector<int> info_class;
                        QString info;
                        Mat colorResult=nearest(src_image.toStdString(),trainningData,trainningLabel,info_class);
                        for(int i=0;i<model_item_count;i++)
                        {
                             info+="第"+QString::number(i+1)+"类像素点个数:  "+QString::number(info_class[i])+"\n";
                        }
                        QString strr = file.filename[file.match(currentItem->child(p,0)->index().data().toString())];
                        strr = filePath + "/mindis_" + str;
                        imwrite(strr.toStdString(), colorResult);


                        //此处初始化写文件需要的函数
                        strr = strr + ".txt";
                         QFile _f(strr);

                        if(!_f.open(QIODevice::ReadWrite | QIODevice::Text))
                        {
                               QMessageBox::warning(this,"","错误代码 0006",QMessageBox::Ok);

                        }
                        QTextStream _in(&_f);

                        _in<<info;
                         _f.close();
                        v.clear();
                    }
                    ttar.clear();
                    progress->setValue(100);
                    sizeLabel->setText("处理完成");
                    progress->setVisible(false);
                    progress->setValue(0);
                    infoLabel->setVisible(true);
                    emit refresh();
                    QMessageBox::information(NULL,NULL,"分类完成",NULL);
                }
              }
            //mindis单张图片分类处理
            else
            {
                //添加进度条并初始化
                infoLabel->setVisible(false);
                sizeLabel->setText("共有 1 张图片，当前正在处理 1/1 ……");
                progress->setVisible(true);
                progress->setValue(0);

                //QMessageBox::information(NULL,NULL,"这里是mindis分类",NULL);
                QString str=currentIndex.data().toString();
                //获取当前选中图片的路径
                int j=file.match(str);
                src_image=file.filenameList[j];                 //str为当前选中图片的路径

                Mat image=imread(src_image.toStdString());      //以Mat格式读取图片

                cvtColor(image,image,CV_BGR2GRAY);              //转化为灰度图
                progress->setValue(12);
                vector<float> v;    //存储每类样本均值
                for(int i=0;i<model_item_count;i++)
                {
                    vector<float> temp_i;//临时存储每一类的样本
                    for(int j=0;j<ttar.size();j++)
                    {

                        Data a=ttar[j];
                        int info=a.info;
                        if(info==i+1)
                        {
                            int x=a.point.x();
                            int y=a.point.y();
                            float temp=image.at<uchar>(y,x);//根据坐标获取对应像素点的值
                            temp_i.push_back(temp);
                        }
                    }
                    float temp_i_tomat[temp_i.size()];   //将temp_i数组的值赋给数组temp_i_tomat
                    for(int ii=0;ii<temp_i.size();ii++)
                    {
                        temp_i_tomat[ii]=temp_i[ii];
                    }
                    Mat trainingDataMat_i(temp_i.size(),1,CV_32FC1,temp_i_tomat);
                    float average_i=trainData(trainingDataMat_i);
                    progress->setValue(24);
                    v.push_back(average_i);
                }
                float trainningData_tomat[v.size()];  //将v数组的值赋给数组trainningData_tomat
                for(int i=0;i<v.size();i++)
                {
                    trainningData_tomat[i]=v[i];
                }
                Mat  trainningData(v.size(),1,CV_32FC1,trainningData_tomat);//存储每类样本的均值
                float label_mindis[v.size()];
                for(int i=0;i<v.size();i++)
                {
                    label_mindis[i]=i+1;
                }
                Mat trainningLabel(v.size(),1,CV_32FC1,label_mindis); //存储每类样本的标签
                progress->setValue(36);
                QString info_class;
                vector<int> info;
                Mat colorResult=nearest(src_image.toStdString(),trainningData,trainningLabel,info);
                for(int i=0;i<model_item_count;i++)
                {
                     info_class+="第"+QString::number(i+1)+"类像素点个数:  "+QString::number(info[i])+"\n";
                }
                emit refresh();
                sizeLabel->setText("处理完成");
                progress->setVisible(false);
                progress->setValue(0);
                infoLabel->setVisible(true);
                //弹出分类信息
                QMessageBox::about(this,file.filename[file.match(ui->dockWidget->treeview->currentIndex().data().toString())] + tr("分类结果信息"),info_class);

                ttar.clear();
                namedWindow(str.toStdString(), CV_WINDOW_FREERATIO);
                imshow(str.toStdString(),colorResult);
            }
        }
        if(Global::cf_property==3)
        {
            //svm批量分类处理================================
            if(currentItem->data(ROLE_MARK)!=currentItem->data(ROLE_MARK_FOLDER))
            {
                QString str=currentIndex.data().toString();
                QString filePath = QFileDialog::getExistingDirectory(this, tr("选择保存位置"), "C:/Users/Public/Pictures");
                if(filePath.isEmpty()){
                    QMessageBox::information(this, "", "错误代码 0005", QMessageBox::Ok);
                }
                else
                {
                    int count_image=currentItem->rowCount();
                    for(int i=0;i<count_image;i++)
                    {
                        //显示进度条并初始化
                        infoLabel->setVisible(false);
                        progress->setVisible(true);
                        progress->setValue(0);
                        QString sttr = "共有 " + QString::number(count_image) +
                                " 张图片，当前正在处理 " + QString::number(i+1) +
                                "/" + QString::number(count_image) + "  ……";
                        sizeLabel->setText(sttr);
                        qApp->processEvents();


                        //获取当前处理的图片名称
                        QString str=currentIndex.child(i,0).data().toString();
                        //QMessageBox::information(NULL,NULL,str,NULL);


                        //获取当前处理图片的路径
                        int j=file.match(str);
                        src_image=file.filenameList[j];   //str为当前选中图片的路径
                        Mat image=imread(src_image.toStdString());//以Mat格式读取图片
                        progress->setValue(12);
                        cvtColor(image,image,CV_BGR2GRAY);
                        progress->setValue(14);

                        vector<float> v;                            //临时存储训练样本像素值

                        //根据坐标获取图像的像素值
                        for(int ii=0;ii<ttar.size();ii++)
                        {
                            Data a=ttar[ii];
                            int x=a.point.x();
                            int y=a.point.y();
                            float temp=image.at<uchar>(y,x);
                            v.push_back(temp);
                        }
                         progress->setValue(20);

                        //将vector数组赋值到b中
                        float *b=new float[v.size()];
                        for(int ii=0;ii<v.size();ii++)
                        {
                            b[ii]=v[ii];
                        }
                        //得到Mat格式的训练样本
                        Mat trainingDataMat(v.size(),1,CV_32FC1,b);
                        progress->setValue(30);

                        //获取训练样本标签
                        vector<float>label;
                        for(int ii=0;ii<ttar.size();ii++)//获取训练样本标签并其push到vector(label)
                        {
                            Data data;
                            data=ttar[ii];
                            float info=data.info;
                            label.push_back(info);
                        }

                        //将vector数组赋值给label1数组中
                        float label1[label.size()];
                        for(int ii=0;ii<label.size();ii++)
                        {
                            label1[ii]=label[ii];
                        }

                        //Mat类型的训练样本标签
                        Mat labelsMat(label.size(),1,CV_32FC1,label1);
                        progress->setValue(40);
                        //调用KNN分类方法
                        vector<int> info_class;
                        QString info;

                        Mat colorResults=Svmclass(src_image.toStdString(),trainingDataMat,labelsMat,info_class);
                        for(int i=0;i<model_item_count;i++)
                        {
                             info+="第"+QString::number(i+1)+"类像素点个数:  "+QString::number(info_class[i])+"\n";
                        }

                        QString strr = file.filename[file.match(currentItem->child(i,0)->index().data().toString())];
                        strr = filePath + "/SVM_" + str;
                        imwrite(strr.toStdString(), colorResults);


                        //此处初始化写文件需要的函数
                        strr = strr + ".txt";
                         QFile _f(strr);

                        if(!_f.open(QIODevice::ReadWrite | QIODevice::Text))
                        {
                               QMessageBox::warning(this,"","错误代码 0006",QMessageBox::Ok);

                        }
                        QTextStream _in(&_f);

                        _in<<info;
                         _f.close();

                        v.clear();
                    }
                    ttar.clear();
                    emit refresh();
                    progress->setValue(100);
                    sizeLabel->setText("处理完成");
                    progress->setVisible(false);
                    progress->setValue(0);
                    infoLabel->setVisible(true);
                    QMessageBox::information(NULL,NULL,"分类完成",NULL);
                }

            }
            //svm单张图分类处理
            else
            {
                //QMessageBox::information(NULL,NULL,"这里是SVM分类",NULL);
                //添加进度条并初始化
                infoLabel->setVisible(false);
                sizeLabel->setText("共有 1 张图片，当前正在处理 1/1 ……");
                progress->setVisible(true);
                progress->setValue(0);
                QString str=currentIndex.data().toString();

                //获取当前选中图片的路径
                int j=file.match(str);
                src_image=file.filenameList[j];                 //str为当前选中图片的路径
                Mat image=imread(src_image.toStdString());      //以Mat格式读取图片
                progress->setValue(12);
                cvtColor(image,image,CV_BGR2GRAY);
                progress->setValue(14);

                vector<float> v;                            //临时存储训练样本像素值

                //根据坐标获取图像的像素值
                for(int i=0;i<ttar.size();i++)
                {
                    Data a=ttar[i];
                    int x=a.point.x();
                    int y=a.point.y();
                    float temp=image.at<uchar>(y,x);
                    v.push_back(temp);
                }
                progress->setValue(20);

                //将vector数组赋值到b中
                float *b=new float[v.size()];
                for(int i=0;i<v.size();i++)
                {
                    b[i]=v[i];
                }
                //得到Mat格式的训练样本
                Mat trainingDataMat(v.size(),1,CV_32FC1,b);
                progress->setValue(30);
                //获取训练样本标签
                vector<float>label;
                for(int i=0;i<ttar.size();i++)              //获取训练样本标签并其push到vector(label)
                {
                    Data data;
                    data=ttar[i];
                    float info=data.info;
                    label.push_back(info);
                }

                //将vector数组赋值给label1数组中
                float label1[label.size()];
                for(int i=0;i<label.size();i++)
                {
                    label1[i]=label[i];
                }
                //Mat类型的训练样本标签
                Mat labelsMat(label.size(),1,CV_32FC1,label1);
                //调用svM分类方法
                //ttar.clear();
                //emit refresh();
                progress->setValue(40);
                vector<int> info;
                QString info_class;
                Mat colorResults= Svmclass(src_image.toStdString(),trainingDataMat,labelsMat,info);
                for(int i=0;i<model_item_count;i++)
                {
                    info_class+="第"+QString::number(i+1)+"类像素点个数:  "+QString::number(info[i])+"\n";
                }
                ttar.clear();
                emit refresh();
                progress->setValue(100);
                sizeLabel->setText("处理完成");
                progress->setVisible(false);
                progress->setValue(0);
                infoLabel->setVisible(true);
                QMessageBox::about(this,file.filename[file.match(ui->dockWidget->treeview->currentIndex().data().toString())] + tr("分类结果信息"),info_class);
                namedWindow(str.toStdString(), CV_WINDOW_FREERATIO);
                imshow(str.toStdString(),colorResults);
            }
        }
        //mlc55555
        /*
        if(Global::cf_property==4)
        {
            //MLC批量分类处理================================
            if(currentItem->data(ROLE_MARK)!=currentItem->data(ROLE_MARK_FOLDER))
            {
                QString str=currentIndex.data().toString();
                QString filePath = QFileDialog::getExistingDirectory(this, tr("选择保存位置"), "C:/Users/Public/Pictures");
                if(filePath.isEmpty()){
                    QMessageBox::information(this, "", "错误代码 0005", QMessageBox::Ok);
                }
                else
                {
                    QMessageBox::information(NULL,NULL,"这里是最大似然分类",NULL);
                }
            }
            //MLC单张图分类处理
            else
            {
                QString str=currentIndex.data().toString();
                int j=file.match(str);
                src_image=file.filenameList[j];
                Scalar colorTab[] =     //10个颜色(染色),即最多可分10个类别
                {
                     Scalar(255,255,255),
                     Scalar(0, 0, 255),
                     Scalar(0, 255, 255),
                     Scalar(0, 255, 0),
                     Scalar(255, 0, 0),
                     Scalar(0, 0, 0),
                     Scalar(255, 0, 0),
                     Scalar(255, 255, 0),
                     Scalar(255, 0, 100),
                     Scalar(100, 100, 100),
                     Scalar(50, 125, 125),
                 };
                 int Countnum[4]={0};
                 vector<covarry>pre;
                 vector<float>labels;
                 float label=0;
                 GetTraindata(model_item_count,pre,src_image);//把每一类的概率以及标签获得并压入pre中
                 for(int i=0; i<pre.size(); i++)
                 {
                     // cout<<pre.at(i).label<<endl;
                     cout<<pre.at(i).hcovar<<endl;  //输出协方差行列式的值
                 }
                 Mat image,image1;
                 Mat sample;
                 image=imread(src_image.toStdString());//读入图片
                 cvtColor(image,image1,CV_BGR2GRAY); //转化为灰度图
                 for(int i=0; i<image.rows; i++)
                 {
                     for(int j=0; j<image.cols; j++)
                     {
                         sample=(Mat_<Vec3b>(1,1)<<image.at<Vec3b>(i,j));
                         //cout<<sample<<endl;
                         if(image.at<Vec3b>(i,j)[0]==255&&image.at<Vec3b>(i,j)[1]==255&&image.at<Vec3b>(i,j)[2]==255)
                         {
                             label=-1;
                             labels.push_back(label);
                         }


                              label=predict(pre,sample);
                              Countnum[(int)label]++;
                              labels.push_back(label);
                     }
                 }
                Mat resultimage(image.rows,image.cols,CV_32FC3);
                for (int i = 0; i < image.rows;++i)
                 {
                     for (int j = 0; j < image1.cols;++j)
                     {
                         circle(resultimage, Point(j,i), 1, colorTab[(int)labels[i*image.cols + j]]);        //标记像素点的类别，颜色区分
                     }
                 }

                float Count,num,testlabel;
                Count=num=0;

                for(int i=0; i<4; i++)
                {
                    //Mat testdata;
                     //getData(testdata, root[i]);//获取第i类的样本
                     vector<float> temp_i;//临时存储每一类的样本
                     for(int j=0;j<ttar.size();j++)
                     {

                         Data a=ttar[j];
                         int info=a.info;
                         if(info==i+1)
                         {
                             int x=a.point.x();
                             int y=a.point.y();
                             float temp=image.at<uchar>(y,x);//根据坐标获取对应像素点的值
                             temp_i.push_back(temp);
                         }
                     }
                     float temp_i_tomat[temp_i.size()];   //将temp_i数组的值赋给数组temp_i_tomat
                     for(int ii=0;ii<temp_i.size();ii++)
                     {
                         temp_i_tomat[ii]=temp_i[ii];
                     }
                     Mat testdata(temp_i.size(),1,CV_32FC1,temp_i_tomat);
                     cout<<testdata.rows<<endl;
                     num=num+testdata.rows;
                     for(int j=0; j<testdata.rows; j++)
                     {
                         sample=(Mat_<Vec3b>(1,1)<<image.at<Vec3b>(i,j));
                         testlabel=predict(pre,sample);
                         if(testlabel==i)
                             Count++;
                     }
                }
                cout<<Count/num<<endl;     //精度
                for(int i=0; i<4; i++)
                {
                    cout<<Countnum[i]<<endl;
                }
                 imshow("result",resultimage);
                 waitKey(0);
            }
        }*/

     }
    else
    {
        QMessageBox::warning(NULL,"错误编号0005","所选样本为空",QMessageBox::Ok);
    }
    //emit refresh();
}

//点击mindis分类相应事件
void MainWindow::on_actionMindisf_triggered()
{
    Global::drag_position = 2;
    Global::cf_property=2; //控制分类属性，选择knn分类时属性值为1,选择minddis的属性值为2，选择SVM的属性值为3
    ui->actionChoose_position->setVisible(true);
    ui->actionChoose_position->setVisible(true);
    ui->actionChoose_position->setChecked(true);
    ui->actionCursor->setChecked(false);
    ui->actionHands->setChecked(false);
    ttar.clear();
    //此处初始化分类样本选择器
    k->show();
}

void MainWindow::on_actionSVM_triggered()
{
    Global::drag_position = 2;
    Global::cf_property=3;//控制分类属性，选择knn分类时属性值为1,选择minddis的属性值为2，选择SVM的属性值为3
    ui->actionChoose_position->setVisible(true);
    ui->actionChoose_position->setChecked(true);
    ui->actionCursor->setChecked(false);
    ui->actionHands->setChecked(false);
    ttar.clear();
    //此处初始化分类样本选择器
    k->show();
}

