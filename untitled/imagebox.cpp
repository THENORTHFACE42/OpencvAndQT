#include "imagebox.h"

using namespace std;

ImageBox::ImageBox(QWidget *parent) : QWidget(parent)
{
    menu = new QMenu();
    refresh = new QAction("刷新", this);
    label_16 = new QAction("设置网格 16*16", this);
    label_16->setCheckable(true);
    label_32 = new QAction("设置网格 32*32", this);
    label_32->setCheckable(true);
    label_non = new QAction("无网格", this);
    label_non->setCheckable(true);
    label_non->setChecked(true);

    connect(refresh, &QAction::triggered, this, &ImageBox::actionRefresh_tragged);
    connect(label_16, &QAction::triggered, this, &ImageBox::action_16_label);
    connect(label_32, &QAction::triggered, this, &ImageBox::action_32_label);
    connect(label_non, &QAction::triggered, this, &ImageBox::action_non_label);
}

ImageBox::~ImageBox()
{

}

bool ImageBox::isInPixmap(QPoint pos){
    QRect pixmapRect(CC_x, CC_y, CC_width, CC_height);

    return pixmapRect.contains(pos);
}

void ImageBox::contextMenuEvent(QContextMenuEvent *event){
    Q_UNUSED(event);

    menu->clear();
    menu->addAction(refresh);
    menu->addAction(label_non);
    menu->addAction(label_16);
    menu->addAction(label_32);

    menu->exec(QCursor::pos());
    menu->acceptDrops();
}

//===============此处设置菜单项================
void ImageBox::actionRefresh_tragged(){
    update();
}

void ImageBox::action_16_label(){
    Global::label = 16;
    label_16->setChecked(true);
    label_32->setChecked(false);
    label_non->setChecked(false);
    update();
}

void ImageBox::action_32_label(){
    Global::label = 32;
    label_16->setChecked(false);
    label_32->setChecked(true);
    label_non->setChecked(false);
    update();
}

void ImageBox::action_non_label(){
    Global::label = 0;
    label_16->setChecked(false);
    label_32->setChecked(false);
    label_non->setChecked(true);
    update();
}

void ImageBox::setPixmap(const QPixmap &pixmap){
    m_pixmap = pixmap;
    showSuitableSize();

    ttar.clear();
    cc = 0;

    SingleTonSToast::getInstance().setMessageVDuration(QString::number(m_percentage * 100) + " %", 0.25);
}

void ImageBox::ariseScale(int rate){
    double old_percentage = m_percentage;
    double step = static_cast<double>(rate)/100.0*5*old_percentage; //步进值
    m_percentage += step;
    if(m_percentage < 0.01)
    {
        m_percentage = 0.01;
    }
    else if(m_percentage > 1000)
    {
        m_percentage = 1000.0;
    }

    m_scale = m_percentage*m_scale/old_percentage;

    update();

    if(!m_pixmap.isNull()){
        QString str = QString::number(m_percentage * 100) + " %";
        SingleTonSToast::getInstance().setMessageVDuration(str, 0.25);
    }
}

void ImageBox::showOriginalSize(){
    double old_percentage = m_percentage;
    m_percentage = 1.0;
    m_scale = m_percentage*m_scale/old_percentage;

    update();
}

void ImageBox::showSuitableSize(){
    double pixwidth = static_cast<double>(m_pixmap.width());
    double pixheight = static_cast<double>(m_pixmap.height());

    double showwidth = static_cast<double>(width());
    double showheight = static_cast<double>(height());

    double Wpercentage = showwidth / pixwidth;
    double Hpercentage = showheight / pixheight;

    m_percentage = qMin(Wpercentage, Hpercentage);

    m_suitableWidth = pixwidth*m_percentage;
    m_suitableHeight = pixheight*m_percentage;

    if(m_percentage < 0.01)
    {
        m_percentage = 0.01;
    }
    else if(m_percentage > 1)
    {
        m_percentage = 1.0;
    }

    m_scale = 1.0;

    m_basicX = showwidth/2.0 - pixwidth*m_percentage/2.0;
    m_originX = m_basicX;
    m_basicY = showheight/2.0- pixheight*m_percentage/2.0;
    m_originY = m_basicY;

    update();
}

void ImageBox::zoomIn(){
    ariseScale(1);
}

void ImageBox::zoomOut(){
    ariseScale(-1);
}

void ImageBox::clockWise(){
    QMatrix matrix;
    matrix.rotate(90);

    m_pixmap = m_pixmap.transformed(matrix, Qt::FastTransformation);
    showSuitableSize();
}

void ImageBox::anticlockWise(){
    QMatrix matrix;
    matrix.rotate(-90);

    m_pixmap = m_pixmap.transformed(matrix, Qt::FastTransformation);
    showSuitableSize();
}

void ImageBox::paintEvent(QPaintEvent *event){
    Q_UNUSED(event);

    double pixwidth = static_cast<double>(m_pixmap.width());
    double pixheight = static_cast<double>(m_pixmap.height());
    double showwidth = static_cast<double>(width());
    double showheight = static_cast<double>(height());

    double Wscalerate = pixwidth / showwidth;
    double Hscalerate = pixheight / showheight;
    double compare = qMax(Wscalerate, Hscalerate);

    if(compare < 1.0){
        compare = 1.0;
    }

    QPainter painter(this);

    painter.save();
    QRect backgroundRect = rect();
    QColor color(255, 255, 255, 255);
    painter.setPen(color);
    painter.setBrush(QBrush(color));
    painter.drawRect(backgroundRect);
    painter.restore();

    //此处开始正式绘制m_pixmap
    QRect showRect = QRect(m_originX, m_originY, pixwidth/compare, pixheight/compare);
    painter.save();

    painter.scale(m_scale, m_scale);

    CC_x = m_originX * m_scale;
    CC_y = m_originY * m_scale;
    CC_width = (pixwidth/compare) * m_scale;
    CC_height = (pixheight/compare) * m_scale;

    painter.drawPixmap(showRect, m_pixmap);
    painter.restore();

    //此处开始绘制栅格
    QColor qc(200, 200, 200, 200);
    if(!m_pixmap.isNull()){
        if(Global::label == 16){
            painter.save();
            painter.setPen(qc);
            for(double i = CC_x; i <= CC_x + CC_width; i += 16 * m_percentage){
                painter.drawLine(i, 0, i, showheight);
            }
            for(double i = CC_y; i <= CC_y + CC_height; i += 16 * m_percentage){
                painter.drawLine(0, i, showwidth, i);
            }
            qApp->processEvents();
            painter.restore();
        }
        else if(Global::label == 32){
            painter.save();
            painter.setPen(qc);
            for(double i = CC_x; i <= CC_x + CC_width; i += 32 * m_percentage){
                painter.drawLine(i, 0, i, showheight);
            }
            for(double i = CC_y; i <= CC_y + CC_height; i += 32 * m_percentage){
                painter.drawLine(0, i, showwidth, i);
            }
            qApp->processEvents();
            painter.restore();
        }
    }

    //此处开始绘制被点选的点
    if(!ttar.empty()){
        //QMessageBox::information(NULL, NULL, NULL, NULL);
        painter.save();

        for(int i = 0; i < ttar.size(); i++){
            //Data dd = ttar[i];
            Data dd;
            dd=ttar[i];

            int cc_ = dd.info;
            QColor ccolor(255 - (25 * cc_), 255 - (25 * cc_)
                          , 255 - (25 * cc_), 255);
            painter.setPen(ccolor);

            QPoint pp = dd.point;
            //QRect re = QRect((CC_x + pp.x() * m_percentage) - (m_percentage * 2),
            //                 (CC_y + pp.y() * m_percentage) - (m_percentage * 2),
            //                 m_percentage * 6, m_percentage * 6);
            //painter.drawRect(re);
            pp.setX(CC_x + pp.x() * m_percentage);
            pp.setY(CC_y + pp.y() * m_percentage);
            painter.drawPoint(pp);
        }
        painter.restore();
    }
}

void ImageBox::wheelEvent(QWheelEvent *event){
    int numDegrees = event->delta() / 8;
    int numSteps = numDegrees / 15;
    if (event->orientation() == Qt::Horizontal) {
        event->accept();
    }
    else {
        ariseScale(numSteps);
    }
}

void ImageBox::mousePressEvent(QMouseEvent *event){
    if(Global::drag_position == 1){
        if(event->button() == Qt::LeftButton)
        {
            QCursor cursor;
            cursor.setShape(Qt::OpenHandCursor);
            setCursor(cursor);

            m_pressPoint = event->pos();
        }
        else if(event->button() == Qt::RightButton){
            m_pressPoint = event->pos();
        }
    }
    else if(Global::drag_position == 2 && isInPixmap(event->pos())
            && Global::ispress && Global::label == 0)
    {
        if(event->button() == Qt::LeftButton){
            emit require();
            QPoint point;
            int x=static_cast<int>((event->pos().x() - CC_x) / m_percentage);
            int y=static_cast<int>((event->pos().y() - CC_y) / m_percentage);
            point.setX(x);
            point.setY(y);

            Data temp;
            temp.info = cc;
            temp.point = point;

            ttar.push_back(temp);

            update();
            emit sendData(point);

        }
        else if(event->button() == Qt::RightButton){

        }
    }
    else if(Global::drag_position == 2 && isInPixmap(event->pos())
            && Global::ispress && Global::label != 0){
        if(event->button() == Qt::LeftButton){
            emit require();
            QPoint point;
            Data temp;
            int x=static_cast<int>((event->pos().x() - CC_x) / m_percentage);
            int y=static_cast<int>((event->pos().y() - CC_y) / m_percentage);

            //QMessageBox::information(NULL, NULL, QString::number(x/16) + "," + QString::number(y/16), NULL);

            x /= Global::label;
            y /= Global::label;

            //QMessageBox::information(NULL, NULL, QString::number(x * Global::label) + "," + QString::number(y * Global::label), NULL);
            for(int ii = x * Global::label; ii <= (x + 1) * Global::label; ii++){
                for(int jj = y * Global::label; jj <= (y + 1) * Global::label; jj++){
                    //int xx = static_cast<int>(ii / m_percentage);
                    //int yy = static_cast<int>(jj / m_percentage);

                    point.setX(ii);
                    point.setY(jj);

                    temp.info = cc;
                    temp.point = point;

                    ttar.push_back(temp);

                    update();
                    emit sendData(point);
                }
            }
        }
        else if(event->button() == Qt::RightButton){

        }
    }
}

void ImageBox::mouseReleaseEvent(QMouseEvent *event){
    Q_UNUSED(event);
    if(event->button() == Qt::LeftButton)
    {
        QCursor cursor;
        cursor.setShape(Qt::ArrowCursor);
        setCursor(cursor);

        m_basicX = m_originX;
        m_basicY = m_originY;
    }
    else{
        QCursor cursor;
        cursor.setShape(Qt::ArrowCursor);
        setCursor(cursor);

        m_basicX = m_originX;
        m_basicY = m_originY;
    }
}

void ImageBox::mouseMoveEvent(QMouseEvent *event){
    QPoint move_pos = event->pos();
    QCursor cursor;
    cursor.setShape(Qt::OpenHandCursor);

    if(Global::drag_position == 1){
        if(rect().contains(event->pos()))
        {
            move_pos -= m_pressPoint;
            m_originX = m_basicX + move_pos.x()/m_scale;
            m_originY = m_basicY + move_pos.y()/m_scale;
            update();
        }
        else
        {
            QPoint point;
            if(event->pos().x() < 0)
            {
                point = QPoint(0, event->pos().y());
            }
            else if(event->pos().x() > rect().width()-1)
            {
                point = QPoint(rect().width()-1, event->pos().y());
            }
            else if(event->pos().y() < 0)
            {
                point = QPoint(event->pos().x(), 0);
            }
            else if(event->pos().y() > rect().height()-1)
            {
                point = QPoint(event->pos().x(), rect().height()-1);
            }

            cursor.setPos(mapToGlobal(point));
            setCursor(cursor);
        }
    }
}
void ImageBox::receive(int a)
{
    cc = a+1;
}

void ImageBox::rrefresh(){
    update();
}
