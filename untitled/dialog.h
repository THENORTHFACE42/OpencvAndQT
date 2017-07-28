#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>

namespace Ui {
class Dialog;
}

class Dialog : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog(QWidget *parent = 0);
    ~Dialog();

private slots:
   void on_buttonBox_accepted();

signals:
    void  sendData(QString s1,QString s2,QString s3,QString s4, QString s5, QString s6);

private:
    Ui::Dialog *ui;
};

#endif // DIALOG_H
