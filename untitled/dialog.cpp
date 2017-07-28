#include "dialog.h"
#include "ui_dialog.h"

#include<QMessageBox>

Dialog::Dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog)
{
    ui->setupUi(this);
    this->setWindowTitle("ISODATA参数设置");
    this->setWindowIcon(QIcon(":/pic/main/main"));
    this->setFixedSize(287,265);
}

Dialog::~Dialog()
{
    delete ui;
}

void Dialog::on_buttonBox_accepted()
{
    emit sendData(ui->spinBox->text(),ui->lineEdit_two->text(),
                  ui->lineEdit_three->text(),ui->lineEdit_four->text(),
                  ui->lineEdit_five->text(),ui->lineEdit_six->text());
}
