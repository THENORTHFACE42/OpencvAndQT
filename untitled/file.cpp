#include "file.h"

File::File()
{

}

File::~File()
{

}

void File::conn(QStringList filenamelist){
    QPixmap pixmap;
    QFileInfo fileInfo;
    QString str;

    for(int i = 0; i < filenamelist.count(); i++){
        if(pixmap.load(filenamelist[i])){
            filenameList.push_back(filenamelist[i]);
            pixmapList.push_back(pixmap);

            fileInfo = QFileInfo(filenamelist[i]);
            str = fileInfo.fileName();
            filename.push_back(str);
        }
    }
}

bool File::isEmpty(){
    if(pixmapList.empty()){
        return true;
    }
    return false;
}

//此处需要处理图片没有路径的情况
void File::save(QString filePath, int i){
    QPixmap temp;

    if(filenameList[i] == "default")    //默认处理过的图片的路径为空
    {
        temp = pixmapList[i];
        temp.save(filePath + "/" + filename[i]);

    }
    else{
        temp = pixmapList[i];
        temp.save(filenameList[i]);
    }
}

//此处不需要处理图片没有路径的情况
bool File::save_as(QString filePath, int i){
    if(filePath.isEmpty()){
        return false;
    }
    else{
        QPixmap temp;

        temp = pixmapList[i];
        temp.save(filePath + "/" + filename[i]);
        return true;
    }
}

int File::match(QString s){
    for(int i = 0; i < filename.size(); i++){
        if(s == filename[i]){
            return i;
        }
    }
    return -1;
}

void File::del(int position){
    if(pixmapList.empty()){
        return;
    }
    else{
        pixmapList.erase(pixmapList.begin() + position);
        filename.erase(filename.begin() + position);
        filenameList.erase(filenameList.begin() + position);
    }
}
