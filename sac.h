#ifndef SAC_H
#define SAC_H

#include <torch/torch.h>
#include <torch/script.h>
#include <tensorboard_logger.h>


#include <QObject>
#include <QOpenGLBuffer>
#include <QCheckBox>
#include <QVector2D>
#include <QVector3D>
#include <QMatrix4x4>

#include "opencv2/aruco.hpp"
#include "opencv2/aruco/charuco.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d.hpp"

#include <yolo_v2_class.hpp>

#include <math.h>
#include <QProcess>
#include <QFile>
#include <QTextStream>
#include <QThread>
#include <QPushButton>

class QLabel;
class LP_ObjectImpl;
class QOpenGLShaderProgram;

class Sac
{
public:
    virtual ~Sac();

        bool Run() override;

        // LP_ActionPlugin interface
        QString MenuName();
        QAction *Trigger();

signals:


        // LP_Functional interface
public slots:

        void Reinforcement_Learning();
        void savedata(QString fileName, std::vector<float> datas);
        void loaddata(std::string fileName, std::vector<float> &datas);

private:
        bool mRunReinforcementLearning = false;
        bool mTraining = false;
        bool gQuit = false;

        double pi = M_PI;

        float total_reward, total_critic_loss, total_policy_loss;
        int episodecount;
};


#endif // SAC_H
