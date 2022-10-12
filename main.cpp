#include "sac.h"

#include <math.h>
#include <fstream>
#include <filesystem>

#include <QVBoxLayout>
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QOpenGLExtraFunctions>
#include <QLabel>
#include <QMatrix4x4>
#include <QPushButton>
#include <QtConcurrent/QtConcurrent>
#include <QFileDialog>
#include <QPainter>

#include <gym_torch.cpp>


/**
 * @brief BulletPhysics Headers
 */

# define M_PI 3.14159265358979323846  /* pi */

//tensorboard --logdir "path"
const std::string kLogFile = "path";

const QString memoryPath("path");
const QString modelPath("path");

int maxepisode = 100000, maxstep = 500, batch_size = 256, total_test_episode = 5;
const float ALPHA = 0.2, GAMMA = 0.99, POLYAK = 0.995;
const int LOG_SIG_MIN = -2, LOG_SIG_MAX = 20, EXPLORE = 10000, SAVEMODELEVERY = 500, START_STEP = 2000;
const double x_threshold = 4 * 2.4;
int STATE_DIM = 1568, ACT_DIM = 1;
float lrp = 1e-3, lrc = 1e-3;
bool gg = false;
torch::Tensor tmp_state;

QFuture<void> gFuture;
QReadWriteLock gLock;
QImage gImage;

torch::Device device(torch::kCPU);

auto build_fc_layers (std::vector<int> dims) {
        torch::nn::Sequential layers;
        for(auto i=0; i<dims.size()-1; i++){
            if(i == dims.size()-2) {
                layers->push_back(torch::nn::LinearImpl(dims[i], dims[i+1]));
            } else {
                layers->push_back(torch::nn::LinearImpl(dims[i], dims[i+1]));
                layers->push_back(torch::nn::ReLUImpl());
            }
        }
        return layers;
}

// Memory
struct Data {
    torch::Tensor before_state;
    torch::Tensor after_state;
    torch::Tensor action;
    torch::Tensor reward;
    torch::Tensor done;
};
std::deque<Data> memory;

struct policy_output
{
    torch::Tensor action;
    torch::Tensor logp_pi;
};

struct PolicyImpl : torch::nn::Module {
    PolicyImpl(std::vector<int> fc_dims)
        : conv1(torch::nn::Conv2dOptions(2, 16, 5).stride(2).padding(2).bias(false)),
          conv2(torch::nn::Conv2dOptions(16, 32, 3).stride(2).padding(1).bias(false)),
          conv3(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)),
          maxpool(torch::nn::MaxPool2dOptions(3).stride({2, 2}))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("maxpool", maxpool);
        mlp = register_module("mlp", build_fc_layers(fc_dims));
        mlp->push_back(torch::nn::ReLUImpl());
        mean_linear = register_module("mean_linear", torch::nn::Linear(fc_dims[fc_dims.size()-1], ACT_DIM));
        log_std_linear = register_module("log_std_linear", torch::nn::Linear(fc_dims[fc_dims.size()-1], ACT_DIM));
    }

    policy_output forward(torch::Tensor state, bool deterministic, bool log_prob) {
//        torch::Tensor x = state;
        torch::Tensor x = conv1(state); // 510*510

        //std::cout << "x: " << x.sizes() << std::endl;

        x = torch::relu(maxpool(x)); // 254*254

        //std::cout << "x: " << x.sizes() << " " << x.dtype() << std::endl;

//        torch::Tensor out_tensor1 = x;
//        out_tensor1 = out_tensor1.index({0, 14}).to(torch::kF32).clone().detach().to(torch::kCPU);
//        std::cout << "out_tensor1: " << out_tensor1.sizes() << " " << out_tensor1.dtype() << std::endl;
//        cv::Mat cv_mat1(254, 254, CV_32FC1, out_tensor1.data_ptr());
//        auto min1 = out_tensor1.min().item().toFloat();
//        auto max1 = out_tensor1.max().item().toFloat();
//        std::cout << "min1: " << min1 << "max1: " << max1 << std::endl;
//        cv_mat1.convertTo(cv_mat1, CV_8U, 255.0/(max1-min1));
//        std::cout << cv_mat1.type() << std::endl;
//        cv::cvtColor(cv_mat1, cv_mat1, CV_GRAY2BGR);

        x = conv2(x); // 254*254

        //std::cout << "x: " << x.sizes() << std::endl;

        x = torch::relu(maxpool(x)); // 126*126

        //std::cout << "x: " << x.sizes() << std::endl;

//        torch::Tensor out_tensor2 = x*255;
//        out_tensor2 = out_tensor2.index({0, 3}).to(torch::kF32).clone().detach().to(torch::kCPU);
//        std::cout << "out_tensor2: " << out_tensor2.sizes() << " " << out_tensor2.dtype() << std::endl;
//        cv::Mat cv_mat2(126, 126, CV_32FC1, out_tensor2.data_ptr());
//        auto min2 = out_tensor2.min().item().toFloat();
//        auto max2 = out_tensor2.max().item().toFloat();
//        std::cout << "min2: " << min2 << "max2: " << max2 << std::endl;
//        cv_mat2.convertTo(cv_mat2, CV_8U);
//        cv::cvtColor(cv_mat2, cv_mat2, CV_GRAY2BGR);

        //x = conv3(x); // 126*126

        //std::cout << "x: " << x.sizes() << std::endl;

        //x = torch::relu(maxpool(x)); // 62*62

        //std::cout << "x: " << x.sizes() << std::endl;

//        torch::Tensor out_tensor3 = x*255;
//        out_tensor3 = out_tensor3.index({0, 2}).to(torch::kF32).clone().detach().to(torch::kCPU);
//        std::cout << "out_tensor3: " << out_tensor3.sizes() << " " << out_tensor3.dtype() << std::endl;
//        cv::Mat cv_mat3(62, 62, CV_32FC1, out_tensor3.data_ptr());
//        auto min3 = out_tensor3.min().item().toFloat();

//        auto max3 = out_tensor3.max().item().toFloat();
//        std::cout << "min3: " << min3 << "max3: " << max3 << std::endl;
//        cv_mat3.convertTo(cv_mat3, CV_8U);
//        cv::cvtColor(cv_mat3, cv_mat3, CV_GRAY2BGR);

        //std::cout << cv_mat1.size() << " " << cv_mat2.size() << " " << cv_mat3.size() << std::endl;

//        gLock.lockForWrite();
//        gWarpedImage = QImage((uchar*) cv_mat1.data, cv_mat1.cols, cv_mat1.rows, cv_mat1.step, QImage::Format_BGR888).copy();
//        gInvWarpImage = QImage((uchar*) cv_mat2.data, cv_mat2.cols, cv_mat2.rows, cv_mat2.step, QImage::Format_BGR888).copy();
//        gEdgeImage = QImage((uchar*) cv_mat3.data, cv_mat3.cols, cv_mat3.rows, cv_mat3.step, QImage::Format_BGR888).copy();
//        gLock.unlock();

        x = x.view({x.size(0), -1});

        //std::cout << "x: " << x.sizes() << std::endl;

        torch::Tensor netout = mlp->forward(x);

        torch::Tensor mean = mean_linear(netout);

        torch::Tensor log_std = log_std_linear(netout);

        log_std = torch::clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX);

        torch::Tensor std = log_std.exp();

        torch::Tensor action;
        if(deterministic){
            // Only used for evaluating policy at test time.
            action = mean;
        } else {
            auto shape = mean.sizes();
            auto eps = torch::randn(shape) * torch::ones(shape, mean.dtype()) + torch::zeros(shape, mean.dtype());
            action = mean + std * eps.to(device);  // for reparameterization trick (mean + std * N(0,1))

//            auto eps = at::normal(0, 1, mean.sizes()).to(mean.device());
//            eps.set_requires_grad(false);
//            action = mean + eps * std;// for reparameterization trick (mean + std * N(0,1))
        }

        //# action rescaling
//        torch::Tensor action_scale = torch::ones({1}).to(device) * 1.0;
//        torch::Tensor action_bias = torch::ones({1}).to(device) * 0.0;

//        static auto logSqrt2Pi = torch::zeros({1}).to(mean.device());
//        static std::once_flag flag;
//        std::call_once(flag, [](){
//            logSqrt2Pi[0] = 2*M_PI;
//            logSqrt2Pi = torch::log(torch::sqrt(logSqrt2Pi));
//        });
//        static auto log_prob_func = [](torch::Tensor value, torch::Tensor mean, torch::Tensor std){
//            auto var = std.pow(2);
//            auto log_scale = std.log();
//            return -(value - mean).pow(2) / (2 * var) - log_scale - logSqrt2Pi;
//        };

        torch::Tensor logp_pi;
        if(log_prob){
            // Calculate log_prob
            auto var = pow(std, 2);
            auto log_scale = log(std);
            logp_pi = -pow(action - mean, 2) / (2.0 * var) - log_scale - log(sqrt(2.0 * M_PI));

            // Enforcing Action Bound
            logp_pi = logp_pi.sum(-1);
            logp_pi -= torch::sum(2.0 * (log(2.0) - action - torch::nn::functional::softplus(-2.0 * action)), 1);
            logp_pi = torch::unsqueeze(logp_pi, -1);

//            logp_pi = log_prob_func(action, mean, std);
//            // Enforcing Action Bound
//            logp_pi -= torch::log(action_scale * (1 - torch::tanh(action).pow(2)) + 1e-6);
//            logp_pi = logp_pi.sum(1, true);
        } else {
            logp_pi = torch::zeros(1).to(device);
        }

        //action = torch::tanh(action) * action_scale + action_bias;
        action = torch::tanh(action);

        policy_output output = {action, logp_pi};

        return output;
    }
    torch::nn::Sequential mlp{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::MaxPool2d maxpool;
    torch::nn::Linear mean_linear{nullptr}, log_std_linear{nullptr};
};
TORCH_MODULE(Policy);


struct MLPQFunctionImpl : torch::nn::Module {
    MLPQFunctionImpl(std::vector<int> fc_dims)
        : conv1(torch::nn::Conv2dOptions(2, 16, 5).stride(2).padding(2).bias(false)),
          conv2(torch::nn::Conv2dOptions(16, 32, 3).stride(2).padding(1).bias(false)),
          conv3(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1).bias(false)),
          maxpool(torch::nn::MaxPool2dOptions(3).stride({2, 2}))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("maxpool", maxpool);
        fc_dims.push_back(1);
        q = register_module("q", build_fc_layers(fc_dims));
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action){
//        torch::Tensor x = state;
        torch::Tensor x = conv1(state); //

        x = torch::relu(maxpool(x)); //

        x = conv2(x); // 254*254

        x = torch::relu(maxpool(x)); //

//        x = conv3(x); // 126*126

//        x = torch::relu(maxpool(x)); // 62*62

        x = x.view({x.size(0), -1});

//        x = torch::cat({x, pick_point}, -1); // 62*62*4+3 = 15379

        //std::cout << "x: " << x.sizes() << std::endl;

        x = q->forward(torch::cat({x, action}, -1));

        //x = torch::squeeze(x, -1);

        return x;
    }
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::MaxPool2d maxpool;
    torch::nn::Sequential q{nullptr};
};
TORCH_MODULE(MLPQFunction);


struct ActorCriticImpl : torch::nn::Module {
    ActorCriticImpl(std::vector<int> policy_fc_dims, std::vector<int> critic_fc_dims) {
          pi = Policy(policy_fc_dims);
          q1 = MLPQFunction(critic_fc_dims);
          q2 = MLPQFunction(critic_fc_dims);
    }

    torch::Tensor act(torch::Tensor state, bool deterministic) {
        torch::NoGradGuard disable;

        policy_output p = pi->forward(state, deterministic, false);

        torch::Tensor action = p.action;

        return action;
    }

    Policy pi{nullptr};
    MLPQFunction q1{nullptr}, q2{nullptr};
};
TORCH_MODULE(ActorCritic);


~LP_Plugin_Sac()
{
    gQuit = true;
    gFuture.waitForFinished();
}

class Sleeper : public QThread
{
public:
    static void usleep(unsigned long usecs){QThread::usleep(usecs);}
    static void msleep(unsigned long msecs){QThread::msleep(msecs);}
    static void sleep(unsigned long secs){QThread::sleep(secs);}
};

bool Run()
{
    mLabel->setText("Left click to start training");

//    mCart_posi = 0;

//    bool b2D = false;
//    CartPole_Continous Env(b2D);
//    Env.reset();
//    for(int i=0; i<50; i++){
//        auto action = Env.sample_action();
//        std::cout << "act: " << action << std::endl;
//        auto out = Env.step(action);
//        std::cout << "mState: " << std::get<0>(out) << ", "
//                  << "reward: " << std::get<1>(out) << ", "
//                  << "done: " << std::get<2>(out) << ", "
//                  << "tmp: " << std::get<3>(out) << ", " << std::endl;
//    }

    return false;
}

void savedata(QString fileName, std::vector<float> datas){
    std::vector<std::string> data_string;

    for (auto i=0; i<datas.size(); i++){
        data_string.push_back(QString("%1").arg(datas[i]).toStdString());
    }

    QByteArray filename = fileName.toLocal8Bit();
    const char *filenamecc = filename.data();
    std::ofstream output_file(filenamecc);
    std::ostream_iterator<std::string> output_iterator(output_file, "\n");
    std::copy(data_string.begin(), data_string.end(), output_iterator);
    output_file.close();
}

void loaddata(std::string fileName, std::vector<float> &datas){
    // Open the File
    std::ifstream in(fileName.c_str());
    std::string str;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Build an istream that holds the input string
        std::istringstream iss(str);

        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0){
            // Iterate over the istream, using >> to grab floats
            // and push_back to store them in the vector
            std::copy(std::istream_iterator<float>(iss),
                  std::istream_iterator<float>(),
                  std::back_inserter(datas));
        }
    }
    //Close The File
    in.close();
}

void Reinforcement_Learning(){
//    for(int i=0; i <10000; i++){
//        QString filename_id = QString(memoryPath + "/%1").arg(i);

//        torch::Tensor before_state_CPU;
//        QString filename_before_state = QString(filename_id + "/before_state.pt");
//        torch::load(before_state_CPU, filename_before_state.toStdString());

//        torch::Tensor after_state_CPU;
//        QString filename_after_state = QString(filename_id + "/after_state.pt");
//        torch::load(after_state_CPU, filename_after_state.toStdString());

//        torch::Tensor action_CPU;
//        QString filename_action = QString(filename_id + "/action.pt");
//        torch::load(action_CPU, filename_action.toStdString());

//        torch::Tensor reward_CPU;
//        QString filename_reward = QString(filename_id + "/reward.pt");
//        torch::load(reward_CPU, filename_reward.toStdString());

//        torch::Tensor done_CPU;
//        QString filename_done = QString(filename_id + "/done.pt");
//        torch::load(done_CPU, filename_done.toStdString());

//        std::cout << "before_state_CPU: " << before_state_CPU.mean() << std::endl
//                  << "after_state_CPU: " << after_state_CPU.mean() << std::endl
//                  << "action_CPU: " << action_CPU << std::endl
//                  << "reward_CPU: " << reward_CPU << std::endl
//                  << "done_CPU: " << done_CPU << std::endl;
//    }

//    return;

    auto rl1current = QtConcurrent::run([this](){
        try {
            torch::manual_seed(0);

            bool b2D = false;
            CartPole_Continous Env(b2D);

            if (torch::cuda::is_available()) {
                std::cout << "CUDA is available! Training on GPU." << std::endl;
                device = torch::Device(torch::kCUDA);
            }

            torch::autograd::DetectAnomalyGuard detect_anomaly;

            qDebug() << "Creating models";

            std::vector<int> policy_mlp_dims{STATE_DIM, 128, 128};
            std::vector<int> critic_mlp_dims{STATE_DIM + ACT_DIM, 128, 128};

            auto actor_critic = ActorCritic(policy_mlp_dims, critic_mlp_dims);
            auto actor_critic_target = ActorCritic(policy_mlp_dims, critic_mlp_dims);

            qDebug() << "Creating optimizer";

            torch::AutoGradMode copy_disable(false);

            std::vector<torch::Tensor> q_params;
            for(size_t i=0; i<actor_critic->q1->parameters().size(); i++){
                q_params.push_back(actor_critic->q1->parameters()[i]);
            }
            for(size_t i=0; i<actor_critic->q2->parameters().size(); i++){
                q_params.push_back(actor_critic->q2->parameters()[i]);
            }

            for(size_t i=0; i < actor_critic_target->pi->parameters().size(); i++){
                actor_critic_target->pi->parameters()[i].copy_(actor_critic->pi->parameters()[i]);
                actor_critic_target->pi->parameters()[i].set_requires_grad(false);
            }
            for(size_t i=0; i < actor_critic_target->q1->parameters().size(); i++){
                actor_critic_target->q1->parameters()[i].copy_(actor_critic->q1->parameters()[i]);
                actor_critic_target->q1->parameters()[i].set_requires_grad(false);
            }
            for(size_t i=0; i < actor_critic_target->q2->parameters().size(); i++){
                actor_critic_target->q2->parameters()[i].copy_(actor_critic->q2->parameters()[i]);
                actor_critic_target->q2->parameters()[i].set_requires_grad(false);
            }

            torch::AutoGradMode copy_enable(true);

            torch::optim::Adam policy_optimizer(actor_critic->pi->parameters(), torch::optim::AdamOptions(lrp));
            torch::optim::Adam critic_optimizer(q_params, torch::optim::AdamOptions(lrc));

            actor_critic->pi->to(device);
            actor_critic->q1->to(device);
            actor_critic->q2->to(device);
            actor_critic_target->pi->to(device);
            actor_critic_target->q1->to(device);
            actor_critic_target->q2->to(device);

            int episode = 0, total_steps = 0;
            float test_reward = 0;

            int thickness = -1;

            GOOGLE_PROTOBUF_VERIFY_VERSION;
            TensorBoardLogger logger(kLogFile.c_str());

            while(episode < maxepisode){
                //qDebug() << "\033[0;35m--------------------------------------------" << "\n";
                //qDebug() << "Episode" << episode << "start";
                //qDebug() << "Step: [" << total_steps << "]\033[0m";
                int step = 0;
                torch::Tensor state, after_state, action, reward, done;
                float episode_critic1_loss = 0, episode_critic2_loss = 0, episode_policy_loss = 0, episode_reward = 0;

                //std::cout << "p para: " << std::endl << actor_critic->state, pi->parameters()[0].mean() << std::endl;
                //std::cout << "mean para: " << std::endl << actor_critic->pi->mean_linear->parameters()[0].mean() << std::endl;
                //std::cout << "q1 para: " << std::endl << actor_critic->q1->parameters()[0].mean() << std::endl;
                //std::cout << "q2 para: " << std::endl << actor_critic->q2->parameters()[0].mean() << std::endl;

                // Reset Environment
                state = Env.reset();
                state = torch::unsqueeze(state.clone().detach(), 0);

                while(step < maxstep){
                    //qDebug() << "\033[0;35m--------------------------------------------";
                    //qDebug() << "Episode" << episode << ", Step [" << step << "/" << maxstep << "] start";
                    //qDebug() << "Total step: [" << total_steps << "]\033[0m";

                    torch::AutoGradMode enable(true);
                    if(total_steps < START_STEP){
                        action = Env.sample_action();
                    } else {
                        auto state_device = state.detach().clone().to(device);
                        action = actor_critic->act(state_device, false);
                        action = action.squeeze(0).to(torch::kCPU);
                        //std::cout << action;
                    }

                    auto out = Env.step(action);
                    after_state = std::get<0>(out);
                    reward = std::get<1>(out);
                    done = std::get<2>(out);

                    episode_reward += reward.detach().item().toFloat();

//                    torch::Tensor out_tensor = after_state;
//                    out_tensor = out_tensor.permute({1, 2, 0})*255;
//                    out_tensor = out_tensor.toType(torch::kByte);
//                    cv::Mat cv_mat(128, 128, CV_8UC3, out_tensor.data_ptr<uchar>());
//                    QString filename_id = QString("/home/cpii/Desktop/1d_vision_test_img/");
//                    QDir().mkdir(filename_id);
//                    QString filename_before_image = QString(filename_id + "/RGB%1.jpg").arg(step);
//                    QByteArray filename_before_imageqb = filename_before_image.toLocal8Bit();
//                    const char *filename_before_imagechar = filename_before_imageqb.data();
//                    cv::imwrite(filename_before_imagechar, cv_mat);

                    after_state = torch::unsqueeze(after_state.clone().detach(), 0);
                    action = torch::unsqueeze(action.clone().detach(), 0);
                    reward = torch::unsqueeze(reward.clone().detach(), 0);
                    done = torch::unsqueeze(done.clone().detach(), 0);

                    if (memory.size() >= 10000) {
                        memory.pop_front();
                    }

                    memory.push_back({
                        state.clone().detach().to(device),
                        after_state.clone().detach().to(device),
                        action.clone().detach().to(device),
                        reward.clone().detach().to(device),
                        done.clone().detach().to(device),
                    });

                    state = after_state.clone().detach();

                    if(memory.size() > batch_size){
                        int randomnum = rand()%(memory.size() - batch_size + 1);

                        torch::Tensor s_batch, s2_batch, a_batch, r_batch, d_batch;

                        s_batch = memory[randomnum].before_state.clone().detach();
                        s2_batch = memory[randomnum].after_state.clone().detach();
                        a_batch = memory[randomnum].action.clone().detach();
                        r_batch = memory[randomnum].reward.clone().detach();
                        d_batch = memory[randomnum].done.clone().detach();
                        for (int i = 1; i < batch_size; i++) {
                            s_batch = torch::cat({ s_batch, memory[randomnum+i].before_state.clone().detach() }, 0);
                            s2_batch = torch::cat({ s2_batch, memory[randomnum+i].after_state.clone().detach() }, 0);
                            a_batch = torch::cat({ a_batch, memory[randomnum+i].action.clone().detach() }, 0);
                            r_batch = torch::cat({ r_batch, memory[randomnum+i].reward.clone().detach() }, 0);
                            d_batch = torch::cat({ d_batch, memory[randomnum+i].done.clone().detach() }, 0);
                        }

                        // Q-value networks training
                        //qDebug() << "Training Q-value networks";
                        torch::AutoGradMode q_enable(true);
                        torch::Tensor q1 = actor_critic->q1->forward(s_batch, a_batch);
                        torch::Tensor q2 = actor_critic->q2->forward(s_batch, a_batch);

                        torch::AutoGradMode disable(false);
                        // Target actions come from *current* policy
                        policy_output next_state_sample = actor_critic->pi->forward(s2_batch, false, true);
                        torch::Tensor a2_batch = next_state_sample.action;
                        torch::Tensor logp_a2 = next_state_sample.logp_pi;

                        // Target Q-values
                        torch::Tensor q1_pi_target = actor_critic_target->q1->forward(s2_batch, a2_batch);
                        torch::Tensor q2_pi_target = actor_critic_target->q2->forward(s2_batch, a2_batch);
                        torch::Tensor backup = r_batch + GAMMA * (1.0 - d_batch) * (torch::min(q1_pi_target, q2_pi_target) - ALPHA * logp_a2);

                        // MSE loss against Bellman backup
                        torch::AutoGradMode loss_enable(true);
                        torch::Tensor loss_q1 = torch::mean(pow(q1 - backup, 2)); // JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                        torch::Tensor loss_q2 = torch::mean(pow(q2 - backup, 2));
//                        std::cout << backup.sizes() << std::endl
//                                  << r_batch.sizes() << std::endl
//                                  << q1_pi_target.sizes() << std::endl
//                                  << logp_a2.sizes() << std::endl
//                                  << q1.sizes() << std::endl;
                        //torch::Tensor loss_q1 = torch::nn::functional::mse_loss(q1, backup);
                        //torch::Tensor loss_q2 = torch::nn::functional::mse_loss(q2, backup);
                        torch::Tensor loss_q = loss_q1 + loss_q2;

                        episode_critic1_loss += loss_q1.detach().item().toFloat();
                        episode_critic2_loss += loss_q2.detach().item().toFloat();

                        critic_optimizer.zero_grad();
                        loss_q.backward();
                        critic_optimizer.step();

                        // Policy network training
                        //qDebug() << "Training policy network";

                        for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
                            actor_critic->q1->parameters()[i].set_requires_grad(false);
                        }
                        for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
                            actor_critic->q2->parameters()[i].set_requires_grad(false);
                        }

                        policy_output sample = actor_critic->pi->forward(s_batch, false, true);
                        torch::Tensor pi = sample.action;
                        torch::Tensor log_pi = sample.logp_pi;
                        torch::Tensor q1_pi = actor_critic->q1->forward(s_batch, pi);
                        torch::Tensor q2_pi = actor_critic->q2->forward(s_batch, pi);
                        torch::Tensor q_pi = torch::min(q1_pi, q2_pi);

                        // Entropy-regularized policy loss
                        torch::Tensor loss_pi = torch::mean(ALPHA * log_pi - q_pi); // Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

                        episode_policy_loss += loss_pi.detach().item().toFloat();

                        policy_optimizer.zero_grad();
                        loss_pi.backward();
                        policy_optimizer.step();

                        for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
                            actor_critic->q1->parameters()[i].set_requires_grad(true);
                        }
                        for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
                            actor_critic->q2->parameters()[i].set_requires_grad(true);
                        }

                        // Update target networks
                        torch::AutoGradMode softcopy_disable(false);
                        for (size_t i = 0; i < actor_critic_target->pi->parameters().size(); i++) {
                            actor_critic_target->pi->parameters()[i].mul_(POLYAK);
                            actor_critic_target->pi->parameters()[i].add_((1.0 - POLYAK) * actor_critic->pi->parameters()[i]);
                        }
                        for (size_t i = 0; i < actor_critic_target->q1->parameters().size(); i++) {
                            actor_critic_target->q1->parameters()[i].mul_(POLYAK);
                            actor_critic_target->q1->parameters()[i].add_((1.0 - POLYAK) * actor_critic->q1->parameters()[i]);
                        }
                        for (size_t i = 0; i < actor_critic_target->q2->parameters().size(); i++) {
                            actor_critic_target->q2->parameters()[i].mul_(POLYAK);
                            actor_critic_target->q2->parameters()[i].add_((1.0 - POLYAK) * actor_critic->q2->parameters()[i]);
                        }
                        torch::AutoGradMode softcopy_enable(true);
                    }
                    step++;
                    total_steps++;

                    bool _done = done.detach().item().toFloat();
                    if(_done){
                        std::cout << "Done! Episode steps: " << step << std::endl;
                        break;
                    }
                }
                episode++;

                // Test model
                int teststep = 0;
                if(episode%20 == 0){
                    test_reward = 0;
                    torch::AutoGradMode test_disable(false);
                    actor_critic->pi->eval();
                    for(int test_episode=0; test_episode < total_test_episode; test_episode++){
                        state = Env.reset();
                        for(int test_step = 0; test_step < maxstep; test_step++){
                            auto state_device = state.detach().clone().to(device);
                            action = actor_critic->act(torch::unsqueeze(state_device, 0), true);
                            action = action.squeeze(0).to(torch::kCPU);

                            auto out = Env.step(action);
                            after_state = std::get<0>(out);
                            reward = std::get<1>(out);
                            done = std::get<2>(out);

                            state = after_state.clone().detach();

                            test_reward += reward.detach().item().toFloat();

                            teststep++;

                            bool _done = done.detach().item().toFloat();
                            if(_done){
                                break;
                            }
                        }
                    }
                    test_reward /= total_test_episode;
                    torch::AutoGradMode test_enable(true);
                    actor_critic->pi->train();
                }

                // Save
                episode_critic1_loss = episode_critic1_loss / (float)step;
                episode_critic2_loss = episode_critic2_loss / (float)step;
                episode_policy_loss = episode_policy_loss / (float)step;

                qDebug() << "\033[0;35m--------------------------------------------" << "\n"
                    << "Episode: " << episode << "\n"
                    << "Episode Step: " << step << "\n"
                    << "Total Step: " << total_steps << "\n"
                    << "Reward: " << episode_reward << "\n"
                    << "Critic_1 Loss: " << episode_critic1_loss << "\n"
                    << "Critic_2 Loss: " << episode_critic2_loss << "\n"
                    << "Policy Loss: " << episode_policy_loss << "\n"
                    << "Test Reward: " << test_reward << "\n"
                    << "--------------------------------------------\033[0m";
                logger.add_scalar("Episode_Reward", episode, episode_reward);
                logger.add_scalar("Episode_Critic1_Loss", episode, episode_critic1_loss);
                logger.add_scalar("Episode_Critic2_Loss", episode, episode_critic2_loss);
                logger.add_scalar("Episode_Policy_Loss", episode, episode_policy_loss);
                logger.add_scalar("Test_Reward", episode, test_reward);

                int save = SAVEMODELEVERY;
                if (episode % save == 0 || (!gg && episode_reward >= 500)) {
                    qDebug() << "Saving memory";
                    if (!gg && episode_reward >= 500) {
                        gg = true;
                    }
//                    for(int i=0; i <memory.size(); i++){
//                        QString filename_id = QString(memoryPath + "/%1").arg(i);
//                        QDir().mkdir(filename_id);

//                        auto before_state_CPU = memory[i].before_state.clone().detach().to(torch::kCPU);
//                        QString filename_before_state = QString(filename_id + "/before_state.pt");
//                        torch::save(before_state_CPU, filename_before_state.toStdString());

//                        auto after_state_CPU = memory[i].after_state.clone().detach().to(torch::kCPU);
//                        QString filename_after_state = QString(filename_id + "/after_state.pt");
//                        torch::save(after_state_CPU, filename_after_state.toStdString());

//                        auto action_CPU = memory[i].action.clone().detach().to(torch::kCPU);
//                        QString filename_action = QString(filename_id + "/action.pt");
//                        torch::save(action_CPU, filename_action.toStdString());

//                        auto reward_CPU = memory[i].reward.clone().detach().to(torch::kCPU);
//                        QString filename_reward = QString(filename_id + "/reward.pt");
//                        torch::save(reward_CPU, filename_reward.toStdString());

//                        auto done_CPU = memory[i].done.clone().detach().to(torch::kCPU);
//                        QString filename_done = QString(filename_id + "/done.pt");
//                        torch::save(done_CPU, filename_done.toStdString());
//                    }

                    qDebug() << "Saving models";

                    QString pi_para_path = QString(modelPath + "/pi_para/pi_para_" + QString::number(episode) + ".pt");
                    QString q1_para_path = QString(modelPath + "/q1_para/q1_para_" + QString::number(episode) + ".pt");
                    QString q2_para_path = QString(modelPath + "/q2_para/q2_para_" + QString::number(episode) + ".pt");
                    QString target_pi_para_path = QString(modelPath + "/target_pi_para/target_pi_para_" + QString::number(episode) + ".pt");
                    QString target_q1_para_path = QString(modelPath + "/target_q1_para/target_q1_para_" + QString::number(episode) + ".pt");
                    QString target_q2_para_path = QString(modelPath + "/target_q2_para/target_q2_para_" + QString::number(episode) + ".pt");
                    QString policy_opti_path = QString(modelPath + "/policy_optimizer/policy_optimizer_" + QString::number(episode) + ".pt");
                    QString critic_opti_path = QString(modelPath + "/critic_optimizer/critic_optimizer_" + QString::number(episode) + ".pt");

                    torch::save(actor_critic->pi->parameters(), pi_para_path.toStdString());
                    torch::save(actor_critic->q1->parameters(), q1_para_path.toStdString());
                    torch::save(actor_critic->q2->parameters(), q2_para_path.toStdString());
                    torch::save(actor_critic_target->pi->parameters(), target_pi_para_path.toStdString());
                    torch::save(actor_critic_target->q1->parameters(), target_q1_para_path.toStdString());
                    torch::save(actor_critic_target->q2->parameters(), target_q2_para_path.toStdString());
                    torch::save(policy_optimizer, policy_opti_path.toStdString());
                    torch::save(critic_optimizer, critic_opti_path.toStdString());

                    std::vector<float> save_episode_num;
                    save_episode_num.push_back(episode);
                    QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
                    savedata(filename_episode_num, save_episode_num);

                    std::vector<float> totalsteps;
                    totalsteps.push_back(total_steps);
                    QString filename_totalsteps = QString(memoryPath + "/totalsteps.txt");
                    savedata(filename_totalsteps, totalsteps);

                    qDebug() << "Models saved";
                }

                qDebug() << "\033[0;34mEpisode " << episode << "finished\033[0m\n"
                         << "--------------------------------------------";
            }
            mTraining = false;
        } catch (const std::exception &e) {
            auto &&msg = torch::GetExceptionString(e);
            qWarning() << msg.c_str();
        } catch (...) {
            qCritical() << "GG";
        }
    });
}
