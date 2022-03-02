#include <torch/torch.h>
#include <torch/script.h>

#include <QCoreApplication>

#include <tensorboard_logger.h>

#include <Eigen/Core>
#include <string>
#include <tuple>
#include "gym/gym.h"

#include <filesystem>
#include <memory>
#include <math.h>
#include <QProcess>
#include <QFile>
#include <QTextStream>
#include <QThread>
#include <QDebug>

#include <gym_torch.cpp>

# define M_PI 3.14159265358979323846  /* pi */

const QString memoryPath("/home/cpii/projects/test_memory");
const QString modelPath("/home/cpii/storage_d1/RL1/SAC/test/models");

const QString testdataPath("/home/cpii/storage_d1/RL1/SAC/Python_test_data2/testdata");
const QString traindataPath("/home/cpii/storage_d1/RL1/SAC/Python_test_data2/traindata");
const QString testmodelPath("/home/cpii/storage_d1/RL1/SAC/Python_test_data2/initmodel");
const QString trainrsamplePath("/home/cpii/storage_d1/RL1/SAC/Python_test_data2/eps");

int maxepisode = 10000, maxstep = 4000, batch_size = 8, epsid = 0;
const float ALPHA = 0.2, GAMMA = 0.99, POLYAK = 0.995;
const int LOG_SIG_MIN = -2, LOG_SIG_MAX = 20, EXPLORE = 10000, STATE_DIM = 3, ACT_DIM = 1;
float lrp = 1e-3, lrc = 1e-3;
int traincount = 0, testcount = 0, rsamplecount = 0;

torch::Device device(torch::kCPU);

bool Evaluate = false;

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

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
    torch::Tensor before_pick_point;
    torch::Tensor place_point;
    torch::Tensor reward;
    torch::Tensor done;
    torch::Tensor after_state;
    torch::Tensor after_pick_point;
};
std::deque<Data> memory;

struct policy_output
{
    torch::Tensor action;
    torch::Tensor logp_pi;
};


struct PolicyImpl : torch::nn::Module {
    PolicyImpl(std::vector<int> fc_dims) {
        //conv = register_module("conv", torch::nn::ConvTranspose2d(1, 2, 3));
        mlp = register_module("mlp", build_fc_layers(fc_dims));
        mlp->push_back(torch::nn::ReLUImpl());
        mean_linear = register_module("mean_linear", torch::nn::Linear(fc_dims[fc_dims.size()-1], ACT_DIM));
        log_std_linear = register_module("log_std_linear", torch::nn::Linear(fc_dims[fc_dims.size()-1], ACT_DIM));
    }

    policy_output forward(torch::Tensor state, bool deterministic, bool log_prob) {
        //torch::Tensor x = conv(state);

        torch::Tensor netout = mlp->forward(state);

//        std::cout << "mlp para: " << std::endl << mlp->parameters()[0].mean() << std::endl;
//        std::cout << "mlp: " << netout.mean() << std::endl;

        //x = torch::sigmoid(x);

        torch::Tensor mean = mean_linear(netout);

//        std::cout << "mean para: " << std::endl << mean_linear->parameters()[0].mean() << std::endl;
//        std::cout << "mean: " << mean.mean() << std::endl;

        torch::Tensor log_std = log_std_linear(netout);

//        std::cout << "std para: " << std::endl << log_std_linear->parameters()[0].mean() << std::endl;
//        std::cout << "std: " << log_std.mean() << std::endl;

        log_std = torch::clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX);

        torch::Tensor std = log_std.exp();


//        std::cout << "mean: " << std::endl << mean << std::endl;
//        std::cout << "log_std: " << std::endl << log_std << std::endl;
//        std::cout << "std: " << std::endl << std << std::endl;
//        std::cout << "shape: " << std::endl << shape << std::endl;
//        std::cout << "pi d: " << std::endl << pi_distribution << std::endl;

        torch::Tensor action;
        torch::Tensor rsample;
        if(deterministic){
            // Only used for evaluating policy at test time.
            action = mean;
        } else {
            auto shape = mean.sizes();
            auto eps = torch::randn(shape) * torch::ones(shape, mean.dtype()) + torch::zeros(shape, mean.dtype());
//            std::cout << "eps: " << std::endl << eps << std::endl;


//            QString path = QString(trainrsamplePath + "/%1").arg(epsid);
//            epsid++;
//            // Load values by name
//            std::vector<char> f = get_the_bytes(QString(path + "/eps.pt").toStdString());
//            torch::IValue stateI = torch::pickle_load(f);
//            torch::Tensor eps = stateI.toTensor().detach().to(device);

            rsample = mean + std * eps;  // for reparameterization trick (mean + std * N(0,1))

            action = rsample;
            //std::cout << "rsample: " << std::endl << rsample << std::endl;
        }

        torch::Tensor logp_pi;
        if(log_prob){
            // Calculate log_prob
            auto var = pow(std, 2);
            auto log_scale = log(std);

            logp_pi = -pow(action - mean, 2) / (2.0 * var) - log_scale - log(sqrt(2.0 * M_PI));

            //std::cout << "-pow(action - std, 2): " << std::endl << -pow(action - mean, 2) << std::endl;
            //std::cout << "log(sqrt(2.0 * M_PI)): " << std::endl << log(sqrt(2.0 * M_PI)) << std::endl;

            // Enforcing Action Bound
            logp_pi = logp_pi.sum(-1);
            //std::cout << "log prob: " << std::endl << logp_pi << std::endl;
            logp_pi -= torch::sum(2.0 * (log(2.0) - action - torch::nn::functional::softplus(-2.0 * action)), 1);
            //std::cout << "log prob softplus: " << std::endl << logp_pi << std::endl;
        } else {
            logp_pi = torch::zeros(1);
        }

        action = torch::tanh(action);

        //std::cout << "action tanh: " << std::endl << action << std::endl;

        float ACT_LIMIT = 3.0;
        action = action * ACT_LIMIT;

        policy_output output = {action, logp_pi};

        return output;
    }

    //torch::nn::ConvTranspose2d conv{nullptr};
    torch::nn::Sequential mlp{nullptr};
    torch::nn::Linear mean_linear{nullptr}, log_std_linear{nullptr};
};
TORCH_MODULE(Policy);


struct MLPQFunctionImpl : torch::nn::Module {
    MLPQFunctionImpl(std::vector<int> fc_dims) {
        //conv = register_module("conv", torch::nn::ConvTranspose2d(1, 2, 3));
        fc_dims.push_back(1);
        q = register_module("q", build_fc_layers(fc_dims));
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action){
        //torch::Tensor x = conv(state);

        torch::Tensor x = q->forward(torch::cat({state, action}, -1));

        x = torch::squeeze(x, -1);

        return x;
    }

    //torch::nn::ConvTranspose2d conv{nullptr};
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

        //std::cout << "action: " << std::endl << action << std::endl;

        return action;
    }

    Policy pi{nullptr};
    MLPQFunction q1{nullptr}, q2{nullptr};
};
TORCH_MODULE(ActorCritic);


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

int main(int argc, char *argv[])
{
//    std::vector<float> tt{1, 1, 1, 1, 2, 2, 2, 2};
//    auto s_batch = torch::from_blob(tt.data(), { 2, 2, 2 }, at::kFloat);
//    std::cout << "s_batch: " << std::endl << s_batch << std::endl;
//    s_batch = s_batch.flatten(1);
//    std::cout << "s_batch: " << std::endl << s_batch << std::endl;

//    std::vector<float> tt2{3, 3, 3, 3};
//    auto a_batch = torch::from_blob(tt2.data(), {1, 2, 2 }, at::kFloat);
//    std::cout << "a_batch: " << std::endl << a_batch << std::endl;
//    a_batch = a_batch.flatten(1);
//    std::cout << "a_batch: " << std::endl << a_batch << std::endl;

//    torch::Tensor cat = torch::cat({s_batch, a_batch}, 0);
//    std::cout << "cat: " << std::endl << cat << std::endl;

//    cat = torch::reshape(cat, {3, 2, 2});
//    std::cout << "cat: " << std::endl << cat << std::endl;

//    return 0;

//    QString m1 = QString(testmodelPath + "/para%1.pt").arg(0);
//    QString m2 = QString(testmodelPath + "/para%1.pt").arg(1);
//    // Load values by name
//    std::vector<char> f = get_the_bytes(m1.toStdString());
//    torch::IValue m1I = torch::pickle_load(f);
//    torch::Tensor m1T = m1I.toTensor();

//    std::vector<char> f2 = get_the_bytes(m2.toStdString());
//    torch::IValue m2I = torch::pickle_load(f2);
//    torch::Tensor m2T = m2I.toTensor();


//    std::cout << "m1: " << std::endl << m1T << std::endl;
//    std::cout << "m2: " << std::endl << m2T << std::endl;


//    for (const auto & file : std::filesystem::directory_iterator(traindataPath.toStdString())){
//        traincount++;
//    }
//    for (const auto & file : std::filesystem::directory_iterator(testdataPath.toStdString())){
//        testcount++;
//    }
//    qDebug() << "traincount: " << traincount;
//    qDebug() << "testcount: " << testcount;

//    for(int i=testcount-4000; i<testcount; i++){
//        QString path = QString(trainrsamplePath + "/%1").arg(i);
//        // Load values by name
//        std::vector<char> f = get_the_bytes(QString(path + "/rsample.pt").toStdString());
//        torch::IValue stateI = torch::pickle_load(f);
//        auto rsample = stateI.toTensor().to(device);

//        std::cout << rsample << std::endl;

//        QString tpath = QString(testdataPath + "/%1").arg(i);
//        // Load values by name
//        std::vector<char> f2 = get_the_bytes(QString(tpath + "/action.pt").toStdString());
//        torch::IValue stateI2 = torch::pickle_load(f2);
//        auto statet = stateI2.toTensor().to(device);

//        std::cout << statet << std::endl;


//        QString rpath = QString(traindataPath + "/%1").arg(i);
//        // Load values by name
//        std::vector<char> f3 = get_the_bytes(QString(rpath + "/action.pt").toStdString());
//        torch::IValue stateI3 = torch::pickle_load(f3);
//        auto stater = stateI3.toTensor().to(device);

//        std::cout << stater << std::endl;

//        QString state_path = QString(testdataPath + "/%1/state.pt").arg(0);
//        QString action_path = QString(testdataPath + "/%1/action.pt").arg(0);
//        // Load values by name
//        std::vector<char> f = get_the_bytes(state_path.toStdString());
//        torch::IValue state = torch::pickle_load(f);
//        torch::Tensor statet = state.toTensor();

//        std::vector<char> f2 = get_the_bytes(action_path.toStdString());
//        torch::IValue action = torch::pickle_load(f2);
//        torch::Tensor actiond = action.toTensor();

//        std::cout << "state: " << statet << std::endl;
//        std::cout << "ad: " << actiond << std::endl;
//    }


    for (const auto & file : std::filesystem::directory_iterator(traindataPath.toStdString())){
        traincount++;
    }
    for (const auto & file : std::filesystem::directory_iterator(testdataPath.toStdString())){
        testcount++;
    }
    qDebug() << "traincount: " << traincount;
    qDebug() << "testcount: " << testcount;

    torch::manual_seed(0);

//    if (torch::cuda::is_available()) {
//        std::cout << "CUDA is available! Training on GPU." << std::endl;
//        device = torch::Device(torch::kCUDA);
//    } else {
//        std::cout << "CUDA is not available! Training on CPU." << std::endl;
//    }

    //torch::autograd::DetectAnomalyGuard detect_anomaly;

    qDebug() << "Creating models";

    std::vector<int> policy_mlp_dims{STATE_DIM, 4, 4};
    std::vector<int> critic_mlp_dims{STATE_DIM + ACT_DIM, 4, 4};

    auto actor_critic = ActorCritic(policy_mlp_dims, critic_mlp_dims);
    auto actor_critic_target = ActorCritic(policy_mlp_dims, critic_mlp_dims);


    qDebug() << "Creating optimizer";

    torch::AutoGradMode copy_disable(false);
//    for(size_t i=0; i<actor_critic->pi->parameters().size(); i++){
//        QString m = QString(testmodelPath + "/para%1.pt").arg(i);
//        std::vector<char> f = get_the_bytes(m.toStdString());
//        torch::IValue I = torch::pickle_load(f);
//        torch::Tensor t = I.toTensor();
//        actor_critic->pi->parameters()[i].copy_(t);

//        //std::cout << "p para: " << std::endl << t << std::endl;
//    }

    std::vector<torch::Tensor> q_params;
    for(size_t i=0; i<actor_critic->q1->parameters().size(); i++){
//        QString m = QString(testmodelPath + "/para%1.pt").arg(i + actor_critic->pi->parameters().size());
//        std::vector<char> f = get_the_bytes(m.toStdString());
//        torch::IValue I = torch::pickle_load(f);
//        torch::Tensor t = I.toTensor();
//        actor_critic->q1->parameters()[i].copy_(t);
        q_params.push_back(actor_critic->q1->parameters()[i]);

        //std::cout << "q1 para: " << std::endl << t << std::endl;
    }
    for(size_t i=0; i<actor_critic->q2->parameters().size(); i++){
//        QString m = QString(testmodelPath + "/para%1.pt").arg(i + actor_critic->pi->parameters().size() + actor_critic->q1->parameters().size());
//        std::vector<char> f = get_the_bytes(m.toStdString());
//        torch::IValue I = torch::pickle_load(f);
//        torch::Tensor t = I.toTensor();
//        actor_critic->q2->parameters()[i].copy_(t);
        q_params.push_back(actor_critic->q2->parameters()[i]);

        //std::cout << "q2 para: " << std::endl << t << std::endl;
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

    // ------------------------------------------------------------------------------------------------
//    QString pi_para_path = QString(modelPath + "/pi_para_" + QString::number(0) + ".pt");
//    QString q1_para_path = QString(modelPath + "/q1_para_" + QString::number(0) + ".pt");
//    QString q2_para_path = QString(modelPath + "/q2_para_" + QString::number(0) + ".pt");
//    QString target_pi_para_path = QString(modelPath + "/target_pi_para_" + QString::number(0) + ".pt");
//    QString target_q1_para_path = QString(modelPath + "/target_q1_para_" + QString::number(0) + ".pt");
//    QString target_q2_para_path = QString(modelPath + "/target_q2_para_" + QString::number(0) + ".pt");
//    QString policy_opti_path = QString(modelPath + "/policy_optimizer_" + QString::number(0) + ".pt");
//    QString critic_opti_path = QString(modelPath + "/critic_optimizer_" + QString::number(0) + ".pt");

//    torch::save(actor_critic->pi->parameters(), pi_para_path.toStdString());
//    torch::save(actor_critic->q1->parameters(), q1_para_path.toStdString());
//    torch::save(actor_critic->q2->parameters(), q2_para_path.toStdString());
//    torch::save(actor_critic_target->pi->parameters(), target_pi_para_path.toStdString());
//    torch::save(actor_critic_target->q1->parameters(), target_q1_para_path.toStdString());
//    torch::save(actor_critic_target->q2->parameters(), target_q2_para_path.toStdString());
//    torch::save(policy_optimizer, policy_opti_path.toStdString());
//    torch::save(critic_optimizer, critic_opti_path.toStdString());

//    QString pi_para_path = QString(modelPath + "/pi_para_" + QString::number(0) + ".pt");
//    QString q1_para_path = QString(modelPath + "/q1_para_" + QString::number(0) + ".pt");
//    QString q2_para_path = QString(modelPath + "/q2_para_" + QString::number(0) + ".pt");
//    QString target_pi_para_path = QString(modelPath + "/target_pi_para_" + QString::number(0) + ".pt");
//    QString target_q1_para_path = QString(modelPath + "/target_q1_para_" + QString::number(0) + ".pt");
//    QString target_q2_para_path = QString(modelPath + "/target_q2_para_" + QString::number(0) + ".pt");
//    QString policy_opti_path = QString(modelPath + "/policy_optimizer_" + QString::number(0) + ".pt");
//    QString critic_opti_path = QString(modelPath + "/critic_optimizer_" + QString::number(0) + ".pt");

//    std::vector<torch::Tensor> pi_para, q1_para, q2_para, target_pi_para, target_q1_para, target_q2_para;

//    torch::load(pi_para, pi_para_path.toStdString());
//    torch::load(q1_para, q1_para_path.toStdString());
//    torch::load(q2_para, q2_para_path.toStdString());
//    torch::load(target_pi_para, target_pi_para_path.toStdString());
//    torch::load(target_q1_para, target_q1_para_path.toStdString());
//    torch::load(target_q2_para, target_q2_para_path.toStdString());
//    torch::load(policy_optimizer, policy_opti_path.toStdString());
//    torch::load(critic_optimizer, critic_opti_path.toStdString());

//    torch::AutoGradMode data_copy_disable(false);
//    for(size_t i=0; i < actor_critic->pi->parameters().size(); i++){
//        actor_critic->pi->parameters()[i].copy_(pi_para[i]);
//    }
//    for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
//        actor_critic->q1->parameters()[i].copy_(q1_para[i]);
//    }
//    for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
//        actor_critic->q2->parameters()[i].copy_(q2_para[i]);
//    }
//    for(size_t i=0; i < actor_critic->pi->parameters().size(); i++){
//        actor_critic_target->pi->parameters()[i].copy_(target_pi_para[i]);
//    }
//    for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
//        actor_critic_target->q1->parameters()[i].copy_(target_q1_para[i]);
//    }
//    for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
//        actor_critic_target->q2->parameters()[i].copy_(target_q2_para[i]);
//    }
//    torch::AutoGradMode data_copy_enable(true);

//    std::cout << actor_critic->pi->parameters() << std::endl
//              << actor_critic->q1->parameters() << std::endl
//              << actor_critic->q2->parameters() << std::endl
//              << actor_critic_target->pi->parameters() << std::endl
//              << actor_critic_target->q1->parameters() << std::endl
//              << actor_critic_target->q2->parameters() << std::endl;

//    return 0;

//    torch::Tensor s_batch, a_batch, r_batch, d_batch, s2_batch;

//    QString memoryp = QString(traindataPath + "/%1").arg(0);

//    // Load values by name
//    std::vector<char> f = get_the_bytes(QString(memoryp + "/state.pt").toStdString());
//    torch::IValue stateI = torch::pickle_load(f);
//    s_batch = stateI.toTensor().to(device);

//    std::vector<char> f2 = get_the_bytes(QString(memoryp + "/action.pt").toStdString());
//    torch::IValue actionI = torch::pickle_load(f2);
//    a_batch = actionI.toTensor().to(device);

//    std::vector<char> f3 = get_the_bytes(QString(memoryp + "/reward.pt").toStdString());
//    torch::IValue rewardI = torch::pickle_load(f3);
//    r_batch = rewardI.toTensor().to(device);

//    std::vector<char> f4 = get_the_bytes(QString(memoryp + "/next_state.pt").toStdString());
//    torch::IValue next_stateI = torch::pickle_load(f4);
//    s2_batch = next_stateI.toTensor().to(device);

//    std::vector<char> f5 = get_the_bytes(QString(memoryp + "/done.pt").toStdString());
//    torch::IValue doneI = torch::pickle_load(f5);
//    d_batch = doneI.toTensor().to(device);

//    std::vector<float> tt{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
//    s_batch = torch::from_blob(tt.data(), { 2, 4 }, at::kFloat);

//    std::vector<float> tt2{0.1, 0.2};
//    a_batch = torch::from_blob(tt2.data(), { 2, 1 }, at::kFloat);

//    std::vector<float> tt3{0.2, 0.3};
//    r_batch = torch::from_blob(tt3.data(), { 2, 1 }, at::kFloat);

//    std::vector<float> tt4{0, 0};
//    d_batch = torch::from_blob(tt4.data(), { 2, 1 }, at::kFloat);

//    std::vector<float> tt5{0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
//    s2_batch = torch::from_blob(tt5.data(), { 2, 4 }, at::kFloat);

//    std::cout << "p para: " << std::endl << actor_critic->pi->parameters() << std::endl;
//    std::cout << "q1 para: " << std::endl << actor_critic->q1->parameters() << std::endl;
//    std::cout << "q2 para: " << std::endl << actor_critic->q2->parameters() << std::endl;

//    // Q-value networks training
//    //qDebug() << "Training Q-value networks";
//    torch::AutoGradMode train_enable(true);

//    torch::Tensor q1 = actor_critic->q1->forward(s_batch, a_batch);
//    torch::Tensor q2 = actor_critic->q2->forward(s_batch, a_batch);

//    torch::AutoGradMode disable(false);
//    // Target actions come from *current* policy
//    policy_output next_state_sample = actor_critic->pi->forward(s2_batch, false, true);
//    torch::Tensor a2_batch = next_state_sample.action;
//    torch::Tensor logp_a2 = next_state_sample.logp_pi;
//    // Target Q-values
//    torch::Tensor q1_pi_target = actor_critic_target->q1->forward(s2_batch, a2_batch);
//    torch::Tensor q2_pi_target = actor_critic_target->q2->forward(s2_batch, a2_batch);
//    torch::Tensor backup = r_batch + GAMMA * (1.0 - d_batch) * (torch::min(q1_pi_target, q2_pi_target) - ALPHA * logp_a2);

//    // MSE loss against Bellman backup
//    torch::AutoGradMode loss_enable(true);
//    torch::Tensor loss_q1 = torch::mean(pow(q1 - backup, 2));
//    torch::Tensor loss_q2 = torch::mean(pow(q2 - backup, 2));
//    torch::Tensor loss_q = loss_q1 + loss_q2;

//    critic_optimizer.zero_grad();
//    loss_q.backward();
//    critic_optimizer.step();

//    std::cout << "q1: " << std::endl << q1 << std::endl;
//    std::cout << "q2: " << std::endl << q2 << std::endl;
//    std::cout << "a2: " << std::endl << a2_batch << std::endl;
//    std::cout << "logp_a2: " << std::endl << logp_a2 << std::endl;
//    std::cout << "q1_pi_targ: " << std::endl << q1_pi_target << std::endl;
//    std::cout << "q2_pi_targ: " << std::endl << q2_pi_target << std::endl;
//    std::cout << "backup: " << std::endl << backup << std::endl;
//    std::cout << "loss_q1: " << std::endl << loss_q1 << std::endl;
//    std::cout << "loss_q2: " << std::endl << loss_q2 << std::endl;

//    std::cout << "p para: " << std::endl << actor_critic->pi->parameters() << std::endl;
//    std::cout << "q1 para: " << std::endl << actor_critic->q1->parameters()<< std::endl;
//    std::cout << "q2 para: " << std::endl << actor_critic->q2->parameters() << std::endl;

//    // Policy network training
//    //qDebug() << "Training policy network";

//    for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
//        actor_critic->q1->parameters()[i].set_requires_grad(false);
//    }
//    for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
//        actor_critic->q2->parameters()[i].set_requires_grad(false);
//    }

//    policy_output sample = actor_critic->pi->forward(s_batch, false, true);
//    torch::Tensor pi = sample.action;
//    torch::Tensor log_pi = sample.logp_pi;
//    torch::Tensor q1_pi = actor_critic->q1->forward(s_batch, pi);
//    torch::Tensor q2_pi = actor_critic->q2->forward(s_batch, pi);
//    torch::Tensor q_pi = torch::min(q1_pi, q2_pi);

//    // Entropy-regularized policy loss
//    torch::Tensor loss_pi = torch::mean(ALPHA * log_pi - q_pi); // Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

//    policy_optimizer.zero_grad();
//    loss_pi.backward();
//    policy_optimizer.step();

//    for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
//        actor_critic->q1->parameters()[i].set_requires_grad(true);
//    }
//    for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
//        actor_critic->q2->parameters()[i].set_requires_grad(true);
//    }

//    std::cout << "pi: " << std::endl << pi << std::endl;
//    std::cout << "log_pi: " << std::endl << log_pi << std::endl;
//    std::cout << "q1_pi: " << std::endl << q1_pi << std::endl;
//    std::cout << "q2_pi: " << std::endl << q2_pi << std::endl;
//    std::cout << "loss_pi: " << std::endl << loss_pi << std::endl;

//    std::cout << "p para: " << std::endl << actor_critic->pi->parameters() << std::endl;
//    std::cout << "q1 para: " << std::endl << actor_critic->q1->parameters()<< std::endl;
//    std::cout << "q2 para: " << std::endl << actor_critic->q2->parameters() << std::endl;

//    return 0;

//    QString state_path = QString(testdataPath + "/%1/state.pt").arg(0);
//    QString action_path = QString(testdataPath + "/%1/action.pt").arg(0);
//    QString state_bach = QString(traindataPath + "/%1/state.pt").arg(0);
//    // Load values by name
//    std::vector<char> f = get_the_bytes(state_path.toStdString());
//    torch::IValue state = torch::pickle_load(f);
//    torch::Tensor statet = state.toTensor();

//    std::vector<char> f2 = get_the_bytes(action_path.toStdString());
//    torch::IValue action = torch::pickle_load(f2);
//    torch::Tensor actiont = action.toTensor();

//    std::vector<char> f3 = get_the_bytes(state_bach.toStdString());
//    torch::IValue state2 = torch::pickle_load(f3);
//    torch::Tensor statet2 = state2.toTensor();

//    torch::Tensor testpiact = actor_critic->act(statet, false);
//    policy_output testpi = actor_critic->pi->forward(statet2, false, true);
//    auto a = testpi.action;
//    auto pr = testpi.logp_pi;
//    torch::Tensor testq1 = actor_critic->q1->forward(statet, actiont);
//    torch::Tensor testq2 = actor_critic->q2->forward(statet, actiont);

//    std::cout << "act: " << std::endl << testpiact << std::endl;
//    std::cout << "pi action: " << std::endl << a << std::endl;
//    std::cout << "pi prob: " << std::endl << pr << std::endl;
//    std::cout << "q1: " << std::endl << testq1 << std::endl;
//    std::cout << "q2: " << std::endl << testq2 << std::endl;

//    return 0;

//    torch::Tensor rand4_batch = torch::rand({5,4});
//    torch::Tensor rand4 = torch::rand({4});
//    torch::Tensor rand1 = torch::rand({1});

//    torch::Tensor pout_test = actor_critic->act(rand4, true);
//    std::cout << "p out test: " << std::endl << pout_test << std::endl << "------------------------------" << std::endl;

//    policy_output pout_train = actor_critic->pi->forward(rand4_batch, false, true);
//    std::cout << "p out train: " << std::endl << pout_train.action << std::endl << pout_train.logp_pi << std::endl << "------------------------------" << std::endl;

//    torch::Tensor q1out = actor_critic->q1->forward(rand4, rand1);
//    std::cout << "q1 out: " << std::endl << q1out << std::endl << "------------------------------" << std::endl;

//    torch::Tensor q2out = actor_critic->q2->forward(rand4, rand1);
//    std::cout << "q2 out: " << std::endl << q2out << std::endl << "------------------------------" << std::endl;

//    return 0;

//    policy->to(device);
//    critic->to(device);
//    target_critic->to(device);

    int episode = 0;
    int total_steps = 0;

//    bool RestoreFromCheckpoint = false;
//    if(RestoreFromCheckpoint){
//        qDebug() << "Loading models";

//        QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
//        std::vector<float> saved_episode_num;
//        loaddata(filename_episode_num.toStdString(), saved_episode_num);
//        episode = saved_episode_num[0]-1;
//        maxepisode += episode;

//        QString Pmodelname = QString(modelPath + "/policy_model_" + QString::number(episode) + ".pt");
//        QString Poptimodelname = QString(modelPath + "/policy_optimizer_" + QString::number(episode) + ".pt");
//        QString Cmodelname = QString(modelPath + "/critic__model_" + QString::number(episode) + ".pt");
//        QString Coptimodelname = QString(modelPath + "/critic__optimizer_" + QString::number(episode) + ".pt");
//        QString Cmodeltarget1name = QString(modelPath + "/critic_target__model_" + QString::number(episode) + ".pt");
//        torch::load(actor_critic, Pmodelname.toStdString());
//        torch::load(policy_optimizer, Poptimodelname.toStdString());
//        torch::load(critic, Cmodelname.toStdString());
//        torch::load(critic_optimizer, Coptimodelname.toStdString());
//        torch::load(target_critic, Cmodeltarget1name.toStdString());
//    } else {
//        qDebug() << "Copying parameters to target models";
//        torch::AutoGradMode hardcopy_disable(false);
//        for(size_t i=0; i < target_critic->parameters().size(); i++){
//            target_critic->parameters()[i].copy_(critic->parameters()[i]);
//        }
//    }

    const std::string kLogFile = "/home/cpii/projects/log/test/Pendulum_v0_rand_eps/tfevents.pb";
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    TensorBoardLogger logger(kLogFile.c_str());

    std::vector<int> teststeps;
    for(int i=0; i<100; i++){
        teststeps.push_back(i*2000);
    }
    int total_teststeps = 0;

    while(episode < maxepisode){
        qDebug() << "\033[0;35m--------------------------------------------" << "\n";
        qDebug() << "Episode" << episode << "start\033[0m";
        qDebug() << "Step: [" << total_steps << "/" << traincount << "]";
        int step = 0, train_number = 0, tmp = 0;
        bool done = false;
        float episode_critic1_loss = 0, episode_critic2_loss = 0, episode_policy_loss = 0, episode_reward = 0;

        std::cout << "p para: " << std::endl << actor_critic->pi->parameters()[0].mean() << std::endl;
        //std::cout << "mean para: " << std::endl << actor_critic->pi->mean_linear->parameters()[0].mean() << std::endl;
        std::cout << "q1 para: " << std::endl << actor_critic->q1->parameters()[0].mean() << std::endl;
        std::cout << "q2 para: " << std::endl << actor_critic->q2->parameters()[0].mean() << std::endl;

        // Reset Environment

        torch::Tensor pmean;

        while(step < maxstep){
            //qDebug() << "\033[0;35m--------------------------------------------";
            //qDebug() << "Episode" << episode << ", Step[" << step << "/" << maxstep << "] start\033[0m";
            //qDebug() << "Total step: [" << total_steps << "/" << traincount << "]";
            float reward = 0;
            torch::Tensor state, action, after_state;

//            std::cout << "p para: " << std::endl << actor_critic->pi->parameters()[0].mean() << std::endl;
//            std::cout << "mean para: " << std::endl << actor_critic->pi->mean_linear->parameters()[0].mean() << std::endl;
//            std::cout << "q1 para: " << std::endl << actor_critic->q1->parameters()[0].mean() << std::endl;
//            std::cout << "q2 para: " << std::endl << actor_critic->q2->parameters()[0].mean() << std::endl;

//            if(total_steps>1 && pmean.item().toFloat() == actor_critic->pi->parameters()[0].mean().item().toFloat()){
//                qDebug() << total_steps;
//                break;
//            }

//            pmean = actor_critic->pi->parameters()[0].mean();

//            if(total_steps+1>traincount){
//                break;
//            }

            // Do action


            // After state & reward

            // Test reward

            //done = env d;
            //after_state = env after_state;

            // Train models
            //if(memory.size() > batch_size){
                torch::Tensor s_batch, a_batch, r_batch, d_batch, s2_batch;

                QString memoryp = QString(traindataPath + "/%1").arg(total_steps);

                // Load values by name
                std::vector<char> f = get_the_bytes(QString(memoryp + "/state.pt").toStdString());
                torch::IValue stateI = torch::pickle_load(f);
                s_batch = stateI.toTensor().to(device);

                std::vector<char> f2 = get_the_bytes(QString(memoryp + "/action.pt").toStdString());
                torch::IValue actionI = torch::pickle_load(f2);
                a_batch = actionI.toTensor().to(device);

                std::vector<char> f3 = get_the_bytes(QString(memoryp + "/reward.pt").toStdString());
                torch::IValue rewardI = torch::pickle_load(f3);
                r_batch = rewardI.toTensor().to(device);

                std::vector<char> f4 = get_the_bytes(QString(memoryp + "/next_state.pt").toStdString());
                torch::IValue next_stateI = torch::pickle_load(f4);
                s2_batch = next_stateI.toTensor().to(device);

                std::vector<char> f5 = get_the_bytes(QString(memoryp + "/done.pt").toStdString());
                torch::IValue doneI = torch::pickle_load(f5);
                d_batch = doneI.toTensor().to(device);


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
                torch::Tensor loss_q1 = torch::mean(pow(q1 - backup, 2));
                torch::Tensor loss_q2 = torch::mean(pow(q2 - backup, 2));
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
                //qDebug() << "Updating target models";
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

                train_number++;
            //}

//            memory.push_back({

//                             });

//            if(done){
//                break;
//            }

            step++;
            total_steps++;
        }

        // Test model
        int teststep = teststeps[episode] - total_teststeps;
        for(int i=0; i<teststep; i++){
            if(i+total_teststeps>testcount){
                break;
            }
            QString state_path = QString(testdataPath + "/%1/state.pt").arg(i + total_teststeps);
            QString action_path = QString(testdataPath + "/%1/action.pt").arg(i + total_teststeps);
            // Load values by name
            std::vector<char> f = get_the_bytes(state_path.toStdString());
            torch::IValue state = torch::pickle_load(f);
            torch::Tensor statet = state.toTensor();

            std::vector<char> f2 = get_the_bytes(action_path.toStdString());
            torch::IValue action = torch::pickle_load(f2);
            torch::Tensor actiond = action.toTensor();

            torch::Tensor actionp = actor_critic->act(statet, true);

            float reward = -torch::sum(abs(actionp - actiond)).item().toFloat();

            episode_reward += reward;
            //std::cout << "ad: " << actiond << std::endl << "ap: " << actionp << std::endl;
            //std::cout << "reward: " << std::endl << reward << std::endl;
            //std::cout << "state: " << std::endl << state << std::endl;
        }
        total_teststeps += teststep;

        // Save
        episode++;

        episode_critic1_loss = episode_critic1_loss / (float)train_number;
        episode_critic2_loss = episode_critic2_loss / (float)train_number;
        episode_policy_loss = episode_policy_loss / (float)train_number;

        qDebug() << "\033[0;35m--------------------------------------------" << "\n"
            << "Episode: " << episode << "\n"
            //<< "Done(1:yes, 0:no): " << done << "\n"
            << "Reward: " << episode_reward << "\n"
            << "Critic_1 Loss: " << episode_critic1_loss << "\n"
            << "Critic_2 Loss: " << episode_critic2_loss << "\n"
            << "Policy Loss: " << episode_policy_loss << "\n"
            << "--------------------------------------------\033[0m";
        logger.add_scalar("Episode_Reward", episode, episode_reward);
        logger.add_scalar("Episode_Critic1_Loss", episode, episode_critic1_loss);
        logger.add_scalar("Episode_Critic2_Loss", episode, episode_critic2_loss);
        logger.add_scalar("Episode_Policy_Loss", episode, episode_policy_loss);

//        if (episode % 50 == 0) {
//            qDebug() << "Saving models";
//            QString Pmodelname = QString(modelPath + "/policy_model_" + QString::number(episode) + ".pt");
//            QString Poptimodelname = QString(modelPath + "/policy_optimizer_" + QString::number(episode) + ".pt");
//            QString Cmodelname = QString(modelPath + "/critic_model_" + QString::number(episode) + ".pt");
//            QString Coptimodelname = QString(modelPath + "/critic_optimizer_" + QString::number(episode) + ".pt");
//            QString Cmodeltargetname = QString(modelPath + "/critic_target_model_" + QString::number(episode) + ".pt");
//            torch::save(policy, Pmodelname.toStdString());
//            torch::save(policy_optimizer, Poptimodelname.toStdString());
//            torch::save(critic, Cmodelname.toStdString());
//            torch::save(critic_optimizer, Coptimodelname.toStdString());
//            torch::save(target_critic, Cmodeltargetname.toStdString());

//            std::vector<float> save_episode_num;
//            save_episode_num.push_back(episode+1);
//            QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
//            savedata(filename_episode_num, save_episode_num);

//            std::vector<float> save_total_steps;
//            save_total_steps.push_back(total_steps);
//            QString filename_totalsteps = QString(memoryPath + "/total_steps.txt");
//            savedata(filename_totalsteps, save_total_steps);
//            qDebug() << "Models saved";
//        }

        qDebug() << "\033[0;34mEpisode " << episode << "finished\033[0m\n"
                 << "--------------------------------------------";

//        if(pmean.item().toFloat() == actor_critic->pi->parameters()[0].mean().item().toFloat()){
//            qDebug() << total_steps;
//            break;
//        }

        if(total_steps+1>traincount){
            qDebug() << "Finished";
            break;
        }

    }

    return 0;
}
