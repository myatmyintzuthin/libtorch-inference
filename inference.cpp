#include <iostream>
#include <torch/torch.h>
#include <torch/script.h> 
#include <opencv2/opencv.hpp>
#include <argparse/argparse.hpp>
using namespace cv;
using namespace std;

void getInput(argparse::ArgumentParser &program, int argc, const char* argv[]) {
    program.add_argument("--model_path")
        .required()
        .help("specify script model path");
    program.add_argument("--image_path")
        .required()
        .help("specify inference image path");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }
}

torch::Tensor processImage(string image_path){
    
    Mat image, resize_image;
    std::vector<double> norm_mean = {0.485, 0.456, 0.406};
    std::vector<double> norm_std = {0.229, 0.224, 0.225};
    image = imread( image_path, IMREAD_COLOR);
    resize(image, resize_image, Size(224, 224), INTER_LINEAR);
    cvtColor(resize_image, resize_image, COLOR_BGR2RGB);
    resize_image.convertTo(resize_image, CV_32FC3, 1/255.0);
    torch::Tensor tensor_image = torch::from_blob(
        resize_image.data,
        {1, resize_image.rows, resize_image.cols, 3},
        c10::kFloat
    );
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    tensor_image = torch::data::transforms::Normalize<>(norm_mean, norm_std)(tensor_image);
    return tensor_image;  
}

torch::jit::script::Module loadModel(string model_path){
    
    torch::jit::script::Module model;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        abort();
    }
    model.forward({torch::randn({1, 3, 224, 224})});
    return model;
}

torch::Tensor inference(torch::jit::script::Module model, torch::Tensor image){

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones());
    inputs.push_back(image);

    // Execute the model and turn its output into a tensor.
    torch::Tensor output = torch::softmax(model.forward(inputs).toTensor(),1);
    return output;
}

void getResult(torch::Tensor output, std::vector<std::string> labelList){

    tuple<torch::Tensor, torch::Tensor> result = torch::max(output, 1);
    torch::Tensor prob = get<0>(result);
    torch::Tensor index = get<1>(result);

    cout << "Predicted class = " << labelList[index[0].item<int>()] << " | " << "confidence = " << prob[0].item<float>()<< "\n";

}

int main(int argc, const char* argv[]) {

    // Argparser
    argparse::ArgumentParser program("libtorch_inference");
    getInput(program, argc, argv);
    
    auto model_path = program.get<string>("--model_path");
    auto image_path = program.get<string>("--image_path");

    // Model
    torch::jit::script::Module model = loadModel(model_path);
    cout << "Finish model loading ... \n";

    // Image
    torch::Tensor image = processImage(image_path);
    cout << "Finish image processing ... \n";
   
    // Inference
    torch::Tensor output = inference(model, image);

    // get_result
    std::vector<std::string> labelList = {"Ramen", "Sashimi", "Sushi", "Takoyaki"};
    cout << "---------------------------------------\n";
    getResult(output, labelList);
    
}
