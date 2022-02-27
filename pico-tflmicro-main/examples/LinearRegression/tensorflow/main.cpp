
// https://www.electronicshub.org/raspberry-pi-pico-serial-programming/

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"
#include "pico/stdlib.h"
#include <iostream>

// our model
#include "model_data.h"

#define DEBUG 1

namespace{
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* model_input = nullptr;
    TfLiteTensor* model_output = nullptr;

    // create a area in memory which will be occupied by input and output tensors
    constexpr int kTensorArenaSize = 5 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
}

void setup(){
    
    // create the error reporter
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // getting the tf-model
    model = tflite::GetModel(linearRegression_tflite);

    if(model->version() != TFLITE_SCHEMA_VERSION){
        error_reporter->Report("model version does not support");
    }

    // creating the required operations for the neural network
    static tflite::MicroMutableOpResolver<1> micro_mutable_op_resolver;
    micro_mutable_op_resolver.AddFullyConnected();

    // create the interpreter
    static tflite::MicroInterpreter static_interpreter(
        model,
        micro_mutable_op_resolver,
        tensor_arena,
        kTensorArenaSize,
        error_reporter
    );

    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();

    if(allocate_status != kTfLiteOk){
        error_reporter->Report("error in allocating the tensor");
    }

    model_input = interpreter->input(0);
    model_output = interpreter->output(0);
    
    #if DEBUG

        std::cout << "Number of Dimensions \n";
        std::cout << model_input->dims->size << std::endl;
    
        std::cout << "Dim 1 \n";
        std::cout << model_input->dims->data[0] << std::endl;
        
        std::cout << "Dim 2 \n";
        std::cout << model_input->dims->data[1] << std::endl;

        std::cout << "DataType \n";
        std::cout << model_input->type << std::endl;
    
    #endif

}

void prediction(){

    float x_val = 134;

    model_input->data.f[0] = x_val;

    TfLiteStatus invoke_status = interpreter->Invoke();
    if(invoke_status != kTfLiteOk){
        error_reporter->Report("Error in invoking the model");
    }

    float y_val = model_output->data.f[0];
    std::cout << "Prediction :: " << y_val << std::endl;

}

int main(){

    stdio_init_all();
    setup();

    while(true){
        
        prediction();

       sleep_ms(1000);
    }
  

    return 0;
}



