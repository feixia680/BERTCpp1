#include "quantization.h"

namespace lh{
    //Observer 的类，用于观察和跟踪数据集中值的范围
    Observer::Observer(float average_constant){
        min_val_ = NAN;
        max_val_ = NAN;

        average_constant_ = average_constant;
        //平滑常数，用于更新最小值和最大值。该值决定了新观测值对当前记录值的影响程度，更高的值意味着新数据对现有极值的影响更大
    }

    Observer::~Observer(){

    };

    void Observer::find_min_max(float* data, std::size_t size, float& min_input, float& max_input){
        max_input = *std::max_element(data, data + size);
        min_input = *std::min_element(data, data + size);
    }

    void Observer::update_min_max(float min_current, float max_current){
        if(std::isnan(min_val_)) min_val_ = min_current;
        else min_val_ = min_val_ + average_constant_ * (min_current - min_val_);
        if(std::isnan(max_val_)) max_val_ = max_current;
        else max_val_ = max_val_ + average_constant_ * (max_current - max_val_);
    }

    void Observer::compute(float* data, std::size_t size){
        float min_current, max_current;
        find_min_max(data, size, min_current, max_current);
        update_min_max(min_current, max_current);
    }



}