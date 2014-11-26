#ifndef __SVRDYNAMICLINREGRESSORSSCALE_h_
#define __SVRDYNAMICLINREGRESSORSSCALE_h_

#include <vector>
#include <string>

#include <stdio.h>
#include <iostream>

#include <cv.h>

namespace Psyche
{

// Collection of linear SVR regressors for AU prediction that uses per person face nomalisation with the help of a running median
class SVR_dynamic_lin_regressors_scale{

public:

	SVR_dynamic_lin_regressors_scale()
	{}

	// Predict the AU from HOG appearance of the face
	void Predict(std::vector<double>& predictions, std::vector<std::string>& names, const cv::Mat_<double>& descriptor);

	// Reading in the model (or adding to it)
	void Read(std::ifstream& stream, const std::vector<std::string>& au_names);

	// For normalisation (should be done before prediction, hence they are public)
	cv::Mat_<double> means;
	cv::Mat_<double> scaling;

private:

	// The names of Action Units this model is responsible for
	std::vector<std::string> AU_names;

	
	// For actual prediction
	cv::Mat_<double> support_vectors;	
	cv::Mat_<double> biases;

};
  //===========================================================================
}
#endif

