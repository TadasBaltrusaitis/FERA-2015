#ifndef __FACEANALYSER_h_
#define __FACEANALYSER_h_

#include "SVR_dynamic_lin_regressors.h"
#include "SVR_static_lin_regressors.h"
#include "SVM_static_lin.h"
#include "SVM_dynamic_lin.h"

#include <string>
#include <vector>

#include <cv.h>

#include "CLM.h"

namespace Psyche
{

class FaceAnalyser{

public:


	enum RegressorType{ SVR_appearance_static_linear = 0, SVR_appearance_dynamic_linear = 1, SVR_dynamic_geom_linear = 2, SVR_combined_linear = 3, SVM_linear_stat = 4, SVM_linear_dyn = 5, SVR_linear_static_seg = 6, SVR_linear_dynamic_seg =7};

	// Constructor from a model file (or a default one if not provided
	// TODO scale width and height should be part of the model?
	FaceAnalyser(vector<Vec3d> orientation_bins = vector<Vec3d>(), double scale = 0.7, int width = 112, int height = 112, std::string au_location = "AU_regressors/AU_regressors.txt", std::string av_location = "AV_regressors/AV_regressors.txt", std::string tri_location = "model/tris_68_full.txt");

	void AddNextFrame(const cv::Mat& frame, const CLMTracker::CLM& clm, double timestamp_seconds, bool visualise = true);

	// If the features are extracted manually
	void PredictAUs(const cv::Mat_<double>& hog_features, const cv::Mat_<double>& geom_features, const CLMTracker::CLM& clm_model);

	Mat GetLatestHOGDescriptorVisualisation();

	double GetCurrentTimeSeconds();

	double GetCurrentArousal();

	double GetCurrentValence();

	// Use basic emotion rules for inferring them from AUs
	std::string GetCurrentCategoricalEmotion();

	std::vector<std::pair<std::string, double>> GetCurrentAUsClass();
	std::vector<std::pair<std::string, double>> GetCurrentAUsReg();
	std::vector<std::pair<std::string, double>> GetCurrentAUsRegSegmented();

	std::vector<std::pair<std::string, double>> GetCurrentAUsCombined();

	void Reset();

	void ResetAV();

	double GetConfidence()
	{
		double confidence = frames_tracking / (double)frames_for_adaptation;

		if(confidence > 1)
		{
			confidence = 1;
		}

		return confidence;

	}

	void GetLatestHOG(Mat_<double>& hog_descriptor, int& num_rows, int& num_cols);
	void GetLatestAlignedFace(Mat& image);
	
	void GetLatestNeutralHOG(Mat_<double>& hog_descriptor, int& num_rows, int& num_cols);
	void GetLatestNeutralFace(Mat& image);
	
	Mat_<int> FaceAnalyser::GetTriangulation();

	Mat_<uchar> GetLatestAlignedFaceGrayscale();
	
	void GetGeomDescriptor(Mat_<double>& geom_desc);

	void ExtractCurrentMedians(vector<Mat>& hog_medians, vector<Mat>& face_image_medians, vector<Vec3d>& orientations);



private:

	// Where the predictions are kept
	std::vector<std::pair<std::string, double>> AU_predictions_reg;
	std::vector<std::pair<std::string, double>> AU_predictions_reg_segmented;
	std::vector<std::pair<std::string, double>> AU_predictions_class;

	std::vector<std::pair<std::string, double>> AU_predictions_combined;

	double arousal_value;
	double valence_value;
	int frames_tracking;

	// Cache of intermediate images
	Mat_<uchar> aligned_face_grayscale;
	Mat aligned_face;
	Mat hog_descriptor_visualisation;

	// Private members to be used for predictions
	// The HOG descriptor of the last frame
	Mat_<double> hog_desc_frame;
	int num_hog_rows;
	int num_hog_cols;

	// Keep a running median of the hog descriptors and a aligned images
	Mat_<double> hog_desc_median;
	Mat_<double> face_image_median;

	// Use histograms for quick (but approximate) median computation
	// Use the same for
	vector<Mat_<unsigned int> > hog_desc_hist;

	// TODO populate this
	vector<Mat_<unsigned int> > face_image_hist;
	vector<int> face_image_hist_sum;

	vector<Vec3d> head_orientations;

	int num_bins_hog;
	double min_val_hog;
	double max_val_hog;
	vector<int> hog_hist_sum;
	int view_used;

	// The geometry descriptor (rigid followed by non-rigid shape parameters from CLM)
	Mat_<double> geom_descriptor_frame;
	Mat_<double> geom_descriptor_median;
	
	int geom_hist_sum;
	Mat_<unsigned int> geom_desc_hist;
	int num_bins_geom;
	double min_val_geom;
	double max_val_geom;
	
	int frames_for_adaptation;

	// Using the bounding box of previous analysed frame to determine if a reset is needed
	Rect_<double> face_bounding_box;
	
	// The AU predictions internally
	std::vector<std::pair<std::string, double>> PredictCurrentAUs(int view, bool dyn_correct = false);
	std::vector<std::pair<std::string, double>> PredictCurrentAUsSegmented(int view, bool dyn_correct = false);
	std::vector<std::pair<std::string, double>> PredictCurrentAUsClass(int view);

	void PredictCurrentAVs(const CLMTracker::CLM& clm);

	void ReadAU(std::string au_location);
	void ReadAV(std::string av_location);

	void ReadRegressor(std::string fname, const vector<string>& au_names);

	// A utility function for keeping track of approximate running medians used for AU and emotion inference using a set of histograms (the histograms are evenly spaced from min_val to max_val)
	// Descriptor has to be a row vector
	// TODO this duplicates some other code
	void UpdateRunningMedian(cv::Mat_<unsigned int>& histogram, int& hist_sum, cv::Mat_<double>& median, const cv::Mat_<double>& descriptor, bool update, int num_bins, double min_val, double max_val);
	void ExtractMedian(cv::Mat_<unsigned int>& histogram, int hist_count, cv::Mat_<double>& median, int num_bins, double min_val, double max_val);
	
	// The linear SVR regressors
	SVR_static_lin_regressors AU_SVR_static_appearance_lin_regressors;
	SVR_dynamic_lin_regressors AU_SVR_dynamic_appearance_lin_regressors;
		
	// The linear SVR regressors for segmented use
	SVR_static_lin_regressors AU_SVR_static_appearance_lin_regressors_seg;
	SVR_dynamic_lin_regressors AU_SVR_dynamic_appearance_lin_regressors_seg;
		
	// TODO rem
	//SVR_dynamic_lin_regressors_scale arousal_predictor_lin_geom;
	//SVR_dynamic_lin_regressors_scale valence_predictor_lin_geom;

	SVM_static_lin AU_SVM_static_appearance_lin;
	SVM_dynamic_lin AU_SVM_dynamic_appearance_lin;

	// The AUs (and AV) predicted by the model are not always 0 calibrated to a person. That is they don't always predict 0 for a neutral expression
	// Keeping track of the predictions we can correct for this, by assuming that at least "ratio" of frames are neutral and subtract that value of prediction, only perform the correction after min_frames
	void UpdatePredictionTrack(Mat_<unsigned int>& prediction_corr_histogram, int& prediction_correction_count, vector<double>& correction, const vector<pair<string, double>>& predictions, double ratio=0.25, int num_bins = 200, double min_val = 0, double max_val = 5, int min_frames = 10);	
	void GetSampleHist(Mat_<unsigned int>& prediction_corr_histogram, int prediction_correction_count, vector<double>& sample, double ratio, int num_bins = 200, double min_val = 0, double max_val = 5);	

	vector<cv::Mat_<unsigned int>> au_prediction_correction_histogram;
	vector<int> au_prediction_correction_count;

	cv::Mat_<unsigned int> av_prediction_correction_histogram;
	int av_prediction_correction_count;

	// Some dynamic scaling (the logic is that before the extreme versions of expression or emotion are shown,
	// it is hard to tell the boundaries, this allows us to scale the model to the most extreme seen)
	// They have to be view specific
	vector<vector<double>> dyn_scaling;
	
	// Keeping track of predictions for summary stats
	cv::Mat_<double> AU_prediction_track;
	cv::Mat_<double> geom_desc_track;

	// Keep track of the current time
	double current_time_seconds;

	// Used for face alignment
	Mat_<int> triangulation;
	double align_scale;	
	int align_width;
	int align_height;
};
  //===========================================================================
}
#endif

