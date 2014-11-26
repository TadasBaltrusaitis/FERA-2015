#include <Face_utils.h>

#include <CLM_utils.h>

// For FHOG visualisation
#include <dlib/opencv.h>

using namespace cv;
using namespace std;

namespace Psyche
{

	// Pick only the more stable/rigid points under changes of expression
	void extract_rigid_points(Mat_<double>& source_points, Mat_<double>& destination_points)
	{
		if(source_points.rows == 68)
		{
			Mat_<double> tmp_source = source_points.clone();
			source_points = Mat_<double>();

			// Push back the rigid points (some face outline, eyes, and nose)
			source_points.push_back(tmp_source.row(0));
			source_points.push_back(tmp_source.row(2));
			source_points.push_back(tmp_source.row(14));
			source_points.push_back(tmp_source.row(16));
			source_points.push_back(tmp_source.row(36));
			source_points.push_back(tmp_source.row(39));
			source_points.push_back(tmp_source.row(43));
			source_points.push_back(tmp_source.row(38));
			source_points.push_back(tmp_source.row(42));
			source_points.push_back(tmp_source.row(45));
			source_points.push_back(tmp_source.row(31));
			source_points.push_back(tmp_source.row(33));
			source_points.push_back(tmp_source.row(35));

			Mat_<double> tmp_dest = destination_points.clone();
			destination_points = Mat_<double>();

			// Push back the rigid points
			destination_points.push_back(tmp_dest.row(0));
			destination_points.push_back(tmp_dest.row(2));
			destination_points.push_back(tmp_dest.row(14));
			destination_points.push_back(tmp_dest.row(16));
			destination_points.push_back(tmp_dest.row(36));
			destination_points.push_back(tmp_dest.row(39));
			destination_points.push_back(tmp_dest.row(43));
			destination_points.push_back(tmp_dest.row(38));
			destination_points.push_back(tmp_dest.row(42));
			destination_points.push_back(tmp_dest.row(45));
			destination_points.push_back(tmp_dest.row(31));
			destination_points.push_back(tmp_dest.row(33));
			destination_points.push_back(tmp_dest.row(35));
		}
	}

	// Aligning a face to a common reference frame
	void AlignFace(cv::Mat& aligned_face, const cv::Mat& frame, const CLMTracker::CLM& clm_model, double sim_scale, int out_width, int out_height)
	{
		// Will warp to scaled mean shape
		Mat_<double> similarity_normalised_shape = clm_model.pdm.mean_shape * sim_scale;
	
		// Discard the z component
		similarity_normalised_shape = similarity_normalised_shape(Rect(0, 0, 1, 2*similarity_normalised_shape.rows/3)).clone();

		Mat_<double> source_landmarks = clm_model.detected_landmarks.reshape(1, 2).t();
		Mat_<double> destination_landmarks = similarity_normalised_shape.reshape(1, 2).t();

		// Aligning only the more rigid points
		extract_rigid_points(source_landmarks, destination_landmarks);

		Matx22d scale_rot_matrix = CLMTracker::AlignShapesWithScale(source_landmarks, destination_landmarks);
		Matx23d warp_matrix;

		warp_matrix(0,0) = scale_rot_matrix(0,0);
		warp_matrix(0,1) = scale_rot_matrix(0,1);
		warp_matrix(1,0) = scale_rot_matrix(1,0);
		warp_matrix(1,1) = scale_rot_matrix(1,1);

		double tx = clm_model.params_global[4];
		double ty = clm_model.params_global[5];

		Vec2d T(tx, ty);
		T = scale_rot_matrix * T;

		// Make sure centering is correct
		warp_matrix(0,2) = -T(0) + out_width/2;
		warp_matrix(1,2) = -T(1) + out_height/2;

		cv::warpAffine(frame, aligned_face, warp_matrix, Size(out_width, out_height), INTER_LINEAR);
	}

	void Visualise_FHOG(const cv::Mat_<double>& descriptor, int num_rows, int num_cols, cv::Mat& visualisation)
	{

		// First convert to dlib format
		dlib::array2d<dlib::matrix<float,31,1> > hog(num_rows, num_cols);
		
		cv::MatConstIterator_<double> descriptor_it = descriptor.begin();
		for(int y = 0; y < num_cols; ++y)
		{
			for(int x = 0; x < num_rows; ++x)
			{
				for(unsigned int o = 0; o < 31; ++o)
				{
					hog[y][x](o) = *descriptor_it++;
				}
			}
		}

		// Draw the FHOG to OpenCV format
		auto fhog_vis = dlib::draw_fhog(hog);
		visualisation = dlib::toMat(fhog_vis).clone();
	}

	// Create a row vector Felzenszwalb HOG descriptor from a given image
	void Extract_FHOG_descriptor(cv::Mat_<double>& descriptor, const cv::Mat_<uchar>& image, int cell_size)
	{
		dlib::cv_image<uchar> dlib_warped_img(image);

		dlib::array2d<dlib::matrix<float,31,1> > hog;
		dlib::extract_fhog_features(dlib_warped_img, hog, cell_size);

		// Convert to a usable format
		int num_cols = hog.nc();
		int num_rows = hog.nr();

		descriptor = Mat_<double>(1, num_cols * num_rows * 31);
		cv::MatIterator_<double> descriptor_it = descriptor.begin();
		for(int y = 0; y < num_cols; ++y)
		{
			for(int x = 0; x < num_rows; ++x)
			{
				for(unsigned int o = 0; o < 31; ++o)
				{
					*descriptor_it++ = (double)hog[y][x](o);
				}
			}
		}
	}

	// Extract summary statistics (mean, stdev, min, max) from each dimension of a descriptor, each row is a descriptor
	void ExtractSummaryStatistics(const cv::Mat_<double>& descriptors, cv::Mat_<double>& sum_stats, bool use_mean, bool use_stdev, bool use_max_min)
	{
		// Using four summary statistics at the moment 
		// Means, stds, mins, maxs
		int num_stats = 0;

		if(use_mean)
			num_stats++;

		if(use_stdev)
			num_stats++;

		if(use_max_min)
			num_stats++;

		sum_stats = Mat_<double>(1, descriptors.cols * num_stats, 0.0);
		for(int i = 0; i < descriptors.cols; ++i)
		{
			Scalar mean, stdev;
			cv::meanStdDev(descriptors.col(i), mean, stdev);

			int add = 0;

			if(use_mean)
			{
				sum_stats.at<double>(0, i*num_stats + add) = mean[0];
				add++;
			}

			if(use_stdev)
			{
				sum_stats.at<double>(0, i*num_stats + add) = stdev[0];
				add++;
			}

			if(use_max_min)
			{
				double min, max;
				cv::minMaxIdx(descriptors.col(i), &min, &max);
				sum_stats.at<double>(0, i*num_stats + add) = max - min;
				add++;
			}
		}		
	}

	void AddDescriptor(cv::Mat_<double>& descriptors, cv::Mat_<double> new_descriptor, int curr_frame, int num_frames_to_keep)
	{
		if(descriptors.empty())
		{
			descriptors = Mat_<double>(num_frames_to_keep, new_descriptor.cols, 0.0);
		}

		int row_to_change = curr_frame % num_frames_to_keep;

		new_descriptor.copyTo(descriptors.row(row_to_change));
	}	

}