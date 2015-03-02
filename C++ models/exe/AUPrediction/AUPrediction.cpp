///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2012, Tadas Baltrusaitis, all rights reserved.
//
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
//
//     * The software is provided under the terms of this licence stricly for
//       academic, non-commercial, not-for-profit purposes.
//     * Redistributions of source code must retain the above copyright notice, 
//       this list of conditions (licence) and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright 
//       notice, this list of conditions (licence) and the following disclaimer 
//       in the documentation and/or other materials provided with the 
//       distribution.
//     * The name of the author may not be used to endorse or promote products 
//       derived from this software without specific prior written permission.
//     * As this software depends on other libraries, the user must adhere to 
//       and keep in place any licencing terms of those libraries.
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite one of the following works:
//
//       Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. 3D
//       Constrained Local Model for Rigid and Non-Rigid Facial Tracking.
//       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.    
//
//       Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. 
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////


// SimpleCLM.cpp : Defines the entry point for the console application.

#include <CLM.h>
#include <CLMTracker.h>
#include <CLMParameters.h>
#include <CLM_utils.h>

#include <fstream>
#include <sstream>

#include <cv.h>

#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#include <FaceAnalyser.h>
#include <Face_utils.h>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;
using namespace cv;

using namespace boost::filesystem;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

// Extracting the following command line arguments -f, -fd, -op, -of, -ov (and possible ordered repetitions)
void get_output_feature_params(string& au_location, vector<string> &output_aus_class, vector<string> &output_aus_reg, vector<string> &output_aus_segmented, double &similarity_scale, int &similarity_size, double &scaling, bool &video, bool &grayscale, bool &rigid, vector<int>& beg_frames, vector<int>& end_frames, vector<string> &arguments)
{

	bool* valid = new bool[arguments.size()];
	video = false;

	for(size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
	}

	string input_root = "";
	string output_root = "";

	// First check if there is a root argument (so that videos and outputs could be defined more easilly)
	for(size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-root") == 0) 
		{                    
			input_root = arguments[i + 1];
			output_root = arguments[i + 1];
			i++;
		}
		if (arguments[i].compare("-inroot") == 0) 
		{                    
			input_root = arguments[i + 1];
			i++;
		}
		if (arguments[i].compare("-outroot") == 0) 
		{                    
			output_root = arguments[i + 1];
			i++;
		}
	}

	for(size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-auloc") == 0) 
		{                    
			au_location = arguments[i + 1];
			valid[i] = false;
		}	
		else if(arguments[i].compare("-oausclass") == 0) 
		{
			output_aus_class.push_back(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		else if(arguments[i].compare("-oausreg") == 0) 
		{
			output_aus_reg.push_back(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		else if(arguments[i].compare("-oausregseg") == 0) 
		{
			output_aus_segmented.push_back(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		else if(arguments[i].compare("-ef") == 0) 
		{
			end_frames.push_back(stoi(arguments[i + 1]));
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		else if(arguments[i].compare("-bf") == 0) 
		{
			beg_frames.push_back(stoi(arguments[i + 1]));
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		else if(arguments[i].compare("-rigid") == 0) 
		{
			rigid = true;
		}
		else if(arguments[i].compare("-vid") == 0) 
		{
			video = true;
			valid[i] = false;
		}
		else if(arguments[i].compare("-g") == 0) 
		{
			grayscale = true;
			valid[i] = false;
		}
		else if(arguments[i].compare("-scaling") == 0) 
		{
			scaling = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}
		else if (arguments[i].compare("-simscale") == 0) 
		{                    
			similarity_scale = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}		
		else if (arguments[i].compare("-simsize") == 0) 
		{                    
			similarity_size = stoi(arguments[i + 1]);
			valid[i] = false;
			valid[i+1] = false;			
			i++;
		}		
		else if (arguments[i].compare("-help") == 0)
		{
			cout << "Output features are defined as: -simalign <outputfile>\n"; // Inform the user of how to use the program				
		}
	}

	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}

}

// Can process images via directories creating a separate output file per directory
void get_image_input_output_params_feats(vector<vector<string> > &input_image_files, bool& as_video, vector<string> &arguments)
{
	bool* valid = new bool[arguments.size()];
		
	for(size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-fdir") == 0) 
		{                    

			// parse the -fdir directory by reading in all of the .png and .jpg files in it
			path image_directory (arguments[i+1]); 

			try
			{
				 // does the file exist and is it a directory
				if (exists(image_directory) && is_directory(image_directory))   
				{
					
					vector<path> file_in_directory;                                
					copy(directory_iterator(image_directory), directory_iterator(), back_inserter(file_in_directory));

					vector<string> curr_dir_files;

					for (vector<path>::const_iterator file_iterator (file_in_directory.begin()); file_iterator != file_in_directory.end(); ++file_iterator)
					{
						// Possible image extension .jpg and .png
						if(file_iterator->extension().string().compare(".jpg") == 0 || file_iterator->extension().string().compare(".png") == 0)
						{																
							curr_dir_files.push_back(file_iterator->string());															
						}
					}

					input_image_files.push_back(curr_dir_files);
				}
			}
			catch (const filesystem_error& ex)
			{
				cout << ex.what() << '\n';
			}

			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}
		else if (arguments[i].compare("-ftxt") == 0) 
		{                    
			// parse the -fdir directory by reading in all of the .png and .jpg files in it
			std::ifstream txt_reader(arguments[i+1]);

			std::string curr_line;

			std::getline(txt_reader, curr_line);
			vector<string> curr_dir_files;
			string root("");
			if(!boost::filesystem::exists(curr_line))
			{
				string root = path(arguments[0]).parent_path().string();
				string attempt_2 = (path(root) / path(curr_line)).string();
				if(!boost::filesystem::exists(attempt_2))
				{
					cout << "Can't find specified file in the text file either:" << curr_line << "or:" << attempt_2 << std::endl;				
				}
			}
			
			curr_dir_files.push_back((path(root) / path(curr_line)).string());		

			while(std::getline(txt_reader, curr_line))
			{					

				curr_dir_files.push_back((path(root) / path(curr_line)).string());								
			}

			txt_reader.close();

			input_image_files.push_back(curr_dir_files);

			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}
		else if (arguments[i].compare("-asvid") == 0) 
		{
			as_video = true;
		}
		else if (arguments[i].compare("-help") == 0)
		{
			cout << "Input output files are defined as: -fdir <image directory (can have multiple ones)> -asvid <the images in a folder are assumed to come from a video (consecutive)>" << endl; // Inform the user of how to use the program				
		}
	}
	
	// Clear up the argument list
	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}

}

int main (int argc, char **argv)
{
	
	boost::filesystem::path root(argv[0]);
	root = root.parent_path();

	vector<string> arguments = get_arguments(argc, argv);

	// Some initial parameters that can be overriden from command line	
	vector<string> files;

	// Unused elements
	vector<string> depth_directories, pose_output_files, tracked_videos_output, landmark_output_files;
	// By default try webcam 0
	int device = 0;

	// cx and cy aren't necessarilly in the image center, so need to be able to override it (start with unit vals and init them if none specified)
    float fx = 500, fy = 500, cx = 0, cy = 0;
			
	CLMTracker::CLMParameters clm_parameters(arguments);
			
	// Get the input output file parameters
	
	// Indicates that rotation should be with respect to camera plane or with respect to camera
	bool use_camera_plane_pose;
	CLMTracker::get_video_input_output_params(files, depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, use_camera_plane_pose, arguments);

	bool video = true;
	bool images_as_video = false;

	vector<vector<string> > input_image_files;

	// Adding image support for reading in the files
	if(files.empty())
	{
		vector<string> d_files;
		vector<string> o_img;
		vector<Rect_<double>> bboxes;
		get_image_input_output_params_feats(input_image_files, images_as_video, arguments);	

		if(!input_image_files.empty())
		{
			video = false;
		}

	}
	// Get camera parameters
	CLMTracker::get_camera_params(device, fx, fy, cx, cy, arguments);    
	
	if(!boost::filesystem::exists(path(clm_parameters.model_location)))
	{
		clm_parameters.model_location = (root / path(clm_parameters.model_location)).string();
	}

	cout << clm_parameters.model_location << endl;

	// The modules that are being used for tracking
	CLMTracker::CLM clm_model(clm_parameters.model_location);	

	vector<string> output_aus_class;
	vector<string> output_aus_reg;
	vector<string> output_aus_reg_segmented;

	double sim_scale = 0.7;
	int sim_size = 112;
	bool video_output;
	bool grayscale = false;
	bool rigid = true;	
	int num_hog_rows;
	int num_hog_cols;

	double scaling = 1.0;

	string face_analyser_loc("./AU_predictors/AU_SVM_BP4D_static.txt");
	string face_analyser_loc_av("./AV_regressors/av_regressors.txt");
	string tri_location("./model/tris_68_full.txt");

	vector<int> beg_frames;
	vector<int> end_frames;

	get_output_feature_params(face_analyser_loc, output_aus_class, output_aus_reg, output_aus_reg_segmented, sim_scale, sim_size, scaling, video_output, grayscale, rigid, beg_frames, end_frames, arguments);

	if(!boost::filesystem::exists(path(face_analyser_loc)))
	{
		face_analyser_loc = (root / path(face_analyser_loc)).string();
		face_analyser_loc_av = (root / path(face_analyser_loc_av)).string();
		tri_location = (root / path(tri_location)).string();
	}

	// Face analyser (used for neutral expression extraction)
	vector<Vec3d> orientations = vector<Vec3d>();
	orientations.push_back(Vec3d(0.0,0.0,0.0));
	Psyche::FaceAnalyser face_analyser(orientations, sim_scale, sim_size, sim_size, face_analyser_loc, face_analyser_loc_av, tri_location);

	// Will warp to scaled mean shape
	Mat_<double> similarity_normalised_shape = clm_model.pdm.mean_shape * sim_scale;
	// Discard the z component
	similarity_normalised_shape = similarity_normalised_shape(Rect(0, 0, 1, 2*similarity_normalised_shape.rows/3)).clone();

	// If multiple video files are tracked, use this to indicate if we are done
	bool done = false;	
	int f_n = -1;
	int curr_img = -1;

	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	bool cx_undefined = false;
	if(cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}			

	// Retain a matrix of HOG descriptors for AU prediction? TODO
			
	// This is useful for a second pass run (if want AU predictions)
	vector<vector<Vec6d>> params_global_video;
	vector<vector<bool>> successes_video;
	vector<vector<Mat_<double>>> params_local_video;
	vector<vector<Mat_<double>>> detected_landmarks_video;
	
	// TODO this might be done with a matrix
	vector<vector<Mat_<double>>> hog_descriptors;
	vector<vector<Mat_<double>>> geom_descriptors;

	while(!done) // this is not a for loop as we might also be reading from a webcam
	{
		
		string current_file;
		
		VideoCapture video_capture;
		
		Mat captured_image;

		params_global_video.push_back(vector<Vec6d>());
		successes_video.push_back(vector<bool>());
		params_local_video.push_back(vector<Mat_<double>>());
		detected_landmarks_video.push_back(vector<Mat_<double>>());
	
		hog_descriptors.push_back(vector<Mat_<double>>());
		geom_descriptors.push_back(vector<Mat_<double>>());

		if(video)
		{
			// We might specify multiple video files as arguments
			if(files.size() > 0)
			{
				f_n++;			
				current_file = files[f_n];
			}
			else
			{
				// If we want to write out from webcam
				f_n = 0;
			}
			// Do some grabbing
			if( current_file.size() > 0 )
			{
				INFO_STREAM( "Attempting to read from file: " << current_file );
				video_capture = VideoCapture( current_file );
			}
			else
			{
				INFO_STREAM( "Attempting to capture from device: " << device );
				video_capture = VideoCapture( device );

				// Read a first frame often empty in camera
				Mat captured_image;
				video_capture >> captured_image;
			}

			if( !video_capture.isOpened() ) FATAL_STREAM( "Failed to open video source" );
			else INFO_STREAM( "Device or file opened");

			video_capture >> captured_image;	
		}
		else
		{
			f_n++;	
			curr_img++;
			if(!input_image_files[f_n].empty())
			{
				string curr_img_file = input_image_files[f_n][curr_img];
				captured_image = imread(curr_img_file, -1);
			}
			else
			{
				FATAL_STREAM( "No .jpg or .png images in a specified drectory" );
			}

		}			

		// If optical centers are not defined just use center of image
		if(cx_undefined)
		{
			cx = captured_image.cols / 2.0f;
			cy = captured_image.rows / 2.0f;
		}
	
		int frame_count = 0;
		
		// For measuring the timings
		int64 t1,t0 = cv::getTickCount();
		double fps = 10;		

		INFO_STREAM( "Starting tracking");
		while(!captured_image.empty())
		{		
			if(beg_frames.empty() || frame_count >= beg_frames[f_n])
			{				
				if(scaling != 1.0)
				{
					cv::resize(captured_image, captured_image, Size(), scaling, scaling);
				}

				// Reading the images
				Mat_<uchar> grayscale_image;

				if(captured_image.channels() == 3)
				{
					cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);				
				}
				else
				{
					grayscale_image = captured_image.clone();				
				}
		
				// The actual facial landmark detection / tracking
				bool detection_success;
			
				if(video || images_as_video)
				{
					detection_success = CLMTracker::DetectLandmarksInVideo(grayscale_image, clm_model, clm_parameters);
				}
				else
				{
					detection_success = CLMTracker::DetectLandmarksInImage(grayscale_image, clm_model, clm_parameters);
				}
			
				// Do face alignment
				Mat sim_warped_img;			
				Mat_<double> hog_descriptor;

				// Use face analyser only if outputting neutrals and AUs
				if(!output_aus_class.empty() || !output_aus_reg.empty() || !output_aus_reg_segmented.empty())
				{
					face_analyser.AddNextFrame(captured_image, clm_model, 0, false);

					params_global_video[f_n].push_back(clm_model.params_global);
					params_local_video[f_n].push_back(clm_model.params_local.clone());
					successes_video[f_n].push_back(detection_success);
					detected_landmarks_video[f_n].push_back(clm_model.detected_landmarks.clone());
				
					face_analyser.GetLatestAlignedFace(sim_warped_img);
					face_analyser.GetLatestHOG(hog_descriptor, num_hog_rows, num_hog_cols);
					hog_descriptors[f_n].push_back(hog_descriptor.clone());
					
					Mat_<double> geom_desc;
					face_analyser.GetGeomDescriptor(geom_desc);
					geom_descriptors[f_n].push_back(geom_desc);
				}
				else
				{
					Psyche::AlignFaceMask(sim_warped_img, captured_image, clm_model, face_analyser.GetTriangulation(), rigid, sim_scale, sim_size, sim_size);
					Psyche::Extract_FHOG_descriptor(hog_descriptor, sim_warped_img, num_hog_rows, num_hog_cols);			
				}

				//cv::imshow("sim_warp", sim_warped_img);			
			
				//Mat_<double> hog_descriptor_vis;
				//Psyche::Visualise_FHOG(hog_descriptor, num_hog_rows, num_hog_cols, hog_descriptor_vis);
				//cv::imshow("hog", hog_descriptor_vis);	

				// Work out the pose of the head from the tracked model
				Vec6d pose_estimate_CLM;
				if(use_camera_plane_pose)
				{
					pose_estimate_CLM = CLMTracker::GetCorrectedPoseCameraPlane(clm_model, fx, fy, cx, cy, clm_parameters);
				}
				else
				{
					pose_estimate_CLM = CLMTracker::GetCorrectedPoseCamera(clm_model, fx, fy, cx, cy, clm_parameters);
				}

				// Visualising the results
				// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
				double detection_certainty = clm_model.detection_certainty;

				double visualisation_boundary = 0.2;
			
				// Only draw if the reliability is reasonable, the value is slightly ad-hoc
				if(detection_certainty < visualisation_boundary)
				{
					CLMTracker::Draw(captured_image, clm_model);
					//CLMTracker::Draw(captured_image, clm_model);

					if(detection_certainty > 1)
						detection_certainty = 1;
					if(detection_certainty < -1)
						detection_certainty = -1;

					detection_certainty = (detection_certainty + 1)/(visualisation_boundary +1);

					// A rough heuristic for box around the face width
					int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);
				
					Vec6d pose_estimate_to_draw = CLMTracker::GetCorrectedPoseCameraPlane(clm_model, fx, fy, cx, cy, clm_parameters);

					// Draw it in reddish if uncertain, blueish if certain
					CLMTracker::DrawBox(captured_image, pose_estimate_to_draw, Scalar((1-detection_certainty)*255.0,0, detection_certainty*255), thickness, fx, fy, cx, cy);

				}
			
				// Work out the framerate
				if(frame_count % 10 == 0)
				{      
					t1 = cv::getTickCount();
					fps = 10.0 / (double(t1-t0)/cv::getTickFrequency()); 
					t0 = t1;
				}
			
				// Write out the framerate on the image before displaying it
				char fpsC[255];
				sprintf(fpsC, "%d", (int)fps);
				string fpsSt("FPS:");
				fpsSt += fpsC;
				cv::putText(captured_image, fpsSt, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0));		
			
				if(!clm_parameters.quiet_mode)
				{
					namedWindow("tracking_result",1);		
					imshow("tracking_result", captured_image);
				}
			}
			if(video)
			{
				video_capture >> captured_image;
			}
			else
			{
				curr_img++;
				if(curr_img < (int)input_image_files[f_n].size())
				{
					string curr_img_file = input_image_files[f_n][curr_img];
					captured_image = imread(curr_img_file, -1);
				}
				else
				{
					captured_image = Mat();
				}
			}
			// detect key presses
			char character_press = cv::waitKey(1);
			
			// restart the tracker
			if(character_press=='q')
			{
				return(0);
			}

			// Update the frame count
			frame_count++;

			if(!end_frames.empty())
			{
				if(frame_count > end_frames[f_n])
				{
					break;
				}
			}

		}
		
		
		frame_count = 0;
		curr_img = -1;

		// Reset the model, for the next video
		clm_model.Reset();
		
		if(video)
		{
			// break out of the loop if done with all the files (or using a webcam)
			if(f_n == files.size() -1 || files.empty())
			{
				done = true;
			}
		}
		else
		{
			// break out of the loop if done with all the files (or using a webcam)
			if(f_n == input_image_files.size() -1 || input_image_files.empty())
			{
				done = true;
			}
		}
	}

	int num_outputs = 0;

	if(!output_aus_class.empty())	
		num_outputs = output_aus_class.size();

	if(!output_aus_reg.empty())	
		num_outputs = output_aus_reg.size();

	if(!output_aus_reg_segmented.empty())	
		num_outputs = output_aus_reg_segmented.size();


	for(int i = 0; i < num_outputs; ++i) // this is not a for loop as we might also be reading from a webcam
	{

		// Collect all of the predictions
		vector<vector<double>> all_predictions_class;
		vector<vector<double>> all_predictions_reg;
		vector<vector<double>> all_predictions_reg_segmented;

		vector<string> pred_names_class;
		vector<string> pred_names_reg;
		vector<string> pred_names_reg_segmented;
		
		for(size_t frame = 0; frame < params_global_video[i].size(); ++frame)
		{
		
			clm_model.detected_landmarks = detected_landmarks_video[i][frame].clone();
			clm_model.params_local = params_local_video[i][frame].clone();
			clm_model.params_global = params_global_video[i][frame];
			clm_model.detection_success = successes_video[i][frame];
				
			face_analyser.PredictAUs(hog_descriptors[i][frame], geom_descriptors[i][frame], clm_model);

			auto au_preds_class = face_analyser.GetCurrentAUsClass();
			auto au_preds_reg = face_analyser.GetCurrentAUsReg();
			auto au_preds_reg_segmented = face_analyser.GetCurrentAUsRegSegmented();

			if(frame == 0)
			{
				all_predictions_class.resize(au_preds_class.size());
				for(int au = 0; au < au_preds_class.size(); ++au)
				{
					pred_names_class.push_back(au_preds_class[au].first);
				}

				all_predictions_reg.resize(au_preds_reg.size());
				for(int au = 0; au < au_preds_reg.size(); ++au)
				{
					pred_names_reg.push_back(au_preds_reg[au].first);
				}

				all_predictions_reg_segmented.resize(au_preds_reg_segmented.size());
				for(int au = 0; au < au_preds_reg_segmented.size(); ++au)
				{
					pred_names_reg_segmented.push_back(au_preds_reg_segmented[au].first);
				}
			}

			for(int au = 0; au < au_preds_class.size(); ++au)
			{				
				all_predictions_class[au].push_back(au_preds_class[au].second);
			}

			for(int au = 0; au < au_preds_reg.size(); ++au)
			{				
				all_predictions_reg[au].push_back(au_preds_reg[au].second);
			}

			for(int au = 0; au < au_preds_reg_segmented.size(); ++au)
			{				
				all_predictions_reg_segmented[au].push_back(au_preds_reg_segmented[au].second);
			}

		}		
				
		int window = 7;
		int sub_window = 3;
		// Some running average smoothing of classes

		for(size_t frame = 0; frame < sub_window || frame < params_global_video[i].size() - sub_window; ++frame)
		{
			for(int au = 0; au < pred_names_reg.size(); ++au)
			{			
				if(all_predictions_reg[au][frame] < 0.01)
					all_predictions_reg[au][frame] = 0;
				
				if(all_predictions_reg[au][frame] > 5)
					all_predictions_reg[au][frame] = 5;
			}
			

			for(int au = 0; au < pred_names_reg_segmented.size(); ++au)
			{			
				all_predictions_reg_segmented[au][frame] = all_predictions_reg[au][frame];

				if(all_predictions_reg_segmented[au][frame] < 1)
					all_predictions_reg_segmented[au][frame] = 1;
				
				if(all_predictions_reg_segmented[au][frame] > 5)
					all_predictions_reg_segmented[au][frame] = 5;
			}
		}


		for(size_t frame = params_global_video[i].size() - sub_window; frame < params_global_video[i].size(); ++frame)
		{
			for(int au = 0; au < pred_names_reg.size(); ++au)
			{			
				if(all_predictions_reg[au][frame] < 0.01)
					all_predictions_reg[au][frame] = 0;
				
				if(all_predictions_reg[au][frame] > 5)
					all_predictions_reg[au][frame] = 5;
			}

			for(int au = 0; au < pred_names_reg_segmented.size(); ++au)
			{			
				all_predictions_reg_segmented[au][frame] = all_predictions_reg[au][frame];

				if(all_predictions_reg_segmented[au][frame] < 1)
					all_predictions_reg_segmented[au][frame] = 1;
				
				if(all_predictions_reg_segmented[au][frame] > 5)
					all_predictions_reg_segmented[au][frame] = 5;				
			}
		}

		for(size_t frame = sub_window; frame < params_global_video[i].size() - sub_window; ++frame)
		{
			auto copy_class(all_predictions_class);

			for(int au = 0; au < pred_names_class.size(); ++au)
			{			
				all_predictions_class[au][frame] = 0;
				for(int w = 0; w < window; ++w)
				{
					all_predictions_class[au][frame] += copy_class[au][frame + w - sub_window];
				}
				all_predictions_class[au][frame] = all_predictions_class[au][frame] / window;
				if(all_predictions_class[au][frame] > 0.5)
					all_predictions_class[au][frame] = 1;
				else
					all_predictions_class[au][frame] = 0;

				//if(!successes_video[i][frame])
				//	all_predictions_class[au][frame] = 0;
			}

			auto copy_reg(all_predictions_reg);

			for(int au = 0; au < pred_names_reg.size(); ++au)
			{			
				all_predictions_reg[au][frame] = 0;
				for(int w = 0; w < window; ++w)
				{
					all_predictions_reg[au][frame] += copy_reg[au][frame + w - sub_window];
				}
				all_predictions_reg[au][frame] = all_predictions_reg[au][frame] / window;
				
				if(all_predictions_reg[au][frame] < 0.01)
					all_predictions_reg[au][frame] = 0;
				
				if(all_predictions_reg[au][frame] > 5)
					all_predictions_reg[au][frame] = 5;

				//if(!successes_video[i][frame])
				//	all_predictions_reg[au][frame] = 0;


			}

			auto copy_reg_segmented(all_predictions_reg_segmented);

			for(int au = 0; au < pred_names_reg_segmented.size(); ++au)
			{			
				//all_predictions_reg_segmented[au][frame] = 0;
				//for(int w = 0; w < window; ++w)
				//{
				// all_predictions_reg_segmented[au][frame] += copy_reg_segmented[au][frame + w - sub_window];
				//}
				//all_predictions_reg_segmented[au][frame] = all_predictions_reg_segmented[au][frame] / window;
				
				// Use the regular predictions instead of retraining the models
				all_predictions_reg_segmented[au][frame] = all_predictions_reg[au][frame];

				if(all_predictions_reg_segmented[au][frame] < 1)
					all_predictions_reg_segmented[au][frame] = 1;
				
				if(all_predictions_reg_segmented[au][frame] > 5)
					all_predictions_reg_segmented[au][frame] = 5;

				//if(!successes_video[i][frame])
				//	all_predictions_reg_segmented[au][frame] = 0;
			}
		}

		vector<string> sorted(pred_names_class);
		std::sort(sorted.begin(), sorted.end());

		if(!output_aus_class.empty())	
		{
			std::ofstream au_output_file_class;
			au_output_file_class.open(output_aus_class[i], ios_base::out);

			// Print the results here
			for(int au_out = 0; au_out < pred_names_class.size(); ++au_out)
			{			
				for(int au = 0; au < pred_names_class.size(); ++au)
				{
					if(pred_names_class[au].compare(sorted[au_out]) == 0)
					{
						au_output_file_class << pred_names_class[au];					
						for(int frame = 0; frame < all_predictions_class[au].size(); ++frame)
						{
							au_output_file_class << " " << all_predictions_class[au][frame];
						}
						au_output_file_class << std::endl;			
						break;
					}
				}
			}

			au_output_file_class.close();
		}

		vector<string> sorted_reg(pred_names_reg);
		std::sort(sorted_reg.begin(), sorted_reg.end());

		if(!output_aus_reg.empty())	
		{

			std::ofstream au_output_file_reg;
			au_output_file_reg.open(output_aus_reg[i], ios_base::out);

			// Print the results here
			for(int au_out = 0; au_out < pred_names_reg.size(); ++au_out)
			{			
				for(int au = 0; au < pred_names_reg.size(); ++au)
				{
					if(pred_names_reg[au].compare(sorted_reg[au_out]) == 0)
					{
						au_output_file_reg << pred_names_reg[au];					
						for(int frame = 0; frame < all_predictions_reg[au].size(); ++frame)
						{
							au_output_file_reg << " " << all_predictions_reg[au][frame];
						}
						au_output_file_reg << std::endl;			
						break;
					}
				}
			}

			au_output_file_reg.close();
		}

		vector<string> sorted_reg_segmented(pred_names_reg_segmented);
		std::sort(sorted_reg_segmented.begin(), sorted_reg_segmented.end());

		if(!output_aus_reg_segmented.empty())	
		{
			std::ofstream au_output_file_reg_segmented;
			au_output_file_reg_segmented.open(output_aus_reg_segmented[i], ios_base::out);

			// Print the results here
			for(int au_out = 0; au_out < pred_names_reg_segmented.size(); ++au_out)
			{			
				for(int au = 0; au < pred_names_reg_segmented.size(); ++au)
				{
					if(pred_names_reg_segmented[au].compare(sorted_reg_segmented[au_out]) == 0)
					{
						au_output_file_reg_segmented << pred_names_reg_segmented[au];					
						for(int frame = 0; frame < all_predictions_reg_segmented[au].size(); ++frame)
						{
							au_output_file_reg_segmented << " " << all_predictions_reg_segmented[au][frame];
						}
						au_output_file_reg_segmented << std::endl;			
						break;
					}
				}
			}

			au_output_file_reg_segmented.close();
		}
	}
	return 0;
}

