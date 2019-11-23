#include <iostream>
#include "pch.h"
//opencv - https://opencv.org/
#include <opencv2/opencv.hpp>



using namespace std;
using namespace cv; 
using namespace cv::ml;

//dlib - http://dlib.net/
/*#include <dlib/matrix.h>
#include <dlib/dnn.h>
#include <dlib/opencv.h>
using namespace dlib;
*/

struct space
{
	int x01, y01, x02, y02, x03, y03, x04, y04, occup; 
};

int load_parking_geometry(const char *filename, space *spaces);
void extract_space(space *spaces, Mat in_mat, std::vector<Mat> &vector_images);
void draw_detection(space *spaces, Mat &frame);
void evaluation(fstream &detectorOutputFile, fstream &groundTruthFile);

int train_parking();
void test_parking(int treshold);

void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData );

int spaces_num = 56; 
cv::Size space_size(80, 80);

int main(int argc, char** argv)
{      
    cout << "Train OpenCV Start" <<endl;
	int treshold = train_parking();
    cout << "Train OpenCV End" <<endl;

    cout << "Test OpenCV Start" <<endl;
	test_parking(treshold);
    cout << "Test OpenCV End" <<endl;

}

int train_parking()
{
    //load parking lot geometry
    space *spaces = new space[spaces_num];
    load_parking_geometry("../../parking_map.txt", spaces);
    
    std::vector<Mat> train_images;
    std::vector<int> train_labels;

    fstream train_file("train_images.txt"); 
    string train_path;

    while(train_file >> train_path) 
    {
    
        //cout << "train_path: " << train_path << endl;
        Mat frame;
        
        //read training images
        frame = imread(train_path, 0);
        
        // label = 1;//occupied place
        // label = 0;//free place         
        int label = 0;
        if (train_path.find("full") != std::string::npos) label = 1;
        
        //extract each parking space
        extract_space(spaces, frame, train_images);
        
        //training label for each parking space
        for(int i = 0; i < spaces_num; i++)
        {
            train_labels.push_back(label);           
        }

    }

    delete spaces;
    
    cout << "Train images: " << train_images.size() << endl;
    cout << "Train labels: " << train_labels.size() << endl;

	std::vector<int> occupied_count;
	std::vector<int> free_count;
	Mat edge;
	float m[9] = { -1,0,1  -2,0,2, -1,0,1};
	Mat kernel(Size(3, 3), CV_32F, m);

	for (int i = 0; i < train_images.size(); i++)
	{
		filter2D(train_images[i], edge, -1, kernel);
		threshold(edge, edge, 200, 255, CV_THRESH_BINARY);

		int count = cv::countNonZero(edge);
		if (train_labels[i] == 1)
		{
			occupied_count.push_back(count);
		}
		else
		{
			free_count.push_back(count);
		}
	}
	
	sort(occupied_count.begin(), occupied_count.end());
	sort(free_count.begin(), free_count.end());
	int first_occupied = occupied_count.front();
	int last_free = free_count.back();

	int treshold = (first_occupied + last_free) / 2;


	//cv::Mat image_keypoint = cv::Mat(train_images[0]);
	//cv::drawKeypoints(train_images[0], keypoints, image_keypoint, Scalar(255, 255, 255, 0));

	//cv::imshow("keypoints", image_keypoint);
	//cv::waitKey(0);
	return treshold;
}

void test_parking(int treshold)
{

    space *spaces = new space[spaces_num];
    load_parking_geometry("../../parking_map.txt", spaces);
    
    fstream test_file("test_images.txt");
    ofstream out_label_file("out_prediction.txt");
    string test_path;
          
	Mat edge;
	float m[9] = { -1,0,1 - 2,0,2, -1,0,1 };
	Mat kernel(Size(3, 3), CV_32F, m);
    while(test_file >> test_path) 
    {
        //cout << "test_path: " << test_path << endl;
        Mat frame, gradX, gradY;
        //read testing images
        frame = imread( test_path, 1 );
        Mat draw_frame = frame.clone();
        cvtColor( frame, frame, COLOR_BGR2GRAY );  
        
        std::vector<Mat> test_images;              
        extract_space(spaces, frame, test_images);
        
        int colNum = 0;
        for(int i = 0; i < test_images.size(); i++)
        {
			filter2D(test_images[i], edge, -1, kernel);
			threshold(edge, edge, 200, 255, CV_THRESH_BINARY);

			int count = cv::countNonZero(edge);
            int predict_label =  0;
			if (count > treshold)
			{
				predict_label = 1;
			}
            out_label_file << predict_label << endl;
            spaces[i].occup = predict_label;                                    
/*		    imshow("test_img", test_images[i]);  
		    waitKey(0);*/                           
        }
        
        //draw detection
        draw_detection(spaces, draw_frame);
/*        namedWindow("draw_frame", 0);*/
        imshow("draw_frame", draw_frame);
        waitKey(0);  

    }
    
    //evaluation    
	fstream detector_file("out_prediction.txt");
	fstream groundtruth_file("../../groundtruth.txt");
	evaluation(detector_file, groundtruth_file);
}

int load_parking_geometry(const char *filename, space *spaces)
{
    FILE *file = fopen(filename, "r");
	if (file == NULL) return -1;
    int ps_count, i, count;
    count = fscanf(file,"%d\n", &ps_count); // read count of polygons
    for (i=0;i<ps_count;i++){
        int p =0;
        int poly_size;
        count = fscanf(file,"%d->", &poly_size); // read count of polygon vertexes
        int *row = new int[poly_size*2];
        int j;
        for (j=0;j<poly_size;j++){
            int x,y;
            count = fscanf(file,"%d,%d;",&x,&y); // read vertex coordinates
            row[p] = x;
            row[p+1] = y;
            p=p+2;
        }
		 spaces[i].x01 = row[0];
		 spaces[i].y01 = row[1];
		 spaces[i].x02 = row[2];
		 spaces[i].y02 = row[3];
		 spaces[i].x03 = row[4];
		 spaces[i].y03 = row[5];
		 spaces[i].x04 = row[6];
		 spaces[i].y04 = row[7];
        //printf("}\n");
        free(row);
        count = fscanf(file, "\n"); // read end of line
    }
    fclose(file);
	return 1;
}

void extract_space(space *spaces, Mat in_mat, std::vector<Mat> &vector_images)
{
	for (int x = 0; x < spaces_num; x++)
	{
		Mat src_mat(4, 2, CV_32F);
		Mat out_mat(space_size, CV_8U, 1);
		src_mat.at<float>(0, 0) = spaces[x].x01;
		src_mat.at<float>(0, 1) = spaces[x].y01;
		src_mat.at<float>(1, 0) = spaces[x].x02;
		src_mat.at<float>(1, 1) = spaces[x].y02;
		src_mat.at<float>(2, 0) = spaces[x].x03;
		src_mat.at<float>(2, 1) = spaces[x].y03;
		src_mat.at<float>(3, 0) = spaces[x].x04;
		src_mat.at<float>(3, 1) = spaces[x].y04;

		Mat dest_mat(4, 2, CV_32F);
		dest_mat.at<float>(0, 0) = 0;
		dest_mat.at<float>(0, 1) = 0;
		dest_mat.at<float>(1, 0) = out_mat.cols;
		dest_mat.at<float>(1, 1) = 0;
		dest_mat.at<float>(2, 0) = out_mat.cols;
		dest_mat.at<float>(2, 1) = out_mat.rows;
		dest_mat.at<float>(3, 0) = 0;
		dest_mat.at<float>(3, 1) = out_mat.rows;

		Mat H = findHomography(src_mat,dest_mat,0);
		warpPerspective(in_mat, out_mat, H, space_size);
        
		/*imshow("out_mat", out_mat);
        waitKey(0);*/

        vector_images.push_back(out_mat);
		
	}

}

void draw_detection(space *spaces, Mat &frame)
{
    int sx, sy;
    for (int i=0; i< spaces_num; i++)
    {      
        Point pt1, pt2;
        pt1.x = spaces[i].x01;
        pt1.y = spaces[i].y01;
        pt2.x = spaces[i].x03;
        pt2.y = spaces[i].y03;
		sx=(pt1.x+pt2.x)/2;
        sy=(pt1.y+pt2.y)/2;
    	if ( spaces[i].occup )
    	{  	
            circle( frame, Point(sx,sy-25), 12, CV_RGB(255,0,0), -1); 
    	}
    	else
    	{		
            circle( frame, Point(sx,sy-25), 12, CV_RGB(0,255,0), -1);
    	}    	
    }              
}

void evaluation(fstream &detectorOutputFile, fstream &groundTruthFile)
{
	int detectorLine, groundTruthLine;
	int falsePositives = 0;
	int falseNegatives = 0;
	int truePositives = 0;
	int trueNegatives = 0;
	while(true)
	{
		if( !(detectorOutputFile >> detectorLine)) break;
		groundTruthFile >> groundTruthLine;

		int detect = detectorLine;
		int ground = groundTruthLine;

		//false positives
		if( (detect==1) && (ground==0) )
		{
			falsePositives++;
		}

		//false negatives
		if( (detect==0) && (ground==1) )
		{
			falseNegatives++;
		}

		//true positives
		if( (detect==1) && (ground==1) )
		{
			truePositives++;
		}

		//true negatives
		if( (detect==0) && (ground==0) )
		{
			trueNegatives++;
		}

	}
	cout << "falsePositives " << falsePositives<< endl;
	cout << "falseNegatives " << falseNegatives<< endl;
	cout << "truePositives " << truePositives<< endl;
	cout << "trueNegatives " << trueNegatives<< endl;
	float acc = (float)(truePositives + trueNegatives)/(float)(truePositives + trueNegatives + falsePositives + falseNegatives);
	cout << "Accuracy " << acc<< endl;
}

void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = cv::Mat(rows, cols, CV_32FC1 );
    std::vector< Mat >::const_iterator itr = train_samples.begin();
    std::vector< Mat >::const_iterator end = train_samples.end();
    for( int i = 0 ; itr != end ; ++itr, ++i )
    {
        CV_Assert( itr->cols == 1 ||
            itr->rows == 1 );
        if( itr->cols == 1 )
        {
            transpose( *(itr), tmp );
            tmp.copyTo( trainData.row( i ) );
        }
        else if( itr->rows == 1 )
        {
            itr->copyTo( trainData.row( i ) );
        }
    }
}


