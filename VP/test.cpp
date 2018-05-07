#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <functional>
#include <numeric>
#include <iostream>
#include <string>
#include <time.h>
#include <iomanip>
#include <mmintrin.h>
#include <sstream>
#include <filesystem>
using namespace cv;
using namespace std;
namespace fs = std::experimental::filesystem;
#define PI 3.141592653f
struct vals {
	float arc;
	float d;
};
vals arcMatrix[200][400];
Mat oddKernels[36], evenKernels[36];
uchar directions[85][128];  
int vp_scale_x, vp_scale_y; //vp position on resized image
int width = 128; //resize width
int height = 85; //resize height
float scale_factor_width;
float scale_factor_height;

void myGaborKernel(float theta, Mat &oddKernel, Mat &evenKernel)
{
	float lamda = 5.0f;
	int size_kernel = 17;
	float sigma = (float)size_kernel / 9.0f;
	int k = (size_kernel - 1) / 2;
	oddKernel.create(size_kernel, size_kernel, CV_32F);
	evenKernel.create(size_kernel, size_kernel, CV_32F);
	for (int x = -1 * k; x <= k; x++)
		for (int y = -1 * k; y <= k; y++)
		{
			float a = x * cos(theta) + y * sin(theta);
			float b = y * cos(theta) - x * sin(theta);
			float oddResp = exp(-1.0f / 8.0f / sigma / sigma * (4.0f*a*a + b * b)) * sin(2.0f*PI*a / lamda);
			float evenResp = exp(-1.0f / 8.0f / sigma / sigma * (4.0f*a*a + b * b)) * cos(2.0f*PI*a / lamda);
			oddKernel.at<float>(x + k, y + k) = oddResp;
			evenKernel.at<float>(x + k, y + k) = evenResp;
		}

	float u1 = mean(oddKernel)[0];
	float u2 = mean(evenKernel)[0];
	float *f1 = (float*)oddKernel.datastart;
	float *f2 = (float*)oddKernel.dataend;
    float l2sum1 = 0, l2sum2 = 0;
	float x = 0, y = 0;

	for (int i = 0; i < size_kernel; i++)
	{
		for (int j = 0; j < size_kernel; j++)
		{
			oddKernel.at<float>(i, j) = oddKernel.at<float>(i, j) - u1;
			evenKernel.at<float>(i, j) = evenKernel.at<float>(i, j) - u2;
			x = oddKernel.at<float>(i, j);
			y = evenKernel.at<float>(i, j);
			l2sum1 += x * x / (17.0f*17.0f);
			l2sum2 += y * y / (17.0f*17.0f);
		}
	}

	for (int i = 0; i < size_kernel; i++)
		for (int j = 0; j < size_kernel; j++)
		{
			oddKernel.at<float>(i, j) = oddKernel.at<float>(i, j) / l2sum1;
			evenKernel.at<float>(i, j) = evenKernel.at<float>(i, j) / l2sum2;
		}

}

Mat computeVpScore(Mat image_origin, Mat image)
{
	int n_theta = 36;
	scale_factor_width = (float)width / (float)image_origin.cols;
	scale_factor_height = (float)height / (float)image_origin.rows;
	printf("img rows=%d, cols=%d ", image_origin.rows, image_origin.cols);
	int m = image.rows, n = image.cols;
	//printf("m=%d, n=%d\n", m, n);
	float scores[85][128];
	float ***gabors = new float**[m];

	for (int i = 0; i<m; i++) {
		gabors[i] = new float*[n];
		for (int j = 0; j<n; j++)
			gabors[i][j] = new float[n_theta];
	}

	Mat	oddfiltered(image.rows, image.cols, CV_32F), evenfiltered(image.rows, image.cols, CV_32F);

	for (int t = 0; t < n_theta; t++)
	{
		filter2D(image, oddfiltered, -1, oddKernels[t]);
		filter2D(image, evenfiltered, -1, evenKernels[t]);

		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				gabors[i][j][t] = pow(oddfiltered.at<float>(i, j), 2.0f) + pow(evenfiltered.at<float>(i, j), 2.0f);


	}

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
		{
			int idx = (float)(max_element(gabors[i][j], gabors[i][j] + n_theta) - gabors[i][j]);
			directions[i][j] = (uchar)idx;

		}
	}
	int thresh = 2.0f * 180.0f / (float)n_theta;
	int r = (m + n) / 7;
	float r_dia = sqrtf(m*m + n * n);
	int gamma, c;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			scores[i][j] = 0;
			float tmepScore = 0;
			for (int i1 = i + 1; i1 < m && i1<i + 40; i1++)
			{
				for (int j1 = 0; j1 < n; j1++)
				{

					c = (float)directions[i1][j1] / (float)n_theta * 180.0f;
				
					gamma = arcMatrix[i1 - i][j1 - j + 200].arc;

					if (abs(c - gamma) < thresh)
					{
						tmepScore += 1;
					}
				}
			}
			scores[i][j] = tmepScore;
		}
	}

	Point p_max, p_min;
	double score_max, score_min;
	Mat matscores(m, n, CV_32F);
	memcpy(matscores.data, scores, m*n * sizeof(float));
	cv::minMaxLoc(matscores, &score_min, &score_max, &p_min, &p_max);
	float scale = score_max / 255.0f;
	float vp_x = p_max.x / scale_factor_width;
	float vp_y = p_max.y / scale_factor_height;
	vp_scale_x = p_max.x;
	vp_scale_y = p_max.y;
	printf("vp_scale_x:%d ", vp_scale_x);
	printf("vp_scale_y:%d\n", vp_scale_y);

	cv::circle(image_origin, cvPoint(vp_x, vp_y), 5, Scalar(0,0,255),2);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			delete[] gabors[i][j];
	for (int i = 0; i < m; i++)
		delete[] gabors[i];
	delete[] gabors;
	return image_origin;
}
int* findDominantEdges(float candidate_x, float candidate_y, bool multiple) {
	// Assume the length of road border should be at least a third of the image height
	float dis_threshold = height / 3;
	int dominant_angle; // Angle of the dominant line
	int dominant_OCR = -1;
	// First three from the left half of the image, latter three from the right half of image
	int *dominant_angles = new int[6];
	int *dominant_OCRs = new int[6];
	for (int i = 0; i < 6; i++) {
		dominant_OCRs[i] = -6 + i;
		dominant_angles[i] = -6 + i;
	}

	float bottom_left = atan2((float)(height - candidate_y), (float)candidate_x) * 180.0f / PI;
	float bottom_right = atan2((float)(height - candidate_y), (float)(width - candidate_x)) * 180.0f / PI;
	cout << "bottom_left " << bottom_left << endl;
	cout << "bottom_right " << bottom_right << endl;
	for (int i = 3; i < bottom_left; i += 5) {
		int OCR = 0;
		float dis = (float)candidate_x / cos(i * PI / 180.0f);
		if (dis < dis_threshold) {
			continue;
		}
		for (int j = (int)(dis / 9); j < dis; j += (int)(dis / 9)) {
			float sample_point_x = candidate_x - j * cos(i * PI / 180.0f);
			float sample_point_y = candidate_y + j * sin(i * PI / 180.0f);
			float angle = (float)directions[(int)sample_point_y][(int)sample_point_x] / 36.0f * 180.0f;
			if (angle >= (i - 2.5) && angle <= (i + 2.5))
				OCR += 1;
		}

		float y_border = (float)candidate_x * tan(i * PI / 180.0f);

		if (multiple == false && OCR > dominant_OCR) {
			dominant_OCR = OCR;
			dominant_angle = i;
		}
		else if (multiple == true) {
			if (OCR > dominant_OCRs[0]) {
				dominant_OCRs[2] = dominant_OCRs[1];
				dominant_OCRs[1] = dominant_OCRs[0];
				dominant_OCRs[0] = OCR;
				dominant_angles[2] = dominant_angles[1];
				dominant_angles[1] = dominant_angles[0];
				dominant_angles[0] = i;
				continue;
			}
			else if (OCR > dominant_OCRs[1]) {
				dominant_OCRs[2] = dominant_OCRs[1];
				dominant_OCRs[1] = OCR;
				dominant_angles[2] = dominant_angles[1];
				dominant_angles[1] = i;
				continue;
			}
			else if (OCR > dominant_OCRs[2]) {
				dominant_OCRs[2] = OCR;
				dominant_angles[2] = i;
			}	
		}
	}
	for (int i = 3; i < bottom_right; i += 5) {
		int OCR = 0;
		float dis = (float)(width - candidate_x) / cos(i * PI / 180.0f);
		if (dis < dis_threshold) {
			continue;
		}
		for (int j = (int)(dis / 9); j < dis; j += (int)(dis / 9)) {
			float sample_point_x = candidate_x + j * cos(i * PI / 180.0f);
			float sample_point_y = candidate_y + j * sin(i * PI / 180.0f);
			float angle = (float)directions[(int)sample_point_y][(int)sample_point_x] / 36.0f * 180.0f;
			if (angle >= (180 - i - 2.5) && angle <= (180 - i + 2.5))
				OCR += 1;
		}
		float y_border = (float)(width - candidate_x) * tan(i * PI / 180.0f);
		if (multiple == false && OCR > dominant_OCR) {
			dominant_OCR = OCR;
			dominant_angle = 180 - i;
		}
		else if (multiple == true) {
			if (OCR > dominant_OCRs[3]) {
				dominant_OCRs[5] = dominant_OCRs[4];
				dominant_OCRs[4] = dominant_OCRs[3];
				dominant_OCRs[3] = OCR;
				dominant_angles[5] = dominant_angles[4];
				dominant_angles[4] = dominant_angles[3];
				dominant_angles[3] = 180 - i;
				continue;
			}
			else if (OCR > dominant_OCRs[4]) {
				dominant_OCRs[5] = dominant_OCRs[4];
				dominant_OCRs[4] = OCR;
				dominant_angles[5] = dominant_angles[4];
				dominant_angles[4] = 180 - i;
				continue;
			}
			else if (OCR > dominant_OCRs[5]) {
				dominant_OCRs[5] = OCR;
				dominant_angles[5] = 180 - i;
			}
		}
	}
	// Assume the angle between border is larger than 20 degrees
	for (int i = bottom_left; i < 80; i += 5) {
		int OCR = 0;
		float dis = (float)(height - candidate_y) / sin(i * PI / 180.0f);
		if (dis < dis_threshold) {
			continue;
		}
		for (int j = (int)(dis / 9); j < dis; j += (int)(dis / 9)) {
			float sample_point_x = candidate_x - j * cos(i * PI / 180.0f);
			float sample_point_y = candidate_y + j * sin(i * PI / 180.0f);
			float angle = (float)directions[(int)sample_point_y][(int)sample_point_x] / 36.0f * 180.0f;
			if (angle >= (i - 2.5) && angle <= (i + 2.5))
				OCR += 1;
		}
		float x_border = (float)(height - candidate_y) / tan(i * PI / 180.0f);
		if (multiple == false && OCR > dominant_OCR) {
			dominant_OCR = OCR;
			dominant_angle = i;
		}
		else if (multiple == true) {
			if (OCR > dominant_OCRs[0]) {
				dominant_OCRs[2] = dominant_OCRs[1];
				dominant_OCRs[1] = dominant_OCRs[0];
				dominant_OCRs[0] = OCR;
				dominant_angles[2] = dominant_angles[1];
				dominant_angles[1] = dominant_angles[0];
				dominant_angles[0] = i;
				continue;
			}
			else if (OCR > dominant_OCRs[1]) {
				dominant_OCRs[2] = dominant_OCRs[1];
				dominant_OCRs[1] = OCR;
				dominant_angles[2] = dominant_angles[1];
				dominant_angles[1] = i;
				continue;
			}
			else if (OCR > dominant_OCRs[2]) {
				dominant_OCRs[2] = OCR;
				dominant_angles[2] = i;
			}
		}
	}
	for (int i = bottom_right; i < 80; i += 5) {
		int OCR = 0;
		float dis = (float)(height - candidate_y) / sin(i * PI / 180.0f);
		if (dis < dis_threshold) {
			continue;
		}
			
		for (int j = (int)(dis / 9); j < dis; j += (int)(dis / 9)) {
			float sample_point_x = candidate_x + j * cos(i * PI / 180.0f);
			float sample_point_y = candidate_y + j * sin(i * PI / 180.0f);
			float angle = (float)directions[(int)sample_point_y][(int)sample_point_x] / 36.0f * 180.0f;
			if (angle >= (180 - i - 2.5) && angle <= (180 - i + 2.5))
				OCR += 1;
		}
		float x_border = (float)(height - candidate_y) / tan(i * PI / 180.0f);
		if (multiple == false && OCR > dominant_OCR) {
			dominant_OCR = OCR;
			dominant_angle = 180 - i;
		}
		else if (multiple == true) {
			if (OCR > dominant_OCRs[3]) {
				dominant_OCRs[5] = dominant_OCRs[4];
				dominant_OCRs[4] = dominant_OCRs[3];
				dominant_OCRs[3] = OCR;
				dominant_angles[5] = dominant_angles[4];
				dominant_angles[4] = dominant_angles[3];
				dominant_angles[3] = 180 - i;
				continue;
			}
			else if (OCR > dominant_OCRs[4]) {
				dominant_OCRs[5] = dominant_OCRs[4];
				dominant_OCRs[4] = OCR;
				dominant_angles[5] = dominant_angles[4];
				dominant_angles[4] = 180 - i;
				continue;
			}
			else if (OCR > dominant_OCRs[5]) {
				dominant_OCRs[5] = OCR;
				dominant_angles[5] = 180 - i;
			}
		}
	}
	if (multiple == false) {
		int *dominant = new int[2];
		dominant[0] = dominant_angle;
		dominant[1] = dominant_OCR;
		return dominant;
	}
	else {
		int *dominant_mul = new int[12];
		for (int i = 0; i < 6; i++) {
			dominant_mul[2 * i] = dominant_angles[i];
			dominant_mul[2 * i + 1] = dominant_OCRs[i];
		}
		return dominant_mul;
	}
	
}

Point updateVP(int dominant_angle, int dominant_OCR, Mat image_origin) {
	float candidate_x;
	float candidate_y;
	int OCR_sum = 0;
	Point updated_vp = cv::Point(0,0);
	float fst_angle = -1.0, snd_angle = -1.0;
	for (int i = 0; i <= 3; i++) {
		int sum = 0;
		candidate_x = vp_scale_x - 5 * i * cos(dominant_angle * PI / 180.0f);
		candidate_y = vp_scale_y + 5 * i * sin(dominant_angle * PI / 180.0f);
		int* dominant_mul = findDominantEdges(candidate_x, candidate_y, true);
		for (int i = 0; i < 6; i++) {
			sum += dominant_mul[2 * i + 1];
			cout << dominant_mul[2 * i] << " " << dominant_mul[2 * i + 1]<<" ";
		}
		cout << endl;
		if (sum > OCR_sum) {
			OCR_sum = sum;
			fst_angle = dominant_mul[0];
			snd_angle = dominant_mul[6];
			updated_vp = Point(candidate_x, candidate_y);
		}
	}
	cout << OCR_sum << ' ' << updated_vp.x << ' ' << updated_vp.y << endl;
	if (fst_angle > 0) {
		float px_1 = updated_vp.x - 100 * cos(fst_angle * PI / 180.0f);
		float py_1 = updated_vp.y + 100 * sin(fst_angle * PI / 180.0f);
		cv::line(image_origin, cv::Point(px_1 / scale_factor_width, py_1 / scale_factor_height), cv::Point(updated_vp.x / scale_factor_width, updated_vp.y / scale_factor_height), Scalar(0, 255, 0), 2);
	}
	if (snd_angle > 0) {
		float px_2 = updated_vp.x - 100 * cos(snd_angle * PI / 180.0f);
		float py_2 = updated_vp.y + 100 * sin(snd_angle * PI / 180.0f);
		cv::line(image_origin, cv::Point(px_2 / scale_factor_width, py_2 / scale_factor_height), cv::Point(updated_vp.x / scale_factor_width, updated_vp.y / scale_factor_height), Scalar(0, 255, 0), 2);
	}
	return updated_vp;
}

int segmentRoad(Mat image_origin, Mat image) {
	int *dominant = findDominantEdges(vp_scale_x, vp_scale_y, false);
	cout << "dominant angle: " << dominant[0] << endl;
	cout << "dominant ocr: " << dominant[1] << endl;
	float px = vp_scale_x - 100 * cos(dominant[0] * PI / 180.0f);
	float py = vp_scale_y + 100 * sin(dominant[0] * PI / 180.0f);
	cv::line(image_origin, cv::Point(px / scale_factor_width, py / scale_factor_height), cv::Point(vp_scale_x / scale_factor_width, vp_scale_y / scale_factor_height), Scalar(0, 0, 255),2);
	Point update = updateVP(dominant[0], dominant[1], image_origin);
	if (update.x > 0 || update.y > 0) {
		cv::circle(image_origin, cv::Point(update.x / scale_factor_width, update.y / scale_factor_height), 3, Scalar(0, 255, 0), 2);
	}
	namedWindow("Display window 2", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window 2", image_origin);
	return 0;
}

int process(string filePath, bool video) {
	if (video == false) {	
		Mat image_origin = imread(filePath);
		Mat img_gray, image;
		cvtColor(image_origin, img_gray, CV_RGB2GRAY);
		Mat img_resize(height, width, CV_32F);
		resize(img_gray, img_resize, img_resize.size());
		img_resize.convertTo(image, CV_32F);
		computeVpScore(image_origin, image);
		segmentRoad(image_origin, image);
	}
	else {
		VideoCapture video(filePath);
		if (!video.isOpened()) {
			cout << "failed to open video" << endl;
		}
	
		Mat img, res;
		for (int i = 0; i < 50; i++) {
			video >> img;
			if (img.cols*img.rows>0) {
				Mat image_origin = img;
				Mat img_gray, image;
				cvtColor(image_origin, img_gray, CV_RGB2GRAY);
				Mat img_resize(height, width, CV_32F);
				resize(img_gray, img_resize, img_resize.size());
				img_resize.convertTo(image, CV_32F);
				computeVpScore(image_origin, image);
				segmentRoad(image_origin, image);
				waitKey(500);
			}
		}
	}
	
	return 0;
}


int main()
{
	for (int j = 0; j < 200; j++)
	{
		for (int i = 0; i < 400; i++)
		{
			float t1 = sqrtf(pow(i - 200.0f, 2.0f) + pow(j, 2.0f));
			arcMatrix[j][i].arc = acos((200.0f - (float)i) / t1) / PI * 180.0f;
			arcMatrix[j][i].d = t1;
		}
	}
	int n_theta = 36;
	for (int t = 0; t < n_theta; t++)
	{
		float theta = PI * (float)t / (float)n_theta;
		Mat oddKernel(17, 17, CV_32F), evenKernel(17, 17, CV_32F);
		myGaborKernel(theta, oddKernels[t], evenKernels[t]);
	}
	// comment out this one if you want to apply on images
	/*string path = "C:\\Users\\Lina\\source\\repos\\CVtest\\CVtest\\img";
	for (auto & p : fs::directory_iterator(path))
	{
        cout << p << endl;
		process(p.path().string(), false);
	}*/
	// video
	string path = "C:\\Users\\Lina\\source\\repos\\CVtest\\CVtest\\video\\vr2.mp4";
	process(path, true);
		
	system("pause");
	return 0;
}
