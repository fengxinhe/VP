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
using namespace cv;
using namespace std;
#define PI 3.141592653f
struct vals {
	float arc;
	float d;
};
vals arcMatrix[200][400];
Mat oddKernels[36], evenKernels[36];

void myGaborKernel(float theta, Mat &oddKernel, Mat &evenKernel)
{
	//gittest
	float lamda = 5.0f;
	int size_kernel = 17;
	float sigma = (float)size_kernel / 9.0f;
	int k = (size_kernel - 1) / 2;
	oddKernel.create(size_kernel, size_kernel, CV_32F);
	evenKernel.create(size_kernel, size_kernel, CV_32F);
	for (int x = -1 * k; x <= k; x++)
		for (int y = -1 * k; y <= k; y++)
		{
			float a = x*cos(theta) + y*sin(theta);
			float b = y*cos(theta) - x*sin(theta);
			float oddResp = exp(-1.0f / 8.0f / sigma / sigma*(4.0f*a*a + b*b)) * sin(2.0f*PI*a / lamda);
			float evenResp = exp(-1.0f / 8.0f / sigma / sigma*(4.0f*a*a + b*b)) * cos(2.0f*PI*a / lamda);
			oddKernel.at<float>(x + k, y + k) = oddResp;
			evenKernel.at<float>(x + k, y + k) = evenResp;
		}

	float u1 = mean(oddKernel)[0];
	float u2 = mean(evenKernel)[0];
	float *f1 = (float*)oddKernel.datastart;
	float *f2 = (float*)oddKernel.dataend;
	//Mat_<float>::iterator oddit1 = oddKernel.begin<float>(), oddit2 = oddKernel.end<float>();
	//Mat_<float>::iterator evenit1 = evenKernel.begin<float>(), evenit2 = evenKernel.end<float>();

	for (int i = 0; i < size_kernel; i++)
	{
		for (int j = 0; j < size_kernel; j++)
		{
			oddKernel.at<float>(i, j) = oddKernel.at<float>(i, j) - u1;
			evenKernel.at<float>(i, j) = evenKernel.at<float>(i, j) - u2;
		}
	}

	//get the L2Norm
	float l2sum1 = 0, l2sum2 = 0;
	float x=0,y=0;
	for(int i=0;i<size_kernel;i++){
		for(int j=0;j<size_kernel;j++){
			x=oddKernel.at<float>(i,j);
			y=evenKernel.at<float>(i,j);
			l2sum1+=x*x;
			l2sum2+=y*y;
		}
	}
	//for_each(oddit1, oddit2, [&l2sum1](float x) { l2sum1 += x*x; });
	//for_each(evenit1, evenit2, [&l2sum2](float x) { l2sum2 += x*x; });
	l2sum1 /= (17.0f*17.0f);
	l2sum2 /= (17.0f*17.0f);
	//divide the L2Norm
	for (int i = 0; i < size_kernel; i++)
		for (int j = 0; j < size_kernel; j++)
		{
			oddKernel.at<float>(i, j) = oddKernel.at<float>(i, j) / l2sum1;
			evenKernel.at<float>(i, j) = evenKernel.at<float>(i, j) / l2sum2;
		}
	//for (int i = 0; i < size_kernel; i++)
	//{
	//	for (int j = 0; j < size_kernel; j++)
	//		cout << setw(8) << setiosflags(ios::fixed) << setprecision(2) << oddKernel.at<float>(i, j) << ' ';
	//	cout << endl;
	//}
	//cout << endl;
	//for (int i = 0; i < size_kernel; i++)
	//{
	//	for (int j = 0; j < size_kernel; j++)
	//		cout << setw(8) << setiosflags(ios::fixed) << setprecision(2) << evenKernel.at<float>(i, j) << ' ';
	//	cout << endl;
	//}

	//for_each(oddit1, oddit2, [&l2sum1](float x){ x /= l2sum1; });
	//for_each(evenit1, evenit2, [&l2sum2](float x){ x /= l2sum2; });
}


Mat computeVpScore(string filePath)
{
	Mat image_origin = imread(filePath);
	Mat img_gray, img_float;
	cvtColor(image_origin, img_gray, CV_RGB2GRAY);
	img_gray.convertTo(img_float, CV_32F);

	int n_theta = 36;
	int width = 128;
	float scale_factor = (float)width / (float)image_origin.cols;
	Mat image(img_gray.rows * scale_factor, width, CV_32F);
	resize(img_float, image, image.size());
	int m = image.rows, n = image.cols;
	cout << img_gray.rows << endl;

	//Mat scores(m, n, CV_32F);
	float scores[85][128];
	float ***gabors = new float**[m];
	//for_each(gabors, gabors + m, [n](float** &x) {x = new float*[n]; });
	//for_each(gabors, gabors + m, [n, n_theta](float** x) {for_each(x, x + n, [&, n_theta](float* &y) { y = new float[n_theta]; }); });
	for( int i=0;i<m;i++){
		gabors[i]=new float*[n];
		for(int j=0;j<n;j++)
			gabors[i][j]=new float[n_theta];
	}
	//Mat filtered(image.rows, image.cols, CV_32F);
	Mat	oddfiltered(image.rows, image.cols, CV_32F), evenfiltered(image.rows, image.cols, CV_32F);
	//float filtered[85][128];
	for (int t = 0; t < n_theta; t++)
	{
		filter2D(image, oddfiltered, -1, oddKernels[t]);
		filter2D(image, evenfiltered, -1, evenKernels[t]);

		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				//filtered.at<float>(i, j) = pow(oddfiltered.at<float>(i, j), 2.0f) + pow(evenfiltered.at<float>(i, j), 2.0f);
				gabors[i][j][t] = pow(oddfiltered.at<float>(i, j), 2.0f) + pow(evenfiltered.at<float>(i, j), 2.0f);

		//for (int i = 0; i < m; i++)
		//	for (int j = 0; j < n; j++)
				//gabors[i][j][t] = filtered.at<float>(i, j);
			//	gabors[i][j][t] = filtered[i][j];
	}
	cout << image.rows << endl;
	cout << image.cols << endl;

	//Mat directions(image.rows, image.cols, CV_8U);
	//Mat confidences(image.rows, image.cols, CV_32F);
	//float confidences[85][128];
	uchar directions[85][128];
	
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
		{

			int idx = (float)(max_element(gabors[i][j], gabors[i][j] + n_theta) - gabors[i][j]);
			//directions.at<uchar>(i, j) = (uchar)idx;
			directions[i][j] = (uchar)idx;
			//float max_resp = gabors[i][j][idx];
			/*sort(gabors[i][j], gabors[i][j] + n_theta, greater<float>());
			if (max_resp > 0.5f)
			confidences.at<float>(i, j) = (1 - accumulate(gabors[i][j] + 4, gabors[i][j] + 15, 0.0f) / 11.0f / max_resp);
			else
			confidences.at<float>(i, j) = 0;*/
		}

	int thresh = 2.0f * 180.0f / (float)n_theta;
	int r = (m + n) / 7;
	float r_dia = sqrtf(m*m + n*n);
	int gamma, c;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			//scores.at<float>(i, j) = 0;
			scores[i][j]=0;
			float tmepScore = 0;
			for (int i1 = i + 1; i1 < m && i1<i + 40; i1++)
			{
				for (int j1 = 0; j1 < n; j1++)
				{
					//c = (float)directions.at<uchar>(i1, j1) / (float)n_theta * 180.0f;
					//cout << "c->" + c << endl;
				    c = (float)directions[i1][j1] / (float)n_theta * 180.0f;
					/*if (c < 5.0f || (85.0f < c && c < 95.0f) || c>175.0f)
					continue;*/
					//float d = sqrtf(pow(i - i1, 2.0) + pow(j - j1, 2.0));
					//float d = arcMatrix[i1-i][j1-j+200].d;
					//float gamma = acosf(((float)j - (float)j1) / d)/ PI * 180.0f;					
					gamma = arcMatrix[i1 - i][j1 - j + 200].arc;

					if (abs(c - gamma) < thresh/* && confidences.at<float>(i1, j1) > 0.35*/)
					{
						//tmepScore += 1 / (1 + pow(c - gamma, 2.0f)*pow(d / r_dia, 2.0));
						tmepScore += 1;
					}
				}
			}
			//scores.at<float>(i, j) = tmepScore;
			scores[i][j] = tmepScore;
		}
	}

	Point p_max, p_min;
	double score_max, score_min;
	Mat matscores(m, n, CV_32F);
	memcpy(matscores.data, scores, m*n * sizeof(float));
	cv::minMaxLoc(matscores, &score_min, &score_max, &p_min, &p_max);
	float scale = score_max / 255.0f;
	//for (int i = 0; i < m; i++)
	//	for (int j = 0; j < n; j++)
	//		matscores.at<float>(i, j) = round(matscores.at<float>(i, j) / scale);

	cv::circle(image_origin, cvPoint(p_max.x / scale_factor, p_max.y / scale_factor), 10, Scalar(0), 5, 8, 0);
	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    	imshow( "Display window", image_origin );                   // Show our image inside it.

    	waitKey(0);                               
	//Release memory
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			delete[] gabors[i][j];
	for (int i = 0; i < m; i++)
		delete[] gabors[i];
	delete[] gabors;
	//cout << scores;
	return image_origin;

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
			//cout << j << ' ' << i << ' ' << arcMatrix[j][i].arc << ' ' << t1<<"   ";
		}
		//cout << endl;
	}
	int n_theta = 36;
	for (int t = 0; t < n_theta; t++)
	{
		float theta = PI*(float)t / (float)n_theta;
		Mat oddKernel(17, 17, CV_32F), evenKernel(17, 17, CV_32F);
		myGaborKernel(theta, oddKernels[t], evenKernels[t]);
	}
	//visit("img", computeVpScore);
	computeVpScore("road.jpg");
	/*
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
	return -1;
	}
	Mat frame;
	bool stop = false;
	while (!stop)
	{
	cap >> frame;
	imshow("µ±Ç°ÊÓÆµ", frame);
	if (waitKey(30) >= 0)
	stop = true;
	}
	*/
	system("pause");
	return 0;
}
