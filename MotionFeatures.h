#ifndef MOTIONFEATURES_H_
#define MOTIONFEATURES_H_

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/ocl/ocl.hpp>
#include <vector>
#include <deque>
#include "Trail.h"

class BallDetector {
private:
	std::vector<cv::Point2f> centers;
	std::vector<std::vector<cv::Point>> contours;
	std::vector<float> areas;
	cv::gpu::MOG_GPU bgsub;
	float thMin, thMax;
	cv::Mat dilEl;
public:
	BallDetector(float thMin, float thMax);
	int eval(cv::Mat &frame, std::deque<cv::Point2f> &out);
	void dbg(cv::Mat &mat);
};

class MotionFeatures {
private:
	int fr, num;
	BallDetector ballDet;
	cv::Mat features;
	std::deque<Trail> tr, trTmp;
	static const float minBallArea, maxBallArea;
	cv::Mat fvect(void);
public:
	MotionFeatures(int num = 10);
	int eval(cv::Mat &frame, cv::Mat &features);
	void dbgBall(cv::Mat &mat);
	void dbgTrails(cv::Mat &mat);
	int trCnt(void);
	stats_t lastStats(void);
	void reset(void);
};

#endif /* MOTIONFEATURES_H_ */
