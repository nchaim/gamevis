#include "MotionFeatures.h"

using namespace cv;
using namespace std;

const float MotionFeatures::minBallArea = 200.0;
const float MotionFeatures::maxBallArea = 600.0;

MotionFeatures::MotionFeatures(int num) : fr(0), num(num),
		ballDet(minBallArea, maxBallArea), features(0, 7, CV_32FC1) {
}

float wrapAngle(float ang) {
	ang = fabs(ang);
	if (ang > 180.0) ang = 360.0 - ang;
	if (ang > 90.0) ang = 180.0 - ang;
	return ang;
}

Mat MotionFeatures::fvect(void) {
	float lastDir = 0; Mat res;
	for(auto trIt = tr.rbegin(); trIt != tr.rend(); trIt++) {
		stats_t st = trIt->getStats();
		float dDir = wrapAngle(st.dir-lastDir);
		lastDir = st.dir;
		if (trIt == tr.rbegin()) continue;
		float v[] = {st.vx, st.vy, st.ax, st.ay, st.curv, st.length, dDir};
		Mat mv = Mat(1, 7, CV_32FC1, &v);
		if (res.empty()) res = mv; else hconcat(mv, res, res);
	}
	return res;
}

int MotionFeatures::eval(cv::Mat &frame, cv::Mat &features) {
	deque<Point2f> pl;
	ballDet.eval(frame, pl);
	features.create(0, num, CV_32FC1);
	for(auto trIt = trTmp.begin(); trIt != trTmp.end(); ) {
		if (trIt->done() && trIt->valid()) {
			tr.push_front(*trIt);
			while((int) tr.size() > num) tr.pop_back();
			if ((int) tr.size() == num) features.push_back(fvect());
		}
		if (trIt->done()) { trIt = trTmp.erase(trIt); continue; }

		for (auto ptIt = pl.begin(); ptIt != pl.end(); ptIt++)
			if (trIt->push(*ptIt)) { pl.erase(ptIt); break; }
		trIt++;
	}
	for(auto pt : pl) trTmp.push_back(Trail(&fr, pt));
	if (!pl.empty()) sort(trTmp.rbegin(), trTmp.rend());
	fr++;
	return features.rows;
}

void arrow(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness, int line_type, float tipSize) {
    const double angle = atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );
    Point p(cvRound(pt2.x + tipSize * cos(angle + CV_PI / 4)),
    		cvRound(pt2.y + tipSize * sin(angle + CV_PI / 4)));
    line(img, p, pt2, color, thickness, line_type);
    p.x = cvRound(pt2.x + tipSize * cos(angle - CV_PI / 4));
    p.y = cvRound(pt2.y + tipSize * sin(angle - CV_PI / 4));
    line(img, p, pt2, color, thickness, line_type);
}

void MotionFeatures::dbgTrails(cv::Mat &mat) {
	Vec3f clr = Vec3f(82, 151, 255);
	Vec3f dc = (Vec3f(255, 82, 154) - clr) / num;
	for(auto trIt = tr.rbegin(); trIt != tr.rend(); trIt++) {
		vector<Point2f> pts = trIt->getPoints();
		for (auto ptIt = pts.begin(); ptIt != pts.end() && pts.size() > 1; ptIt++) {
			Scalar sclr = Scalar(clr[0], clr[1], clr[2]);
			if (ptIt == (pts.end()-1)) arrow(mat, *(ptIt-1), *ptIt, sclr, 1, CV_AA, 15);
			if (ptIt != pts.begin()) line(mat, *(ptIt-1), *ptIt, sclr, 1, CV_AA);
			circle(mat, *ptIt, 2, sclr, CV_FILLED, CV_AA);
		}
		clr += dc;
	}
}

stats_t MotionFeatures::lastStats(void) {
	return tr.empty() ? stats_t() : tr.front().getStats();
}

int MotionFeatures::trCnt(void) { return tr.size(); }
void MotionFeatures::reset(void) { tr.clear(); }
void MotionFeatures::dbgBall(cv::Mat &mat) { ballDet.dbg(mat); }


BallDetector::BallDetector(float thMin, float thMax) :
	thMin(thMin), thMax(thMax) {
	bgsub.noiseSigma = 5.0;
	const int dSize = 5;
	dilEl = getStructuringElement(MORPH_ELLIPSE, Size(2*dSize+1, 2*dSize+1), Point(dSize, dSize));
}

int BallDetector::eval(cv::Mat &frame, std::deque<cv::Point2f> &out) {
	Mat bgmask;
	gpu::GpuMat gFr(frame), gOm;
	bgsub(gFr, gOm, -1);
	gOm.download(bgmask);
	ocl::oclMat oMsk(bgmask);
    ocl::medianFilter(oMsk, oMsk, 5);
    ocl::medianFilter(oMsk, oMsk, 5);
    oMsk.download(bgmask);
	dilate(bgmask, bgmask, dilEl);
	contours.clear(); centers.clear(); areas.clear(); out.clear();
	findContours(bgmask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	for (vector<Point> c : contours) {
		Moments m = moments(c, false);
		Point2f center = Point2f(m.m10/m.m00 , m.m01/m.m00);
		float area = m.m00;
		if (area < thMax && area > thMin) out.push_back(center);
		centers.push_back(center); areas.push_back(area);
	}
	return out.size();
}

void BallDetector::dbg(cv::Mat &mat) {
	for (unsigned int i = 0; i < contours.size(); i++) {
		bool match = (areas[i] < thMax && areas[i] > thMin);
		if (match) circle(mat, centers[i], 2, Scalar(148, 255, 71), CV_FILLED, 8, 0 );
		Scalar clr = match ? Scalar(148, 255, 71) : Scalar(255, 71, 169);
		drawContours(mat, contours, i, clr, 2, 8);
	}
}
