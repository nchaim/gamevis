#ifndef TRAIL_H_
#define TRAIL_H_

#include <opencv2/core/core.hpp>
#include <vector>

struct stats_t { float ax, ay, vx, vy, curv, dir, length; };

class Trail {
private:
	static const float maxPtDist, maxSpeedup, maxAngle, minLength;
	static const int numFrDone, minSize;
	struct frame_t { int n; float angle, totDist; cv::Vec2f point; };

	std::vector<cv::Point2f> points;
	std::vector<frame_t> frames;
	int statsFr, *curFrame;
	float sc;
	stats_t stats;
public:
	Trail(int *curFrame, cv::Point2f point, float scale = 1.0);
	Trail(const Trail&);
	bool testMatch(cv::Vec2f p, int r, float &dist, float &ang);
	bool push(cv::Point2f point);
	stats_t getStats(void);
	bool done(void);
	bool valid(void);
	int size(void) const;
	bool operator <(const Trail& pt) const;
	std::vector<cv::Point2f> getPoints(void);
};

#endif /* TRAIL_H_ */
