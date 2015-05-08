#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Trail.h"

using namespace cv;
using namespace std;

const float Trail::maxPtDist = 50.0;
const float Trail::maxSpeedup = 2.0;
const float Trail::maxAngle = 15.0;
const float Trail::minLength = 50.0;
const int Trail::numFrDone = 3;
const int Trail::minSize = 6;

Trail::Trail(int *curFrame, Point2f point, float scale) :
		statsFr(-1), curFrame(curFrame), sc(scale), stats({}) {
	frames.push_back({*curFrame, 0.0, 0.0, Vec2f(point.x, point.y)});
	points.push_back(point);
}

bool Trail::testMatch(Vec2f p, int r, float &dist, float &ang) {
	int i = frames.size()-r-1; if (i < 0) return false;
	frame_t &fr = frames[i];
	dist = norm(p-fr.point);
	ang = fastAtan2(p[1]-fr.point[1], p[0]-fr.point[0]);
	if ((dist*sc) > maxPtDist) return false;
	if (frames.size() == 1 || dist == 0.0) return true;
	if (abs(ang-fr.angle) > maxAngle) return false;
	float speed = dist/(*curFrame-fr.n);
	frame_t &frp = frames[max(i-1, 0)];
	float tailSpeed = (fr.totDist-frp.totDist)/(fr.n-frp.n);
	float speedRatio = max(speed/tailSpeed, tailSpeed/speed);
	if (speedRatio > maxSpeedup) return false;
	return true;
}

bool Trail::push(Point2f point) {
	float dist, ang;
	Vec2f p(point.x, point.y);
	bool res0 = testMatch(p, 0, dist, ang);
	bool res1 = !res0 && testMatch(p, 1, dist, ang);
	if (res1) { frames.pop_back(); points.pop_back(); }
	if (res0 || res1) {
		float totDist = frames.back().totDist + dist;
		frames.push_back({*curFrame, ang, totDist, p});
		points.push_back(point);
	}
	return (res0 || res1);
}

stats_t Trail::getStats(void) {
	if (frames.size() < 3 || frames.back().n == statsFr)
		return stats;
	Vec2f dirx = normalize(frames.back().point-frames[0].point);
	Vec2f diry = Vec2f(-dirx[1], dirx[0]);
	Vec2f tDist = Vec2f(0, 0);
	vector<Vec2f> vvx, vvy;
	float ang0 = 0.0, ang1 = 0.0;
	for(uint32_t i = 1; i < frames.size(); i++) {
		Vec2f s0 = frames[i].point-frames[i-1].point;
		Vec2f s = Vec2f(s0.dot(dirx), s0.dot(diry));
		Vec2f v = s/(frames[i].n-frames[i-1].n);
		vvx.push_back(Vec2f(frames[i].n, v[0]));
		vvy.push_back(Vec2f(frames[i].n, v[1]));
		tDist += s;
		if (i <= 3) ang0 += frames[i].angle;
		if (i >= (frames.size()-3)) ang1 += frames[i].angle;
	}
	Vec4f lx, ly;
	fitLine(vvx, lx, CV_DIST_L2, 0, 0.01, 0.01);
	fitLine(vvy, ly, CV_DIST_L2, 0, 0.01, 0.01);
	stats.ax = lx[1]/lx[0]*sc;
	stats.ay = ly[1]/ly[0]*sc;
	float tn = frames.back().n - frames.front().n + 1;
	stats.vx = tDist[0]/tn*sc;
	stats.vy = tDist[1]/tn*sc;
	stats.curv = fabs((ang1-ang0)/3.0);
	stats.dir = fastAtan2(dirx[1], dirx[0]);
	stats.length = frames.back().totDist*sc;
	return stats;
}

Trail::Trail(const Trail& t) : points(t.points), frames(t.frames),
		statsFr(t.statsFr), curFrame(t.curFrame), sc(t.sc), stats(t.stats) { }

bool Trail::done(void) { return (*curFrame-frames.back().n) >= numFrDone; }
bool Trail::valid(void) { return size() >= minSize && frames.back().totDist >= (minLength*sc); }
int Trail::size(void) const { return frames.size(); }
bool Trail::operator <(const Trail& pt) const { return size() < pt.size(); }
vector<Point2f> Trail::getPoints(void) { return points; }
