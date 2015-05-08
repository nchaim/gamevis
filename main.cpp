#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/core/core.hpp>
#include "Trail.h"
#include "MotionFeatures.h"

#include <iostream>
#include <vector>
#include <ctype.h>
#include <time.h>
#include <string>
#include <map>

using namespace cv;
using namespace cv::gpu;
using namespace std;

string bpath = ".";
struct dset_t { string file; int flags, start, end, cat; };
map<string, vector<dset_t>> dsets;
map<int, string> labels;
CvFont fontArial;
bool uFast = false;
int uAdv = 0, uJmp = 0;

bool loadVectors(string setName, Mat &fvec, Mat &cat);
int calcVectors(dset_t ds, Mat &fvec, Mat &cat);
void readIndex(void);
float clk(void);

void butCB(int state, void* userdata) {
	// TODO Add synchronization
	int id = (size_t) userdata;
	if (id == 0) uFast = (bool) state;
	else if (id == 1) uAdv = -1;
	else if (id == 2) uJmp = -100;
	else if (id == 3) uJmp = 100;
	else if (id == 4) uAdv = 1;
	else if (id == 5) exit(0);
}

void drawInfo(Mat &mat, stats_t st, int ntr, int pred, dset_t ds, int fr) {
	vector<string> ststr {
		ds.file,
		"[" + to_string(ds.start) + "-" + to_string(ds.end) + "]",
		"frame: " + to_string(fr), "",
		"accel x: " + to_string(st.ax),
		"accel y: " + to_string(st.ay),
		"vel x: " + to_string(st.vx),
		"vel y: " + to_string(st.vy),
		"curv: " + to_string(st.curv),
		"length: " + to_string(st.length),
		"dir: " + to_string(st.dir), "",
		"num trails: "  + to_string(ntr)
	};
	int vpos = 30;
	for(string &s : ststr) {
		putText(mat, s, Point(25, vpos), CV_FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(255, 255, 255), 1, CV_AA);
		vpos += 30;
	}
	String s = (pred == -1) ? "[...]" : labels[pred]; vpos += 70;
	putText(mat, s, Point(25, vpos), CV_FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(255, 255, 255), 2, CV_AA);
}

int main( int argc, const char** argv ) {
	string mode;
	for(int i = 1; i < argc; i++)
		if (argv[i][0] == '-') mode = &argv[i][1]; else bpath = argv[i];
	readIndex();

	if (mode == "vectors") {
		FileStorage fs(bpath+"/vectors", FileStorage::WRITE);
		for(auto &sn : {"train", "test"}) {
			Mat fvec, cat;
			for(dset_t &ds : dsets[sn]) {
				float ts = clk();
				int r = calcVectors(ds, fvec, cat);
				printf("vectors: %s [%d-%d] -> %d (%.2f fps)\n", ds.file.c_str(), ds.start,
						ds.end, r, (ds.end-ds.start+1)/(clk()-ts));
			}
			fs << (string(sn)+"__fvec") << fvec;
			fs << (string(sn)+"__cat") << cat;
		}
		return 0;

	} else if (mode == "train") {
		Mat fvec, cat;
		loadVectors("train", fvec, cat);
		printf("%d vectors loaded\n", fvec.rows);
		CvSVM svm; CvSVMParams prms;
		prms.svm_type    = CvSVM::C_SVC;
		prms.kernel_type = CvSVM::RBF;
		svm.train_auto(fvec, cat, Mat(), Mat(), prms);
		svm.save((bpath+"/model").c_str());
		return 0;

	} else if (mode == "test") {
		CvSVM svm;
		svm.load((bpath+"/model").c_str());
		Mat fvec, cat, pred;
		loadVectors("test", fvec, cat);
		printf("%d vectors loaded\n", fvec.rows);
		svm.predict(fvec, pred);
		int pass = 0;
		for(int i = 0; i < pred.rows; i++) {
			int pv = lround(pred.at<float>(i, 0));
			if (pv == cat.at<int>(i, 0)) pass++;
		}
		printf("Result: %d/%d (%f%%)\n", pass, pred.rows, (float)pass/pred.rows);
		return 0;
	}

	namedWindow("gamevis");
	createButton("Max FPS", butCB, (void *) 0, CV_CHECKBOX);
	createButton("<<", butCB, (void *) 1, CV_PUSH_BUTTON);
	createButton("<", butCB, (void *) 2, CV_PUSH_BUTTON);
	createButton(">", butCB, (void *) 3, CV_PUSH_BUTTON);
	createButton(">>", butCB, (void *) 4, CV_PUSH_BUTTON);
	createButton("Quit", butCB, (void *) 5, CV_PUSH_BUTTON);

	CvSVM svm;
	svm.load((bpath+"/model").c_str());

	VideoCapture cap;
	MotionFeatures mf;
	Mat frame, vis0, vis1, vis2;
	Mat fv, pred;
	float tfr = 0;

	int i = 0, di, cnt = dsets["demo"].size();
	while(true) {
		i = (cnt+i+di)%cnt; di = 1;
		dset_t &ds = dsets["demo"][i];

		cap.open((bpath+"/"+ds.file).c_str());
		float fps = cap.get(CV_CAP_PROP_FPS);
		if (isnan(fps)) fps = 29.965;
		cap.set(CV_CAP_PROP_POS_FRAMES, ds.start);
		mf.reset();

		int pv = -1;
		for(int fr = ds.start; fr <= ds.end; fr++) {
			if (!cap.read(frame)) break;
			if (ds.flags) { frame = frame.t(); flip(frame, frame, 1); }
			int h = frame.size().height, w = frame.size().width;
			if(mf.eval(frame, fv)) {
				svm.predict(fv, pred);
				pv = lround(pred.at<float>(0, 0));
			}
			do { waitKey(1); } while((!uFast) && (clk()-tfr) < (1.0/fps));
			if((clk()-tfr) >= (1.0/30)) {
				tfr = clk();
				mf.dbgBall(frame);
				vis0 = Mat::zeros(h, w, CV_8UC3);
				mf.dbgTrails(vis0);
				hconcat(frame, vis0, frame);
				vis1 = Mat(h, 300, CV_8UC3, CV_RGB(28, 43, 109));
				drawInfo(vis1, mf.lastStats(), mf.trCnt(), pv, ds, fr);
				hconcat(frame, vis1, frame);
				imshow("gamevis", frame);
			}
			if (uAdv != 0) { di = uAdv; uAdv = 0; break; }
			if (uJmp != 0) {
				fr = max(min(fr+uJmp, ds.end), ds.start);
				cap.set(CV_CAP_PROP_POS_FRAMES, fr);
				mf.reset(); uJmp = 0; pv = -1;
			}
		}
		cap.release();
	}
}

bool loadVectors(string setName, Mat &fvec, Mat &cat) {
	FileStorage fs(bpath+"/vectors", FileStorage::READ);
	if (!fs.isOpened()) return false;
	fs[setName+"__fvec"] >> fvec;
	fs[setName+"__cat"] >> cat;
	return true;
}

int calcVectors(dset_t ds, Mat &fvec, Mat &cat) {
	VideoCapture cap;
	MotionFeatures mf;
	Mat frame, fv; int vcnt = 0;
	cap.open((bpath+"/"+ds.file).c_str());
	cap.set(CV_CAP_PROP_POS_FRAMES, ds.start);
	for(int fr = ds.start; fr <= ds.end; fr++) {
		cap >> frame;
		if (ds.flags) { frame = frame.t(); flip(frame, frame, 1); }
		if(mf.eval(frame, fv)) {
			Mat rd(fv.rows, 1, CV_32SC1, Scalar_<int>(ds.cat));
			if (fvec.empty()) fvec = fv; else fvec.push_back(fv);
			if (cat.empty()) cat = rd; else cat.push_back(rd);
			vcnt += fv.rows;
		}
	}
	return vcnt;
}

void readIndex(void) {
	char name[100], fname[100];
	dset_t ds; int id;
	FILE *fp = fopen((bpath + "/index").c_str(), "r");
	if (!fp) return;
	while(!feof(fp)){
		if (fscanf (fp, "label %d %99s\n", &id, name))
			labels[id] = name;
		else if (fscanf (fp, "set %99s %99s %d %d %d %d\n", name, fname, &ds.flags, &ds.start, &ds.end, &ds.cat))
			{ ds.file = fname; dsets[name].push_back(ds); }
		else break;
	}
	fclose (fp);
}

float clk(void) {
	return (float) getTickCount()/getTickFrequency();
}
