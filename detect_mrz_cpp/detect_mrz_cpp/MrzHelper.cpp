//
//  MrzHelper.cpp
//  detect_mrz_cpp
//
//  Created by Nguyen Y Nguyen on 10/17/19.
//  Copyright © 2019 Nguyen Y Nguyen. All rights reserved.
//

#include "MrzHelper.hpp"

Mat findMRZ(Mat original) {
    Mat rectKernel = getStructuringElement(MORPH_RECT, Size(13, 5));
    Mat sqKernel = getStructuringElement(MORPH_RECT, Size(21, 21));
    
    Mat img = original.clone();
    img = resizeImage(img, img.cols * 600 / img.rows, 600);
    
    Mat gray = img.clone();
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(3, 3), 0.0);
    
    Mat blackHat = gray.clone();
    morphologyEx(gray, blackHat, MORPH_BLACKHAT, rectKernel);
    
    Mat gradX = blackHat.clone();
    Sobel(blackHat, gradX, CV_32F, 1, 0, -1);
    absdiff(gradX, Mat::zeros(gradX.rows, gradX.cols, CV_32F), gradX);
    double minVal, maxVal;
    minMaxLoc(gradX, &minVal, &maxVal);
    
    Mat gradXFloat = gradX.clone();
    Mat maskMin(gradXFloat.rows, gradXFloat.cols, CV_32F, Scalar(minVal));
    subtract(gradXFloat, maskMin, gradXFloat);
    Mat maskDiff(gradXFloat.rows, gradXFloat.cols, CV_32F, Scalar(maxVal - minVal));
    divide(gradXFloat, maskDiff, gradXFloat);
    Mat mask255(gradXFloat.rows, gradXFloat.cols, CV_32F, Scalar(255.0));
    multiply(gradXFloat, mask255, gradXFloat);
    gradXFloat.convertTo(gradX, CV_8UC1);
    
    morphologyEx(gradX, gradX, MORPH_CLOSE, rectKernel);
    
    Mat thresh;
    threshold(gradX, thresh, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
    morphologyEx(thresh, thresh, MORPH_CLOSE, sqKernel);
    Mat nullKernel;
    erode(thresh, thresh, nullKernel, Point(-1, -1), 4);
    
    // padding
    int width = thresh.cols;
    int height = thresh.rows;
    int p = (int) (width * 0.01);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < p; j++) {
            thresh.at<uchar>(i, j, 0) = 0;
        }
        for (int l = width - p; l < width; l++) {
            thresh.at<uchar>(i, l, 0) = 0;
        }
    }
    // end padding
    
    // denoise - clear all connections from mrz to others horizontal
    for (int i = 0; i < height; i++) {
        int countWhite = 0;
        int startIdx = 0;
        for (int j = 0; j < width; j++) {
            if (thresh.at<uchar>(i, j, 0) == 255) { // white
                countWhite += 1;
            } else if (countWhite > 0 && countWhite < 80) {
                for (int k = startIdx; k < j + 1; k++) {
                    thresh.at<uchar>(i, k, 0) = 0;
                }
                startIdx = j;
                countWhite = 0;
            }
        }
    }
    // end denoise
    
    // connect nearby contours
    Mat denoiseKernel = Mat::ones(3, 3, CV_8U);
    dilate(thresh, thresh, denoiseKernel);
    // end connect
    
    // find contours in the threshold image and sort them by their size
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thresh.clone(), contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));
    sort(contours.begin(), contours.end(), sortContours);
    // end contours finding
    
    // filter not valid contours with wrong aspect ratio from with and height
    vector<vector<Point>> newContours;
    for (vector<Point> c : contours) {
        Rect rect = boundingRect(c);
        if (rect.width * 1.0 / rect.height > 6.0) {
            newContours.push_back(c);
        }
    }
    contours = newContours;
    // end filter
    
    // detect minarearect
    if (contours.size() > 0) {
        vector<Point2f> cnt0 = vec2vecf(contours[0]);
        
        
        RotatedRect rect0 = minAreaRect(cnt0);
        Point2f box0a[4];
        rect0.points(box0a);
        vector<Point2f> box0f = arr2Vec2f(box0a);
        box0f = sortVertices(box0f);
        vector<Point> box0 = vecf2vec(box0f);
        
        double w0 = rect0.size.width;
        double h0 = rect0.size.height;
        double angle = rect0.angle;
        
        vector<vector<Point>> boxList;
        vector<Point> mops;
        if (contours.size() > 1) {
            vector<Point2f> cnt1 = vec2vecf(contours[1]);
            
            RotatedRect rect1 = minAreaRect(cnt1);
            Point2f box1a[4];
            rect0.points(box1a);
            vector<Point2f> box1f = arr2Vec2f(box1a);
            box1f = sortVertices(box1f);
            vector<Point> box1 = vecf2vec(box1f);
            
            double w1 = rect1.size.width;
            double h1 = rect1.size.height;
            
            if (abs(w1 - w0) / w0 < 0.1 && abs(h1 - h0) / h0 < 0.2) {
                if (box0[1].y < box1[1].y) {
                    mops.push_back(box0[1]);
                    mops.push_back(box0[3]);
                    mops.push_back(box1[2]);
                    mops.push_back(box1[0]);
                } else {
                    mops.push_back(box1[1]);
                    mops.push_back(box1[3]);
                    mops.push_back(box0[2]);
                    mops.push_back(box0[0]);
                }
            } else {
                mops.push_back(box0[0]);
                mops.push_back(box0[1]);
                mops.push_back(box0[3]);
                mops.push_back(box0[2]);
            }
        } else {
            mops.push_back(box0[0]);
            mops.push_back(box0[1]);
            mops.push_back(box0[3]);
            mops.push_back(box0[2]);
        }
        boxList.push_back(mops);
        
        // padding box
        vector<Point> outerPoints = sortVertices(boxList[0]);
        double w = findDis(outerPoints[0], outerPoints[2]);
        double h = findDis(outerPoints[0], outerPoints[1]);
        
        int pX = (int) (w * 0.03);
        
        // bottom left
        outerPoints[0] = Point(outerPoints[0].x - pX, outerPoints[0].y + pX);
        // top left
        outerPoints[1] = Point(outerPoints[1].x - pX, outerPoints[1].y - pX);
        // bottom right
        outerPoints[2] = Point(outerPoints[2].x + pX, outerPoints[2].y + pX);
        // top right
        outerPoints[3] = Point(outerPoints[3].x + pX, outerPoints[3].y - pX);
        
        boxList.clear();
        boxList.push_back(outerPoints);
        
        double scale = original.cols * 1.0 / img.cols;
        Point center = Point((outerPoints[0].x + outerPoints[3].x) / 2 * scale, (outerPoints[0].y + outerPoints[3].y) / 2 * scale);
        RotatedRect minRect = RotatedRect(center, Size((w + pX * 2) * scale, (h + pX * 2) * scale), angle);

        return cropMinAreaRect(original.clone(), minRect);
    }
    
    // end detect
    return thresh;
}

Mat cropMinAreaRect(Mat image, RotatedRect rect) {
    Point center = rect.center;
    int width = rect.size.width < image.size().width ? rect.size.width : image.size().width;
    int height = rect.size.height < image.size().height ? rect.size.height : image.size().height;
    double theta = rect.angle;

    if (theta < -45) {
        theta += 90;
    }

    Size shape = Size(image.size().width, image.size().height);
    Mat matrix = getRotationMatrix2D(center, theta, 1);

    Mat img = Mat();
    warpAffine(image, img, matrix, shape);

    int x = (center.x - width / 2) > 0 ? (center.x - width / 2) : 0;
    int y = (center.y - height / 2) > 0 ? (center.y - height / 2) : 0;
    int right = (x + width) < image.size().width ? (x + width) : image.size().width;
    int bottom = (y + height) < image.size().height ? (y + height) : image.size().height;
    
    Mat roi = img.colRange(x, right).rowRange(y, bottom);
    return roi;
}

double findDis(Point p1, Point p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

Mat vec2Mat(vector<Point> vec) {
    return Mat(vec);
}

vector<Point> mat2Vec(Mat mat) {
    vector<Point> res;
    for (int i = 0; i < mat.rows; i++) {
        res.push_back(Point(mat.at<uchar>(i, 0, 0), mat.at<uchar>(i, 1, 0)));
    }
    return res;
}

vector<Point2f> mat2Vec2f(Mat mat) {
    vector<Point2f> res;
    for (int i = 0; i < mat.rows; i++) {
        res.push_back(Point(mat.at<float>(i, 0, 0), mat.at<uchar>(i, 1, 0)));
    }
    return res;
}

vector<Point2f> arr2Vec2f(Point2f box[]) {
    vector<Point2f> res;
    for(int i = 0; i < 4 ; i++) {
        res.push_back(box[i]);
    }
    return res;
}

Mat sortBox(Mat box) {
    // box is [0, 1, 2, 3]
    if (box.at<uchar>(0, 0, 0) < box.at<uchar>(2, 0, 0)) {
        if (box.at<uchar>(0, 1, 0) > box.at<uchar>(2, 1, 0)) { // [0, 1, 2, 3] --> [0, 1, 3, 2]
            box = swap2Points(box, 2, 3); // [0, 1, 3, 2]
        } else { // [0, 1, 2, 3] --> [3, 0, 2, 1]
            box = swap2Points(box, 0, 3); // [3, 1, 2, 0]
            box = swap2Points(box, 1, 3); // [3, 0, 2, 1]
        }
    } else if(box.at<uchar>(0, 1, 0) > box.at<uchar>(2, 1, 0)) { // [0, 1, 2, 3] --> [1, 2, 0, 3]
        box = swap2Points(box, 0, 1); // [1, 0, 2, 3]
        box = swap2Points(box, 1, 2); // [1, 2, 0, 3]
    } else { // [0, 1, 2, 3] --> [3, 0, 2, 1]
        box = swap2Points(box, 3, 0); // [3, 1, 2, 0]
        box = swap2Points(box, 1, 3); // [3, 0, 2, 1]
    }
    return box;
}

vector<Point2f> sortVertices(vector<Point2f> box) {
    // box is [0, 1, 2, 3]
    if (box[0].x < box[2].x) {
        if (box[0].y > box[2].y) { // [0, 1, 2, 3] --> [0, 1, 3, 2]
            box = swap2Points(box, 2, 3); // [0, 1, 3, 2]
        } else { // [0, 1, 2, 3] --> [3, 0, 2, 1]
            box = swap2Points(box, 0, 3); // [3, 1, 2, 0]
            box = swap2Points(box, 1, 3); // [3, 0, 2, 1]
        }
    } else if(box[0].y > box[2].y) { // [0, 1, 2, 3] --> [1, 2, 0, 3]
        box = swap2Points(box, 0, 1); // [1, 0, 2, 3]
        box = swap2Points(box, 1, 2); // [1, 2, 0, 3]
    } else { // [0, 1, 2, 3] --> [3, 0, 2, 1]
        box = swap2Points(box, 3, 0); // [3, 1, 2, 0]
        box = swap2Points(box, 1, 3); // [3, 0, 2, 1]
    }
    return box;
}

vector<Point> sortVertices(vector<Point> box) {
    // box is [0, 1, 2, 3]
    if (box[0].x < box[2].x) {
        if (box[0].y > box[2].y) { // [0, 1, 2, 3] --> [0, 1, 3, 2]
            box = swap2Points(box, 2, 3); // [0, 1, 3, 2]
        } else { // [0, 1, 2, 3] --> [3, 0, 2, 1]
            box = swap2Points(box, 0, 3); // [3, 1, 2, 0]
            box = swap2Points(box, 1, 3); // [3, 0, 2, 1]
        }
    } else if(box[0].y > box[2].y) { // [0, 1, 2, 3] --> [1, 2, 0, 3]
        box = swap2Points(box, 0, 1); // [1, 0, 2, 3]
        box = swap2Points(box, 1, 2); // [1, 2, 0, 3]
    } else { // [0, 1, 2, 3] --> [3, 0, 2, 1]
        box = swap2Points(box, 3, 0); // [3, 1, 2, 0]
        box = swap2Points(box, 1, 3); // [3, 0, 2, 1]
    }
    return box;
}


vector<Point2f> swap2Points(vector<Point2f> src, int idx1, int idx2) {
    Point2f point = src[idx1];
    src[idx1] = src[idx2];
    src[idx2] = point;
    return src;
}

vector<Point> swap2Points(vector<Point> src, int idx1, int idx2) {
    Point point = src[idx1];
    src[idx1] = src[idx2];
    src[idx2] = point;
    return src;
}

Mat swap2Points(Mat src, int idx1, int idx2) {
    Mat dst = src.clone();
    
    double tmp0 = dst.at<uchar>(idx1, 0, 0);
    double tmp1 = dst.at<uchar>(idx1, 1, 0);
    
    dst.at<uchar>(idx1, 0) = dst.at<uchar>(idx2, 0, 0);
    dst.at<uchar>(idx1, 1) = dst.at<uchar>(idx2, 1, 0);
    
    dst.at<uchar>(idx2, 0) = tmp0;
    dst.at<uchar>(idx2, 1) = tmp1;
    
    return dst;
}

bool sortContours(vector<Point> a, vector<Point> b) {
    return boundingRect(a).width > boundingRect(b).width;
}

Mat resizeImage(Mat src, int newWidth, int newHeight) {
    Mat dst;
    Size size(newWidth, newHeight);
    resize(src, dst, size);
    return dst;
}

vector<Point> vecf2vec(vector<Point2f> src) {
    vector<Point> res;
    for (int i = 0; i < src.size(); i++) {
        res.push_back(Point(src[i].x, src[i].y));
    }
    return res;
}

vector<Point2f> vec2vecf(vector<Point> src) {
    vector<Point2f> res;
    for (int i = 0; i < src.size(); i++) {
        res.push_back(Point2f(src[i].x, src[i].y));
    }
    return res;
}
