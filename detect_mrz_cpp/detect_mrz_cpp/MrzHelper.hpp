//
//  MrzHelper.hpp
//  detect_mrz_cpp
//
//  Created by Nguyen Y Nguyen on 10/17/19.
//  Copyright © 2019 Nguyen Y Nguyen. All rights reserved.
//

#ifndef MrzHelper_hpp
#define MrzHelper_hpp

#include<opencv2/opencv.hpp>
#include<stdio.h>
#include<iostream>

using namespace std;
using namespace cv;

Mat findMRZ(Mat);
Mat resizeImage(Mat, int, int);
bool sortContours(vector<Point>, vector<Point>);
Mat sortBox(Mat);
vector<Point> sortVertices(vector<Point>);
vector<Point2f> sortVertices(vector<Point2f>);
Mat swap2Points(Mat, int, int);
vector<Point> swap2Points(vector<Point>, int, int);
vector<Point2f> swap2Points(vector<Point2f>, int, int);
vector<Point> mat2Vec(Mat);
vector<Point2f> mat2Vec2f(Mat);
double findDis(Point, Point);
Mat vec2Mat(vector<Point>);
Mat cropMinAreaRect(Mat, RotatedRect);
vector<Point2f> arr2Vec2f(Point2f[]);
vector<Point> vecf2vec(vector<Point2f>);
vector<Point2f> vec2vecf(vector<Point>);

#endif /* MrzHelper_hpp */
