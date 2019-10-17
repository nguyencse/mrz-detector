//
//  main.cpp
//  detect_mrz_cpp
//
//  Created by Nguyen Y Nguyen on 10/15/19.
//  Copyright © 2019 Nguyen Y Nguyen. All rights reserved.
//

#include<opencv2/opencv.hpp>
#include<iostream>
#include "MrzHelper.hpp"

int main() {
    
    string fileName;
    cout << "Enter file name (with extension): ";
    cin >> fileName;
    
    Mat img = imread(fileName);
    imshow(fileName, img);
    
    Mat roi = findMRZ(img);
    imshow("roi", roi);
    
    waitKey(0);
    return 0;
}
