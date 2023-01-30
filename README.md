# TXON_03

TXON Machine Learning internship tast 03 - Face Recognition Based Attendance System

Face Recognition is a broad challenge of verifying or identifying people in pictures or video.

I used OpenCV and face_recognition libraries for this.
I have made this project in python using OpenFace and dlib package.

Steps involved in this are:

Encode a picture using the HOG algorithm to create a simplified version of the image. Using this simplified image, find the part of the image that most looks like a generic HOG encoding of a face.

Figure out the pose of the face by finding the main landmarks in the face. Once we find those landmarks, use them to warp the image so that the eyes and mouth are centered.

Pass the centered face image through a neural network that knows how to measure features of the face. Save those 128 measurements.

Looking at all the faces we’ve measured in the past, see which person has the closest measurements to our face’s measurements. That’s our match!
