#! /usr/bin/env python

import sys
import numpy as np
import cv2

from imutils import face_utils
import argparse
import imutils
import dlib
import dippykit as dip

# size = 51
# blur = dip.windows.window_2d((size, size), window_type='gaussian', variance=100)
#
# faceCoords = np.array(
#     [[(0.24735554,0.10720145),(0.36570778,0.16026658)],
#     [(0.29790404,0.25025880),(0.41897747,0.30912185)],
#     [(0.18756560,0.66909546),(0.3116816, 0.72833866)],
#     [(0.33690974,0.39007744),(0.44652465,0.44727406)],
#     [(0.26383144,0.51849484),(0.38366628,0.57943654)],
#     [(0.03893245,0.12060487),(0.11762318,0.15940729)],
#     [(0.16116267,0.81741565),(.27264458,0.8778694)]])
# for faceCoord in faceCoords:
#     for tuples in faceCoord:
#         (tuples[0], tuples[1]) = (960*tuples[1], 600*tuples[0])
#
# faceCoords = np.uint32(faceCoords)
# print(faceCoords)
# print(blur)

def convolve2d(im1, im2, mode):
    # print(im1.shape)
    # print(im2.shape)
    res = dip.utilities.convolve2d(im1, im2, mode=mode)
    # print(res.shape)
    return res


def smooth(img, points):
    temp = np.zeros(img.shape)
    temp[:, :, :] = np.float64(img[:, :, :])
    size = 7
    blur = dip.windows.window_2d((size, size), window_type='gaussian', variance=10)
    points.append(points[0])
    for i in range(len(points) - 1):
        startY = min(points[i][0], points[i + 1][0])
        endY = max(points[i][0], points[i + 1][0])
        startX = min(points[i][1], points[i + 1][1])
        endX = max(points[i][1], points[i + 1][1])

        try:
            temp[startX:endX, startY:endY] = convolve2d(
                temp[startX - size // 2:endX + size // 2, startY - size // 2:endY + size // 2], blur, mode='valid')
        except:
            pass
        # for x in range(startX, endX):
        #     for y in range(startY, endY):
        #         for c in range(3):
        #             pass
        #             temp[x - size//2 : x + size//2, y - size//2 : y + size//2, c] = output[x,y,c] * blur
        #             #temp[x - size//2 : x + size//2, y - size//2 : y + size//2, c] = 0
    # img[tempmask] = temp[tempmask]; return img
    return temp


# Read points from text file
def readPoints(path):
    # Create an array of points.
    points = [];

    # Read points
    with open(path) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    return points


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


# calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # create subdiv
    subdiv = cv2.Subdiv2D(rect);

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList();

    delaunayTri = []

    pt = []

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            # Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
                        # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content-img',type=str,default='content_img.jpg',help='The content image')
    args = parser.parse_args()
    # Make sure OpenCV is version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        print >> sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
        sys.exit(1)

    # Read images
    filename1 = 'pratt.jpg'
    filename2 = 'hemsworth.jpg'
    swap = False
    (filename1, filename2) = (filename2, filename1) if swap else (filename1, filename2)

    img1 = cv2.imread(filename1);
    img2 = cv2.imread(filename2);
    #print(f'img2 size {img2.shape}')
    img1Warped = np.copy(img2);

    # Read array of corresponding points
    # points1 = readPoints(filename1 + '.txt')
    # points2 = readPoints(filename2 + '.txt')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects1s = detector(gray1, 1)
    rects2s = detector(gray2, 1)
    if len(rects2s):
        rects2 = rects2s[0]
        fname = 'content_img.jpg'
        cv2.imwrite(fname, img2[rects2.top():rects2.bottom(), rects2.left():rects2.right(), :])

    shape1 = predictor(gray1, rects1s[0])
    shape1 = face_utils.shape_to_np(shape1)

    # img2temp = np.zeros(img2.shape)
    img2temp = img2[:,:,:]
    # for faceCoord in faceCoords:
    for n, rects2 in enumerate(rects2s):
        try:
            #rects2 = dlib.rectangle(int(faceCoord[0][0]), int(faceCoord[0][1]), int(faceCoord[1][0]), int(faceCoord[1][1]))
            shape2 = predictor(gray2, rects2)
            shape2 = face_utils.shape_to_np(shape2)

            points1 = []
            points2 = []

            indices = [i for i in range(17)]
            indices += [i for i in range(17, 27)][::-1]
            # indices += [i for i in range(17, 42)][::-1]
            # Read points
            for i in indices:
                x, y = (shape1[i, 0], shape1[i, 1])
                points1.append((int(x), int(y)))

                x, y = (shape2[i, 0], shape2[i, 1])
                points2.append((int(x), int(y)))

            # Find convex hull
            hull1 = []
            hull2 = []

            hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

            for i in range(0, len(hullIndex)):
                hull1.append(points1[int(hullIndex[i])])
                hull2.append(points2[int(hullIndex[i])])

            # Find delanauy traingulation for convex hull points
            sizeImg2 = img2.shape
            rect = (0, 0, sizeImg2[1], sizeImg2[0])

            dt = calculateDelaunayTriangles(rect, hull2)

            if len(dt) == 0:
                quit()

            # Apply affine transformation to Delaunay triangles
            for i in range(0, len(dt)):
                t1 = []
                t2 = []

                # get points for img1, img2 corresponding to the triangles
                for j in range(0, 3):
                    t1.append(hull1[dt[i][j]])
                    t2.append(hull2[dt[i][j]])

                warpTriangle(img1, img1Warped, t1, t2)

            # Calculate Mask
            hull8U = []
            for i in range(0, len(hull2)):
                hull8U.append((hull2[i][0], hull2[i][1]))

            mask = np.zeros(img2.shape, dtype=img2.dtype)

            cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

            r = cv2.boundingRect(np.float32([hull2]))
            r1 = cv2.boundingRect(np.float32([hull1]))

            center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

            # Clone seamlessly.
            output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

            img2temp[mask==255] = output[mask==255]
            fname = 'affineSwap' + str(n) + '.jpg'
            fname = 'style_img.jpg'
            cv2.imwrite(fname, img2temp[rects2.top():rects2.bottom(), rects2.left():rects2.right(), :])
            #print(rects2.top(), rects2.bottom(), rects2.left(), rects2.right())
            print('Wrote image ' + str(n))
            # smooth normalize lighting
            # output = np.uint8(smooth(output, points2))
        except Exception as e:
            print(e)

    cv2.imshow("Face Swapped", img2temp)
    #cv2.imwrite('test.jpg', img2temp)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

