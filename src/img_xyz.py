import rospy

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy
import tf
# from math import atan2, pi, sqrt
import math
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from  std_msgs.msg import Int32MultiArray 
import image_geometry
import sys


class pixel_converter():
    def __init__(self) -> None:
        self.tf_listener = tf.TransformListener()
        self.camera_model = None
        self.bridge = CvBridge()
        self.cam_info = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.info_callback)
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.Image_callback)
        self.pcl_sub = rospy.Subscriber(
            "/camera/depth/points", PointCloud2, self.pcl_callback)
        self.road_pub = rospy.Publisher("road",Int32MultiArray,)

    def pcl_callback(self, data):
        self.pc = data

    def info_callback(self, data):
        self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(data)
        self.cam_info.unregister()  

    def Image_callback(self, data): 

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        img = cv_image
        face_cascade = cv.CascadeClassifier('/home/yashas/catkin_ws/src/img_to_xyz/src/face_detect_cascade.xml')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) # Stores all faces. (Check face_detect.py for refrence)
        for (x, y, w, h) in faces:
            try:
                center_x = (x+(x+w))//2
                center_y = (y+(y+h))//2
            except:
                center_x = 0
                center_y = 0
            self.conver_to_cords(center_x,center_y) # to convert to pixel to xyz       
    def get_depth(self, x, y): # Get depth from x,y values
        gen = pc2.read_points(self.pc, field_names="z", 
                              skip_nans=True, uvs=[(x, y)])
        # print(gen)
        return next(gen)

    def conver_to_cords(self, pixel_x, pixel_y):
        try:
            depth = self.get_depth(pixel_x, pixel_y)
        except StopIteration:
            depth = (0,)
        v = self.camera_model.projectPixelTo3dRay((pixel_x, pixel_y))
        d_cam = numpy.concatenate(
            (depth*numpy.array(v), numpy.ones(1))).reshape((4, 1))
        self.tf_listener.waitForTransform(
            '/odom', '/camera_depth_frame', rospy.Time(), rospy.Duration(4)) #Transforms listner

        (trans, rot) = self.tf_listener.lookupTransform(
            '/odom', '/camera_depth_frame', rospy.Time()) #transform function
        camera_to_base = tf.transformations.compose_matrix(
            translate=trans, angles=tf.transformations.euler_from_quaternion(rot))
        d_base = numpy.dot(camera_to_base, d_cam) # Stores to X , Y , Z values.
        return d_base[0:3][0][0], d_base[0:3][1][0] 
    


def main(args):
    rospy.init_node('image_converter', anonymous=True)
    ic = pixel_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
