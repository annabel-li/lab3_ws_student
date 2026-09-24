#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import numpy as np 

LINEAR_SPEED = 0.2
ANGULAR_GAIN = 0.01 

def longestConnectedColour(imgRow, targetColour):
    """!
    @brief Finds the longest connected string of pixels in an image's row with pixel value equal to targetColour.

    @param imgRow An array representing the pixel values in the row of an image. Must be grayscale.
    @param targetColour An int representing the target colour.

    @return An array storing the index numbers of the longest connected strip of targetColour in that row.
    """
    maxLen = 0
    currentIndices = []
    longestString = []

    # if the current pixel is the target colour, add it to the list of indices
    for i in range(len(imgRow)):
        if imgRow[i] == targetColour:
            currentIndices.append(i)
        # else, we have either reached the end of the strip and should check for the strip's length
        else:
            if len(currentIndices) > maxLen:
                maxLen = len(currentIndices)
                longestString = currentIndices
            currentIndices = []  # reset regardless of whether it was the longest

    # save the list if it was the longest
    if len(currentIndices) > maxLen:
        longestString = currentIndices

    return longestString


def findLineCenter(processedFrame, targetColour): 

    # Get the image dimensions
    height, width = processedFrame.shape

    # start from bottom row, go upward
    for row in range(height - 1, -1, -1):
        # only worth checking rows that actually contain the colour
        if np.any(processedFrame[row] == targetColour):
            road = longestConnectedColour(processedFrame[row], targetColour)

            # get the x/y coordinates for the center of the circular marker
            cx = road[len(road) // 2]
            cy = row

            return cx, cy 

    return None, None 

class LineFollower(Node):

    def __init__(self):

        super().__init__('line_follower')
        self.bridge = CvBridge() #CvBridge handles the conversion of raw ROS2 data --> NumPy array 
        self.image_sub = self.create_subscription(Image, '/camera/image_raw',
                                                  self.callback, 1)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.trajectories = [] 

    def callback(self, data):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        height, width, _ = cv_image.shape 
        # Convert BGR to grayscale. 'grayFrame' is a new copy of the original frame, so the information about the original colours are preserved in 'frame'
        grayFrame = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
        # Threshold the image to make the road 'pop' --> use THRESH_BINARY_INV so pixels ABOVE threshold become black (0)
        _, thresh = cv.threshold(grayFrame, 60, 255, cv.THRESH_BINARY_INV)
        lineX, lineY = findLineCenter(thresh, 255) #find the center of the line 

        move = Twist() #creates a new empty "Twist" message object (https://ros2course.readthedocs.io/en/latest/Topics.html) 

        if lineX is not None: 

            imgCenter = width/2 
            diff = imgCenter - lineX
            self.trajectories.append(diff)
            move.linear.x = LINEAR_SPEED
            move.angular.z = ANGULAR_GAIN * diff 

        elif len(self.trajectories) > 0: 

            move.linear.x = LINEAR_SPEED 
            move.angular.z = ANGULAR_GAIN * self.trajectories[-1]

        else: 

            move.linear.x = 0 
            move.angular.z = 0 

        self.cmd_vel_pub.publish(move)



def main(args=None):
    print("Starting...")
    rclpy.init(args=args)
    lf = LineFollower()
    try:
        rclpy.spin(lf)
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        cv.destroyAllWindows()
        lf.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
