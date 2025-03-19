import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

print("Launching modules and hardware...")

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)



import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')

        # Create publishers for RGB and Depth images
        self.rgb_publisher = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.depth_publisher = self.create_publisher(Image, '/camera/depth_registered/image_raw', 10)

        self.bridge = CvBridge()

        # Timer to publish images periodically
        self.timer = self.create_timer(0.1, self.publish_images)  # 10 Hz

    def publish_images(self):
        
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)

        print("Starting the stream...")
        frame = 0

        try:
            while True:

                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((color_image, depth_colormap))
                    # cv2.imwrite("./data/rgb/{}.png".format(frame), resized_color_image)
                    # publish the resized color image to the topic /camera/rbg/image_raw

                    # cv2.imwrite("./data/depth/{}.png".format(frame), depth_image) # or depth_colormap
                else:
                    images = np.hstack((color_image, depth_colormap))
                    # cv2.imwrite("./data/rgb/{}.png".format(frame), color_image)
                    # cv2.imwrite("./data/depth/{}.png".format(frame), depth_image) # or depth_colormap

                # Convert to ROS Image messages
                rgb_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
                depth_msg = self.bridge.cv2_to_imgmsg(depth_colormap, encoding='mono16')

                # Publish images
                self.rgb_publisher.publish(rgb_msg)
                self.depth_publisher.publish(depth_msg)

                self.get_logger().info('Published RGB & Depth Images')
                
                # # Show images
                # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('RealSense', images)
                # cv2.waitKey(100)

                frame += 1
                print(frame)
                
                if frame == 200: # stop the stream
                    break

        finally:

            # Stop streaming
            pipeline.stop()
                
        # # Load or capture an RGB image
        # rgb_image = cv2.imread('rgb_sample.jpg')  # Replace with camera capture if needed
        # if rgb_image is None:
        #     self.get_logger().warn("Could not load RGB image!")
        #     return

        # # Create a synthetic Depth image (16-bit single channel)
        # depth_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint16)

        

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
