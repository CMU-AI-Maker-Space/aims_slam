import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()


cfg = pipeline.start() # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
print(intr)

#pepper_nano@pepper-nano:~/aims-slam$ python3 get_rs_intrinsics.py 
#425.9246520996094
#pepper_nano@pepper-nano:~/aims-slam$ python3 get_rs_intrinsics.py 
#[ 848x480  p[425.925 236.515]  f[423.813 423.813]  Brown Conrady [0 0 0 0 0] ]
