import rosbag

from sensor_msgs.msg import PointCloud2
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

bag = rosbag.Bag('./static_slope_1hz.bag', 'r')


out_bag = rosbag.Bag('./output_pointcloud.bag', 'w')


for topic, msg, t in bag.read_messages():

	pointcloud_1 = msg.pointcloud_1


	if pointcloud_1:
	    out_bag.write('/pointcloud', pointcloud_1, t)


bag.close()
out_bag.close()

print("PointCloud2  'output_pointcloud.bag'")

