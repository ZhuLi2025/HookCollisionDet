#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>

#include <iostream>
#include <string>
#include <string.h>
#include <stdio.h>

#include "HookLoadPositionAcquirer.h"
#include "ICPTracker.h"
#include "CSF/CSF.h"
#include "CSF/point_cloud.h"
#include <chrono>
#include <pcl/common/transforms.h>
#include "CollisionDetector.h"
#include "constants.h"
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#define SAVE_CLOUD 1

using namespace std;

class LivoxReader {
    public:
    LivoxReader():accumulated_cloud(new pcl::PointCloud<pcl::PointXYZI>())
    {
        // accumulated_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        // 订阅Livox发布的PointCloud2话题（根据实际驱动配置调整）
        sub_ = nh_.subscribe("/livox/lidar", 10, &LivoxReader::cloudCallback, this);
        
        // 可选：发布处理后的点云
        pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/processed_cloud", 1);
        
        ROS_INFO("Livox reader node initialized");
    }

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) 
    {   
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

        // pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        // 将ROS的PointCloud2消息转换为PCL点云对象
        pcl::fromROSMsg(*msg, *cloud);

        // 累加当前点云到全局累加点云
        *accumulated_cloud += *cloud;

        //分布出来
        // 发布处理后的点云
        num_count--;

        string pcdFilePath;

        if (num_count <=0)
        {               
            pcl::toROSMsg(*accumulated_cloud, output);
            output.header = msg->header;
            
            pub_.publish(output);

            num_count =30;

            cout<<"publish  lidar data!!!!!!!!!"<<endl;

            // 保存点云为PCD文件
            i++;

            pcdFilePath ="/home/xpp/pcd/" + std::to_string(i) + ".pcd";

            #if SAVE_CLOUD
                if (pcl::io::savePCDFileASCII(pcdFilePath, *accumulated_cloud) == -1) {
                    PCL_ERROR("Couldn't write file test_pcd.pcd\n");
                }
            #endif
            
            accumulated_cloud->clear();
        }

        // 方法1：直接访问原始数据（快速但需了解字段布局）
        //processRawData(*msg);
        
        // 方法2：转换为PCL点云（更易操作但增加开销）
        //processPCLData(*msg);
    }

    private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;

    // 定义全局累加点云指针
    sensor_msgs::PointCloud2 output;
    pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud;
    int num_count  =30;

    int i=0;


    // 方法1：直接解析PointCloud2数据
    void processRawData(const sensor_msgs::PointCloud2& cloud) 
    {
        int point_step = cloud.point_step;  // 每个点的字节步长
        int offset_x = 0, offset_y = 4, offset_z = 8;  // XYZ字段偏移（需确认）
        
        // 遍历所有点
        for (size_t i = 0; i < cloud.width * cloud.height; ++i) 
        {
            const uint8_t* data_ptr = &cloud.data[i * point_step];
            
            // 提取XYZ坐标（假设float32格式）
            float x = *reinterpret_cast<const float*>(data_ptr + offset_x);
            float y = *reinterpret_cast<const float*>(data_ptr + offset_y);
            float z = *reinterpret_cast<const float*>(data_ptr + offset_z);
            
            // 示例：打印前5个点
            if (i < 5) {
                ROS_INFO_STREAM("Point " << i << ": (" << x << ", " << y << ", " << z << ")");
            }
        }
    }

    // 方法2：转换为PCL点云处理
    void processPCLData(const sensor_msgs::PointCloud2& cloud) 
    {
        pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
        pcl::fromROSMsg(cloud, pcl_cloud);
        
        // 示例：滤波处理（需包含pcl/filters/voxel_grid.h）
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>(pcl_cloud));
        pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
        voxel_filter.setInputCloud(cloud_ptr);
        voxel_filter.setLeafSize(0.05f, 0.05f, 0.05f);  // 5cm体素网格
        voxel_filter.filter(*cloud_ptr);
        
        // 发布处理后的点云
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud_ptr, output);
        output.header = cloud.header;
        pub_.publish(output);
    }
};

int main(int argc, char** argv) {

    ros::init(argc, argv, "livox_reader_node");
    ros::NodeHandle nh("~");
    int hook_load_position_acquisition_way;
    std::string init_pcd_file,test_pcd_files_dir;

    if(!nh.getParam("test_pcd_files_dir",test_pcd_files_dir)){
        ROS_ERROR("cant get test pcd files dir");
        return 0;
    }
    if(!nh.getParam("init_pcd_file",init_pcd_file)){
        ROS_ERROR("cant get test pcd file");
        return 0;
    }
    // LivoxReader reader;
    // ros::spin();
//================================ 初始化构建 ===================================
    ROS_INFO("================================ Initialization ===================================\n");
    //创建吊钩吊载位置获取类对象
    auto hookLoadPositionAcquirer =std::make_shared<HookLoadPositionAcquirer<pcl::PointXYZ>>();
    ClusterInfo<pcl::PointXYZ> targetClusterInfo_last_frame;
    ClusterInfo<pcl::PointXYZ> hookClusterInfo;
    ClusterInfo<pcl::PointXYZ> loadClusterInfo;
    bool b_hookload_position_acquisition_successed;

    //基于ipc的帧间点云匹配实现目标点云跟踪
    auto icpTrack =std::make_shared<ICPTracker<pcl::PointXYZ>>();

    //创建碰撞检测类对象
    auto collisionDetector = std::make_shared<CollisionDetector<pcl::PointXYZ>>();

    pcl::PointCloud<pcl::PointXYZ>::Ptr init_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(init_pcd_file, *init_cloud) == -1)
    {
        PCL_ERROR("Couldn't read the pcd file\n");
        return false;
    }

    auto start0 = std::chrono::high_resolution_clock::now();

//================================ 地毯滤波 ===================================
    ROS_INFO("================================ CSF ===================================\n");
    // 坐标变换（CSF要求-Z轴方向为重力方向，改变点云坐标系）
    Eigen::Vector3f current_gravity(1.0f, 0.0f, 0.0f);        // 当前坐标系的重力方向：X轴（即(1, 0, 0)）
    Eigen::Vector3f target_gravity(0.0f, 0.0f, -1.0f);        // 目标坐标系的重力方向：Z轴（即(0, 0, 1)）
    Eigen::Vector3f axis = current_gravity.cross(target_gravity);        // 计算旋转轴：即current_gravity与target_gravity的叉积
    axis.normalize();
    float angle = acos(current_gravity.dot(target_gravity));        // 计算旋转角度：即current_gravity与target_gravity的点积反余弦
    Eigen::Matrix3f rotation_matrix;
    rotation_matrix = Eigen::AngleAxisf(angle, axis); // 使用角轴表示法
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(rotation_matrix);

    // 旋转点云
    auto start = std::chrono::high_resolution_clock::now();
    // 使用 pcl::transformPointCloud 进行坐标变换
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*init_cloud, *transformed_cloud, transform);

    auto end_coord_transform = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_coord_transform - start;
    std::cout << "【time】coord transform took " << elapsed.count() << " seconds." << std::endl;
    std::vector<int> ground_indices, non_ground_indices;

    #ifdef CSF_DATA_TYPE //使用CSF源码中的数据类型
        //--- CSF 地面分割 ---
        //1. 转换为CSF库的点云格式
        csf::PointCloud csf_cloud;
        csf_cloud.reserve(transformed_cloud->size());
        #pragma omp parallel for
        for (const auto& pt : transformed_cloud->points) {
            csf_cloud.emplace_back(pt.x, pt.y, pt.z);
        }
        // 2. 创建CSF对象并设置参数（由rosparam控制）
        CSF csf;
        csf.setPointCloud(csf_cloud);

        csf.params.bSloopSmooth = true;
        csf.params.class_threshold = 0.3;
        csf.params.cloth_resolution = 0.75;
        csf.params.interations = 500;
        csf.params.rigidness = 3;
        csf.params.time_step = 0.65;
        // 3. 调用CSF分割
        csf.do_filtering(ground_indices, non_ground_indices, false);
                            auto end_CSF = std::chrono::high_resolution_clock::now();
        elapsed = end_CSF - end_coord_transform;
        std::cout << "【time】CSF  took " << elapsed.count() << " seconds." << std::endl;

    #else //使用pcl的数据类型
        CSF csf;
        csf.setPointCloud(transformed_cloud);

        csf.params.bSloopSmooth = true;
        csf.params.class_threshold = 0.3;
        csf.params.cloth_resolution = 0.75;
        csf.params.interations = 500;
        csf.params.rigidness = 3;
        csf.params.time_step = 0.65;

        csf.do_filtering_v2(ground_indices, non_ground_indices, false);
        auto end_CSF = std::chrono::high_resolution_clock::now();
        elapsed = end_CSF - end_coord_transform;
        std::cout << "【time】CSF took " << elapsed.count() << " seconds." << std::endl;
    #endif

    // 提取地面/非地面点云
    pcl::PointIndices::Ptr ground_idx(new pcl::PointIndices);
    pcl::PointIndices::Ptr obstacle_idx(new pcl::PointIndices);
    ground_idx->indices = ground_indices;
    obstacle_idx->indices = non_ground_indices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr off_ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(transformed_cloud);
    extract.setIndices(ground_idx);
    extract.setNegative(false);
    extract.filter(*ground_cloud);
    extract.setIndices(obstacle_idx);
    extract.setNegative(false);
    extract.filter(*off_ground_cloud);

    // 计算逆变换矩阵（后续吊钩吊载检测坐标系为未转换的坐标系，因此转换回去）
    Eigen::Affine3f inverse_transform = transform.inverse();
    pcl::transformPointCloud(*off_ground_cloud,*off_ground_cloud,inverse_transform);
    
//================================ 吊钩吊载识别 ===================================
    ROS_INFO("================================ Hook/Load Detection ===================================\n");
    //定义已知参数
    float rope_len{63.0}; //卷扬绳下放长度
    bool load_exist_flag{false}; //吊钩存在标志为
    b_hookload_position_acquisition_successed = hookLoadPositionAcquirer->getHookLoadCluster(off_ground_cloud,rope_len,load_exist_flag,hookClusterInfo,loadClusterInfo);
    Mode curFrameTargetDetMode ;
    if(b_hookload_position_acquisition_successed){
        ROS_INFO("[Hook / Load detection] find load or hook cluster");
        if(loadClusterInfo.exist()){
            targetClusterInfo_last_frame =  loadClusterInfo;
            curFrameTargetDetMode = Mode::TrackMode;
        }
        else if(hookClusterInfo.exist()){
            targetClusterInfo_last_frame =  hookClusterInfo;
            curFrameTargetDetMode = Mode::TrackMode;
        }
        else{
            curFrameTargetDetMode = Mode::DetectMode;
        }
    }
    else{
        ROS_INFO("[Hook / Load detection] cant find load or hook cluster");
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> eps  = end - start0;
    ROS_INFO("hood det took time : %f",eps.count());


    //可视化
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

//================================ 吊钩吊载跟踪 ===================================
    ROS_INFO("================================ Hook/Load Track ===================================\n");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // 读取目录下所有 pcd 文件
    vector<std::string> pcd_files;
    for (const auto& entry : fs::directory_iterator(test_pcd_files_dir)) {
        if (entry.path().extension() == ".pcd") {
            pcd_files.push_back(entry.path().string());
        }
    }

    std::sort(pcd_files.begin(), pcd_files.end());
    if (pcd_files.empty()) {
        ROS_ERROR("No PCD files in dir: %s", test_pcd_files_dir.c_str());
        return 0;
    }

    ROS_INFO("Found %ld pcd files.", pcd_files.size());

    int idx = 0;
    ClusterInfo<pcl::PointXYZ> targetClusterInfo_cur_frame;
    while (1)
    {
        string pcdFilePath = pcd_files[idx];

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcdFilePath, *cloud) == -1) 
        {
            ROS_ERROR("Cannot read PCD: %s", pcdFilePath.c_str());
            idx = (idx + 1) % pcd_files.size();
            continue;
        }

        auto start_track = std::chrono::high_resolution_clock::now();

    // step 1 IPC 匹配吊钩模板点云 与 当前帧点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr env_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if(!icpTrack->targetTrack(cloud,targetClusterInfo_last_frame,targetClusterInfo_cur_frame,env_cloud)){
            ROS_INFO("icp track failed");
            continue;
        }
        targetClusterInfo_last_frame = targetClusterInfo_cur_frame; //深拷贝上帧的吊钩点云识别结果作为下一帧的匹配模板，如果识别错误则后续全部错误
    
   
    // step 2 Collision Detecte
        auto start_detection = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eps_track  = start_detection - start_track;
        ROS_INFO("track took time : %f",eps_track.count());

        targetClusterInfo_last_frame.computeOBB();

        Eigen::Vector3f dir;//障碍物到目标方向向量
        float min_dist;
        collisionDetector->computeMinDistance(env_cloud, targetClusterInfo_last_frame.obb, min_dist,dir, 0.2f, 4 );
        if (std::isfinite(min_dist)) {
            ROS_INFO("[Collision Detecte] Min distance = %f m ; Direction (unit) = [ %f , %f , %f ]",min_dist,dir.x(),dir.y(),dir.z());
        } else {
            ROS_INFO("[Collision Detecte] No nearest point found or env cloud empty.");
        }
        auto end_detection = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> eps_detection  = end_detection - start_detection;
        ROS_INFO("detection took time : %f",eps_detection.count());


        //碰撞分析
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();

        // 吊钩周围区域
        // viewer->addCube(bbMin[0], bbMax[0], bbMin[1], bbMax[1], bbMin[2], bbMax[2], 1.0, 1.0, 0.0, "cube1");
        // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 
        //                                     pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "cube1");

        //绘制场景点云
        viewer->addPointCloud<pcl::PointXYZ>(cloud, "current_frame");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, "current_frame");
        
        // // 添加立方体到可视化窗口
        // viewer->addCube(min_point[0], max_point[0], min_point[1], max_point[1], min_point[2], max_point[2], 0.0, 1.0, 0.0, "cube");
        // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 
        //                                     pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "cube");

        //吊钩点云
        viewer->addPointCloud<pcl::PointXYZ>(targetClusterInfo_last_frame.cloud, "hook_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "hook_cloud");


        //吊钩到障碍物距离(近似，从hook中心到障碍物)
        pcl::PointXYZ p1 (targetClusterInfo_last_frame.centroid.x(), targetClusterInfo_last_frame.centroid.y(),targetClusterInfo_last_frame.centroid.z()  );
        pcl::PointXYZ p2 (p1.x+min_dist*dir.x(),p1.y+min_dist*dir.y(),p1.z+min_dist*dir.z());
        viewer->addLine<pcl::PointXYZ,pcl::PointXYZ>(p1,p2,1.0,1.0,0.0,"distance to obstacle");

        // viewer->addPointCloud<pcl::PointXYZ>(final_cloud, "obstacle_cloud");
        // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "obstacle_cloud");
        // 刷新可视化窗口
        viewer->spinOnce(100);
        sleep(1);
        idx = (idx + 1) % pcd_files.size();   // 循环读取   
    }
    return 0;
}