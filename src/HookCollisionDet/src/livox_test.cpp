#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>

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

using namespace std;


int main(int argc, char** argv) {

    ros::init(argc, argv, "livox_reader_node");
    ros::NodeHandle nh("~");
    int hook_load_position_acquisition_way;
    std::string init_pcd_file,test_pcd_files_dir;
    float rope_len;
    if(!nh.getParam("rope_len",rope_len)){
        ROS_ERROR("cant get rope_len");
        return 0;
    }
    if(!nh.getParam("test_pcd_files_dir",test_pcd_files_dir)){
        ROS_ERROR("cant get test pcd files dir");
        return 0;
    }
    if(!nh.getParam("init_pcd_file",init_pcd_file)){
        ROS_ERROR("cant get test pcd file");
        return 0;
    }
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
    // 原始雷达坐标系中：X+ 向下，所以 -X 是“向上”
    Eigen::Vector3f current_up(-1.0f, 0.0f, 0.0f);

    // CSF 期望的“向上”方向
    Eigen::Vector3f target_up(0.0f, 0.0f, 1.0f);
    Eigen::Vector3f axis = current_up.cross(target_up);
    float axis_norm = axis.norm();

    // 处理平行或反向的极端情况（非常重要）
    if (axis_norm < 1e-6) {
        axis = Eigen::Vector3f(0, 1, 0); // 任意正交轴
    } else {
        axis.normalize();
    }

    float angle = acos(
        std::min(1.0f, std::max(-1.0f, current_up.dot(target_up)))
    );
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(angle, axis));

    // 旋转点云
    auto start = std::chrono::high_resolution_clock::now();
    // 使用 pcl::transformPointCloud 进行坐标变换
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*init_cloud, *transformed_cloud, transform);

    auto end_coord_transform = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_coord_transform - start;
    std::cout << "【time】coord transform took " << elapsed.count() << " seconds." << std::endl;
    std::vector<int> ground_indices, non_ground_indices;
    
    CSF csf;
    /**
        * 是否对地面进行坡度平滑（Slope Smooth）
        * true  ：允许地面是连续变化的斜坡（推荐）
        * false ：地面被假设为阶梯或刚性平面
        *
        * 影响：
        * - 开启后对缓坡、起伏地面更友好
        * - 对向下雷达、非规则地面建议开启
     */
    csf.params.bSloopSmooth = true; 
    /**
        * 时间步长（Time Step）
        * 控制每一次迭代中布的“下落速度”
        *
        * 物理意义：
        * - 类似仿真中的 Δt
        *
        * 当前值 0.65：
        * - 偏快
        * - 有助于快速收敛
        *
        * 影响：
        * - 太小：收敛慢，需要更多迭代
        * - 太大：可能数值不稳定，布抖动
        *
        * 常见稳定范围：0.3 ~ 0.7
        */
    csf.params.time_step = 0.65;
    /**
        * 地面/非地面分类阈值（单位：米）
        * 表示：点到最终“布模型”的垂直距离阈值
        *
        * 判定逻辑：
        * |Z_point - Z_cloth| < class_threshold  → 地面点
        * |Z_point - Z_cloth| ≥ class_threshold  → 非地面点
        *
        * 当前值 0.5m：
        * - 阈值偏大，容忍地面起伏和噪声
        * - 吊钩雷达有高度噪声时较安全
        *
        * 过小风险：
        * - 大量真实地面被误判为非地面
        * 过大风险：
        * - 低矮障碍物被当成地面
        */
    csf.params.class_threshold =0.5; 
    /**
        * 布模型分辨率（单位：米）
        * 表示：布网格节点之间的间距
        *
        * 本质：
        * - 控制“布”的空间采样密度
        * - 决定地面建模的精细程度
        *
        * 当前值 0.75m：
        * - 偏粗
        * - 更偏向“整体地面趋势”
        *
        * 影响：
        * - 值大：地面被强烈平滑，结果更稳定但细节损失
        * - 值小：能贴合局部起伏，但对噪声敏感、计算量增加
        *
        * 建议：
        * - 吊钩防碰撞 / 精细分割：0.2 ~ 0.4
        * - 大范围粗地形：0.5 ~ 1.0
        */
    csf.params.cloth_resolution =0.75;
    /**
        * 布的刚性（Rigidness）
        * 表示：布对弯曲的抵抗能力
        *
        * 数值含义：
        * - 小值：布很“软”，容易下陷
        * - 大值：布很“硬”，更像平板
        *
        * 当前值 3：
        * - 中等偏软
        * - 能适应轻微起伏地面
        *
        * 影响：
        * - 刚性过小：布会“钻进”坑洼，误判地面
        * - 刚性过大：布悬空，斜坡地面识别差
        *
        * 建议范围：2 ~ 4
        */
    csf.params.rigidness = 4;
    /**
        * 布下落迭代次数
        * 表示：布在重力和约束作用下的模拟步数
        *
        * 本质：
        * - 次数越多，布越充分贴合“地面”
        *
        * 当前值 500：
        * - 属于安全偏大的设置
        * - 有助于复杂地面收敛
        *
        * 影响：
        * - 次数太少：布未贴合就停止，地面抬高
        * - 次数太多：计算时间增加，但效果提升有限
        *
        * 实时系统常用：300 ~ 800
        */
    csf.params.interations = 500;



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
        csf.setPointCloud(csf_cloud);
        // 3. 调用CSF分割
        csf.do_filtering(ground_indices, non_ground_indices, false);
        auto end_CSF = std::chrono::high_resolution_clock::now();
        elapsed = end_CSF - end_coord_transform;
        std::cout << "【time】CSF  took " << elapsed.count() << " seconds." << std::endl;

    #else //使用pcl的数据类型
        csf.setPointCloud(transformed_cloud);
        csf.do_filtering_v2(ground_indices, non_ground_indices, false);
        auto end_CSF = std::chrono::high_resolution_clock::now();
        elapsed = end_CSF - end_coord_transform;
        std::cout << "【time】CSF took " << elapsed.count() << " seconds." << std::endl;
    #endif
    ROS_INFO("num total pts:%d ; num ground pts:%d ; num off ground pts: %d",transformed_cloud->size(),ground_indices.size(),non_ground_indices.size());
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

    // 假设 cloud 已经填充数据
    transformed_cloud->width  = transformed_cloud->points.size();
    transformed_cloud->height = 1;
    transformed_cloud->is_dense = false;

    pcl::io::savePCDFileASCII("output_off_ground_cloud.pcd", *transformed_cloud);
    // 计算逆变换矩阵（后续吊钩吊载检测坐标系为未转换的坐标系，因此转换回去）
    Eigen::Affine3f inverse_transform = transform.inverse();
    pcl::transformPointCloud(*off_ground_cloud,*off_ground_cloud,inverse_transform);
    
//================================ 吊钩吊载识别 ===================================
    ROS_INFO("================================ Hook/Load Detection ===================================\n");
    
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
#ifndef HOOK_DET_DEBUG
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
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "current_frame");
        
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
#endif
        // //初始点云
        // viewer->addPointCloud<pcl::PointXYZ>(init_cloud, "init_cloud");
        // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, "init_cloud");
        //CSF滤波点云
        viewer->addPointCloud<pcl::PointXYZ>(off_ground_cloud, "off_ground_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "off_ground_cloud");
        
        while (!viewer->wasStopped())
        {
            viewer->spinOnce();
        }
    return 0;
}