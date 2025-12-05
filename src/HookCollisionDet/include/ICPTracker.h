#pragma once  // 推荐写在文件最开头

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/filters/extract_indices.h>
#include <iostream>
#include <pcl/registration/icp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include "utils.h"

template<typename PointT>
class ICPTracker
{
    private:
        /* data */
        Eigen::Matrix4f last_transform = Eigen::Matrix4f::Identity();
        
        float computeOverlap(   const typename pcl::PointCloud<PointT>::Ptr& aligned,  // ICP后对齐的模板点云
                                const typename pcl::PointCloud<PointT>::Ptr& source,   // 当前帧点云
                                float dist_thresh=0.2f ); // 阈值， 取决于点云分辨率（这里假设 20cm）

    public:
        ICPTracker(/* args */);
        ~ICPTracker();

        void  getTargetClusterAndEnvCloud(  typename pcl::PointCloud<PointT>::Ptr& source,   // 当前帧点云
                                            const typename pcl::PointCloud<PointT>::Ptr& aligned, 
                                            ClusterInfo<PointT>& target_in_source, //target transform 到 source 之后的点云
                                            typename pcl::PointCloud<PointT>::Ptr& env_cloud );
        bool targetTrack(  typename pcl::PointCloud<PointT>::Ptr& source,   // 当前帧点云
                            ClusterInfo<PointT>& target,// 模板点云（上一帧吊钩）
                            ClusterInfo<PointT>& aligned_cloud ,// 当前帧吊钩点云
                            typename pcl::PointCloud<PointT>::Ptr& env_cloud);  //当前帧环境点云
                                                       
};


