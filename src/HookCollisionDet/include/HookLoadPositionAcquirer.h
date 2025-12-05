#pragma once  // 推荐写在文件最开头
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/pca.h>
#include <pcl/common/geometry.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/eigen.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <unordered_set>
#include <random>
#include <queue>
#include <chrono>
#include <pcl/filters/crop_box.h>
#include <omp.h>
#include "utils.h"

using namespace std;
template <typename PointT>
class HookLoadPositionAcquirer
{
    private:
        double euclideanDistance(const  PointT& p1, const   PointT& p2);
        double distanceToEndpoints(const  PointT&point, const  PointT& end1, const  PointT& end2);
        Eigen::Affine3f extendOBB(const Eigen::Affine3f& transform, const Eigen::Vector3f& dimensions, float extension_factor);
        Eigen::Vector3f extendOBBDimensions(const Eigen::Vector3f& dimensions, float extensionFactor);
        void extendOBBInDirection(Eigen::Vector3f& minPoint, Eigen::Vector3f& maxPoint, const Eigen::Vector3f& direction, float length);
        typename pcl::PointCloud<PointT>::Ptr extractPointsInExtendedOBB(const typename pcl::PointCloud<PointT>::Ptr cloud, const Eigen::Vector3f& minPoint, const Eigen::Vector3f& maxPoint);
        void computeOBB(const typename pcl::PointCloud<PointT>::Ptr cloud, Eigen::Vector3f& center, Eigen::Quaternionf& rotation, Eigen::Vector3f& dimensions);
        typename pcl::PointCloud<PointT>::Ptr extractBottomRegionAlongOBB(const typename pcl::PointCloud<PointT>::Ptr cloud, const Eigen::Vector3f& center, const Eigen::Quaternionf& rotation, float bottom_height);    
        typename pcl::PointCloud<PointT>::Ptr traversePointCloudInFixedRegions(const typename pcl::PointCloud<PointT>::Ptr& cloud, const Eigen::Vector3f& center, const Eigen::Quaternionf& rotation,
        const Eigen::Vector3f& dimensions, float bottom_height); 
        pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const typename pcl::PointCloud<PointT>::Ptr cloud);
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFPFH(const typename pcl::PointCloud<PointT>::Ptr cloud, const pcl::PointCloud<pcl::Normal>::Ptr normals);
        std::vector<pcl::PointIndices> clusterPointCloudBasedOnTarget(const typename pcl::PointCloud<PointT>::Ptr cloud, const typename pcl::PointCloud<PointT>::Ptr target);

    public:
        HookLoadPositionAcquirer(/* args */);
        ~HookLoadPositionAcquirer();


        //获取吊钩吊载点云
        bool getHookLoadCluster(typename pcl::PointCloud<PointT>::Ptr &cloud, //【输入】点云
                                            const float &rope_len, //【输入】绳长
                                            const bool &load_exist_flag, //【输入】吊载存在的标志位
                                            ClusterInfo<PointT>& hookClusterInfo, //【输出】吊钩点云
                                            ClusterInfo<PointT>& loadClusterInfo); //【输出】吊载点云


        //基于种子点的区域生长
        bool seedRegionGrowing( const typename pcl::PointCloud<PointT>::Ptr& cloud,
                                const Eigen::Vector3f& seed_point,
                                float search_radius,
                                typename pcl::PointCloud<PointT>::Ptr& cluster);

        //点到直线距离
        float pointToLineDistance(  const PointT& p,
                                    const Eigen::Vector3f& A,
                                    const Eigen::Vector3f& B);
                                    bool fitLineSVD(Eigen::Vector3d& point,Eigen::Vector3d& direction,const typename pcl::PointCloud<PointT>::Ptr& points) ;
        
        //基于点簇区域生长
        bool growFromLocalCluster(  const  typename pcl::PointCloud<PointT>::Ptr &cloud,
                                    const std::vector<int> &local_indices,   // 已知局部簇索引
                                    float search_radius,
                                    typename pcl::PointCloud<PointT>::Ptr &cluster);

            
};