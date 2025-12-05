#pragma once  // 推荐写在文件最开头
#include <ros/ros.h>
#include <vector>
#include <pcl/point_cloud.h>       // pcl::PointCloud
#include <pcl/point_types.h>       // pcl::PointXYZ
#include <pcl/PointIndices.h>      // pcl::PointIndices
#include <pcl/kdtree/kdtree_flann.h> // pcl::KdTreeFLANN
#include <pcl/common/common.h>     // pcl::isFinite
#include <pcl/common/point_tests.h>     // pcl::isFinite
#include <pcl/features/moment_of_inertia_estimation.h> 

// ========== 定义 OBB 结构 ==========
template <typename PointT>
struct OBB {
    Eigen::Vector3f min_point;
    Eigen::Vector3f max_point;
    Eigen::Vector3f position;
    Eigen::Matrix3f rotation;

    OBB() 
        : min_point(Eigen::Vector3f::Zero()),
        max_point(Eigen::Vector3f::Zero()),
        position(Eigen::Vector3f::Zero()),
        rotation(Eigen::Matrix3f::Identity()) {}

    // ========== 计算点云的 OBB ==========
    OBB(const typename pcl::PointCloud<PointT>::Ptr& cloud){
        pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
        feature_extractor.setInputCloud(cloud);
        feature_extractor.compute();

        PointT min_point_OBB, max_point_OBB, position_OBB;
        Eigen::Matrix3f rotational_matrix_OBB;
        feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);

        min_point = Eigen::Vector3f(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
        max_point = Eigen::Vector3f(max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
        position  = Eigen::Vector3f(position_OBB.x, position_OBB.y, position_OBB.z);
        rotation  = rotational_matrix_OBB;
    }
};

// ========== 定义 点簇 结构 ==========
template <typename PointT>
struct ClusterInfo {
    int id;
    typename pcl::PointCloud<PointT>::Ptr cloud;
    Eigen::Vector3f centroid;
    float distance;  // 质心到直线的距离
    std::vector<int> indices;
    OBB<PointT> obb; // 新增：该簇对应的 OBB

    // 默认构造函数
    ClusterInfo()
        : id(-1),
          cloud(new pcl::PointCloud<PointT>),
          centroid(Eigen::Vector3f::Zero()),
          distance(-1.0f),
          obb() {}   // 默认初始化为空 OBB


    // ========== 深拷贝构造函数 ==========
    ClusterInfo(const ClusterInfo& other)
        : id(other.id),
          cloud(new pcl::PointCloud<PointT>(*other.cloud)), // 复制点云内容
          centroid(other.centroid),
          distance(other.distance),
          indices(other.indices),
          obb(other.obb) {}

    // ========== 深拷贝赋值运算符 ==========
    ClusterInfo& operator=(const ClusterInfo& other) {
        if (this != &other) {
            id = other.id;
            cloud.reset(new pcl::PointCloud<PointT>(*other.cloud)); // 复制点云内容
            centroid = other.centroid;
            distance = other.distance;
            indices = other.indices;
            obb = other.obb;
        }
        return *this;
    }

    bool exist() const {
        return !cloud->empty();
    }

    void clearAll() {
        id = -1;
        cloud->clear();
        centroid = Eigen::Vector3f::Zero();
        distance = -1.0f;
        obb = OBB<PointT>(); // 重置 OBB
    }

    void clearCloud() {
        cloud->clear();
        obb = OBB<PointT>(); // 清空时也同步重置 OBB
    }

    // 计算并更新 OBB
    void computeOBB() {
        if (cloud && !cloud->empty()) {
            obb = OBB<PointT>(cloud);
        }
    }
};


// 根据目标区域点云 距离聚类整个点云
inline pcl::PointIndices::Ptr clusterPointCloudByTargetDistance(
    const  pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const  pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    float distance_threshold)
    {
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    if (!cloud || cloud->empty() || !target || target->empty() || distance_threshold <= 0.0f)
        return inliers;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(target);

    inliers->indices.reserve(cloud->size()); // 预分配

    std::vector<int> nn_indices;
    std::vector<float> nn_sqr_dists;
    float thr2 = distance_threshold * distance_threshold;

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        const auto &pt = cloud->points[i];
        if (!pcl::isFinite(pt))
            continue; // 跳过 NaN/Inf 点

        // 方案 A：最近邻（只关心最近的一个）
        nn_indices.resize(1);
        nn_sqr_dists.resize(1);
        if (kdtree.nearestKSearch(pt, 1, nn_indices, nn_sqr_dists) > 0)
        {
            if (nn_sqr_dists[0] <= thr2)
                inliers->indices.push_back(i);
        }

        // 或 方案 B：半径搜索（直接使用阈值）
        // if (kdtree.radiusSearch(pt, distance_threshold, nn_indices, nn_sqr_dists) > 0) {
        //     inliers->indices.push_back(i);
        // }
    }
    return inliers;
};