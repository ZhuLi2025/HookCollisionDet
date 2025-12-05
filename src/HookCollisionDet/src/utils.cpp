#include "utils.h"

// 根据目标区域点云 距离聚类整个点云
template <typename PointT>
pcl::PointIndices::Ptr clusterPointCloudByTargetDistance(
    const typename pcl::PointCloud<PointT>::Ptr& cloud,
    const typename pcl::PointCloud<PointT>::Ptr& target,
    float distance_threshold)
    {
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    if (!cloud || cloud->empty() || !target || target->empty() || distance_threshold <= 0.0f)
        return inliers;

    pcl::KdTreeFLANN<PointT> kdtree;
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
}

// 显式实例化
template pcl::PointIndices::Ptr clusterPointCloudByTargetDistance<pcl::PointXYZ>(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr&,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr&,
    float);

template pcl::PointIndices::Ptr clusterPointCloudByTargetDistance<pcl::PointXYZI>(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr&,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr&,
    float);