#include "ICPTracker.h"
template<typename PointT>
ICPTracker<PointT>::ICPTracker(){}

template<typename PointT>
ICPTracker<PointT>::~ICPTracker(){}

template<typename PointT>
float ICPTracker<PointT>::computeOverlap(
    const typename pcl::PointCloud<PointT>::Ptr& aligned,  // ICP后对齐的模板点云
    const typename pcl::PointCloud<PointT>::Ptr& source,   // 当前帧点云
    float dist_thresh ) // 阈值，取决于点云分辨率（这里假设 20cm）
{
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(source);

    int matched = 0;
    for (const auto& p : aligned->points)
    {
        std::vector<int> indices(1);
        std::vector<float> sqr_dist(1);
        if (kdtree.nearestKSearch(p, 1, indices, sqr_dist) > 0)
        {
            if (sqr_dist[0] < dist_thresh * dist_thresh)
                matched++;
        }
    }
    return static_cast<float>(matched) / static_cast<float>(aligned->size());
}

template<typename PointT>
void  ICPTracker<PointT>::getTargetClusterAndEnvCloud( typename pcl::PointCloud<PointT>::Ptr& source,   // 当前帧点云
                                                            const typename pcl::PointCloud<PointT>::Ptr& aligned,
                                                            ClusterInfo<PointT>& target_in_source, //target transform 到 source 之后的点云
                                                            typename pcl::PointCloud<PointT>::Ptr& safe_region_env_cloud)
    {
//step1 提取当前帧中吊钩吊载点云（相对于原始点云帧的索引）
    typename pcl::PointCloud<PointT>::Ptr hook_region(new pcl::PointCloud<PointT>);
    Eigen::Vector4f bbMin  (std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),1.0f);
    Eigen::Vector4f bbMax  (std::numeric_limits<float>::lowest(),std::numeric_limits<float>::lowest(),std::numeric_limits<float>::lowest(),1.0f);
    for (auto  point: aligned->points)
    {
        bbMin[0] = std::min(bbMin[0], point.x);
        bbMin[1] = std::min(bbMin[1], point.y);
        bbMin[2] = std::min(bbMin[2], point.z);
        bbMax[0] = std::max(bbMax[0], point.x);
        bbMax[1] = std::max(bbMax[1], point.y);
        bbMax[2] = std::max(bbMax[2], point.z);
    }
    bbMin = bbMin.array() - 2.0f;
    bbMax = bbMax.array() + 2.0f;
    pcl::CropBox<PointT> crop; //使用crop降低聚类算法数据量
    crop.setInputCloud(source);
    crop.setMin(bbMin);
    crop.setMax(bbMax);
    std::vector<int> region_indices;//相对于source 点云的原始索引
    crop.filter(region_indices);

    pcl::ExtractIndices<PointT> extract;
    pcl::PointIndices::Ptr region_point_indices(new pcl::PointIndices);
    region_point_indices->indices = region_indices;
    extract.setInputCloud(source);
    extract.setIndices(region_point_indices);
    extract.setNegative(false);
    extract.filter(*hook_region);

    float distance_threshold = 1;
    pcl::PointIndices::Ptr inliers1 = clusterPointCloudByTargetDistance(hook_region, aligned, distance_threshold); // 根据目标区域点云距离聚类吊钩点云

    pcl::PointIndices::Ptr inliers_in_cloud(new pcl::PointIndices);
    for (int idx : inliers1->indices) {
        inliers_in_cloud->indices.push_back(region_indices[idx]);
    }
    extract.setInputCloud(source);
    extract.setIndices(inliers_in_cloud);//索引为相对于source的，为了后续实现索引差集
    extract.filter(*target_in_source.cloud); //得到当前帧中的目标点云

// step 2 剔除原始点云场景中的吊钩吊载点云
    typename pcl::PointCloud<PointT>::Ptr env_cloud(new pcl::PointCloud<PointT>);
    // 2.1 剔除吊钩吊载点云
    extract.setNegative(true);
    extract.filter(*env_cloud); 

    // 2.2 直通选出车体点云
    pcl::PassThrough<PointT> pass_y;
    pass_y.setInputCloud(env_cloud);
    pass_y.setFilterFieldName("y");
    pass_y.setFilterLimits(0,5);
    pass_y.setNegative(true);
    std::vector<int> indices_y;
    pass_y.filter(indices_y);

    pcl::PassThrough<PointT> pass_z;
    pass_z.setInputCloud(env_cloud);
    pass_z.setFilterFieldName("z");
    pass_z.setFilterLimits(-5,100);
    pass_z.setNegative(true);
    std::vector<int> indices_z;
    pass_z.filter(indices_z);
    // 先排序（确保升序）
    std::sort(indices_y.begin(), indices_y.end());
    std::sort(indices_z.begin(), indices_z.end());

    std::vector<int> union_indices;
    std::set_union(indices_y.begin(), indices_y.end(),
                indices_z.begin(), indices_z.end(),
                std::back_inserter(union_indices));

    // 2.3 剔除车体点云
    pcl::PointIndices::Ptr env_without_crane_indices(new pcl::PointIndices);
    env_without_crane_indices->indices = union_indices;
    extract.setInputCloud(env_cloud);
    extract.setIndices(env_without_crane_indices);//索引为相对于source的，为了后续实现索引差集
    extract.setNegative(false);
    extract.filter(*env_cloud); 


    // 2.3 筛选碰撞检测区点云
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*target_in_source.cloud, centroid);//提取吊钩质心
    target_in_source.centroid  = Eigen::Vector3f(centroid[0],centroid[1],centroid[2]);
    float range_x{30},range_y{50},range_z{50};// 定义长方体尺寸（长、宽、高）
    Eigen::Vector4f min_point(centroid[0] - range_x/2, centroid[1] - range_y/2, centroid[2] - range_z/2, 1.0);
    Eigen::Vector4f max_point(centroid[0] + range_x/2, centroid[1] + range_y/2, centroid[2] + range_z/2, 1.0);
    pcl::CropBox<PointT> crop_box;
    crop_box.setMin(min_point);
    crop_box.setMax(max_point);
    crop_box.setInputCloud(env_cloud);
    crop_box.setNegative(false);  // false表示保留盒内点，true表示保留盒外点
    crop_box.filter(*safe_region_env_cloud);
}



template<typename PointT>
bool ICPTracker<PointT>::targetTrack(
    typename pcl::PointCloud<PointT>::Ptr& source,   // 当前帧点云
    ClusterInfo<PointT>& target,    // 模板点云（上一帧吊钩）
    ClusterInfo<PointT>& target_in_source, //target transform 到 source 之后的点云
    typename pcl::PointCloud<PointT>::Ptr& env_cloud )  
{
    // ---------------- 1. 降采样（减少点数，提高ICP速度） ----------------
    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);  // 根据实际场景调整
    typename pcl::PointCloud<PointT>::Ptr src_ds(new pcl::PointCloud<PointT>);
    typename pcl::PointCloud<PointT>::Ptr tgt_ds(new pcl::PointCloud<PointT>);
    voxel_filter.setInputCloud(source);
    voxel_filter.filter(*src_ds);
    voxel_filter.setInputCloud(target.cloud);
    voxel_filter.filter(*tgt_ds);

    // ---------------- 2. 构建 ICP 对象 ----------------
    pcl::IterativeClosestPoint<PointT, PointT> icp;

    icp.setInputSource(tgt_ds);
    icp.setInputTarget(src_ds);

    // 最近邻搜索用 k-d tree，加速匹配
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    icp.setSearchMethodSource(tree);
    icp.setSearchMethodTarget(tree);

    // ---------------- 3. 参数优化 ----------------
    // icp.setMaxCorrespondenceDistance(5);   // 限制匹配点的最大距离（根据吊钩尺寸设定）
    icp.setMaximumIterations(100);           // 最大迭代次数（减少过度计算）
    icp.setTransformationEpsilon(1e-6);     // 收敛判据（变换矩阵收敛）
    icp.setEuclideanFitnessEpsilon(1e-6);   // 收敛判据（误差收敛）

    // ---------------- 4. ICP 执行 ----------------
    typename pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>() );
    icp.align(*aligned,last_transform);
    if (icp.hasConverged()) {
        float score = icp.getFitnessScore();
        float overlap =computeOverlap(aligned, src_ds, 0.2f);

        ROS_INFO("[ICP tracker] ICP converged. score= %f , overlap= %f ",score,overlap);
        if (score < 0.1 && overlap > 0.6f) { // 双条件约束
            last_transform = icp.getFinalTransformation();
            getTargetClusterAndEnvCloud(source, aligned,target_in_source, env_cloud);
            return true;
        }
    }
    ROS_WARN("[ICP tracker] ICP did not converge or overlap too low.");
    return false;
}

template class ICPTracker<pcl::PointXYZ>;