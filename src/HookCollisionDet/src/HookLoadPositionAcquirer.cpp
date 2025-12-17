#include "HookLoadPositionAcquirer.h"


using namespace std;

template<typename PointT>
HookLoadPositionAcquirer<PointT>::HookLoadPositionAcquirer() {}

template<typename PointT>
HookLoadPositionAcquirer<PointT>::~HookLoadPositionAcquirer() {}

// 计算两点间的欧氏距离
template<typename PointT>
double HookLoadPositionAcquirer<PointT>::euclideanDistance(const  PointT& p1, const  PointT& p2)
{
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// 计算点到直线端点的距离
template<typename PointT>
double HookLoadPositionAcquirer<PointT>::distanceToEndpoints(const  PointT&point, const  PointT& end1, const  PointT& end2)
{
    double dist1 = euclideanDistance(point, end1);
    double dist2 = euclideanDistance(point, end2);
    return std::min(dist1, dist2);
}

// 延长 OBB 包围盒
template<typename PointT>
Eigen::Affine3f HookLoadPositionAcquirer<PointT>::extendOBB(const Eigen::Affine3f &transform, const Eigen::Vector3f &dimensions, float extension_factor)
{
    Eigen::Affine3f extended_transform = transform;
    // 延长包围盒的尺寸
    Eigen::Vector3f extended_dimensions = dimensions * extension_factor;
    // 更新包围盒的变换矩阵
    extended_transform.translation() = transform.translation();
    return extended_transform;
}

// 延长 OBB 包围盒尺寸
template<typename PointT>
Eigen::Vector3f HookLoadPositionAcquirer<PointT>::extendOBBDimensions(const Eigen::Vector3f &dimensions, float extensionFactor)
{
    return dimensions * extensionFactor;
}

// 沿固定方向延长 OBB 包围盒
template<typename PointT>
void HookLoadPositionAcquirer<PointT>::extendOBBInDirection(Eigen::Vector3f &minPoint, Eigen::Vector3f &maxPoint, const Eigen::Vector3f &direction, float length)
{
    Eigen::Vector3f extension = direction.normalized() * length;
    // minPoint -= extension;
    maxPoint += extension;
}

// 提取延长后的包络盒中的点云
template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr HookLoadPositionAcquirer<PointT>::extractPointsInExtendedOBB(const typename pcl::PointCloud<PointT>::Ptr cloud, const Eigen::Vector3f &minPoint, const Eigen::Vector3f &maxPoint)
{
    typename pcl::PointCloud<PointT>::Ptr extractedCloud(new pcl::PointCloud<PointT>);
    for (const auto &point : *cloud)
    {
        if (point.x >= minPoint[0] && point.x <= maxPoint[0] &&
            (point.y >= minPoint[1]) && (point.y <= maxPoint[1]) &&
            (point.z >= minPoint[2]) && (point.z <= maxPoint[2]))
        {
            extractedCloud->push_back(point);
        }
    }
    return extractedCloud;
}

// 计算 OBB 的主方向和尺寸
template<typename PointT>
void HookLoadPositionAcquirer<PointT>::computeOBB(const typename pcl::PointCloud<PointT>::Ptr cloud, Eigen::Vector3f &center, Eigen::Quaternionf &rotation, Eigen::Vector3f &dimensions)
{
    PointT min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);

    center = Eigen::Vector3f((min_pt.x + max_pt.x) / 2.0, (min_pt.y + max_pt.y) / 2.0, (min_pt.z + max_pt.z) / 2.0);
    dimensions = Eigen::Vector3f(max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z);
    rotation = Eigen::Quaternionf::Identity();
}

// 沿着 OBB 主方向提取底部区域点云
template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr HookLoadPositionAcquirer<PointT>::extractBottomRegionAlongOBB(const typename pcl::PointCloud<PointT>::Ptr cloud, const Eigen::Vector3f &center, const Eigen::Quaternionf &rotation, float bottom_height)
{
    // 将点云转换到 OBB 坐标系
    typename pcl::PointCloud<PointT>::Ptr transformed_cloud(new pcl::PointCloud<PointT>);
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f rotation_matrix = rotation.inverse().toRotationMatrix();
    transform.block<3, 3>(0, 0) = rotation_matrix;
    transform.block<3, 1>(0, 3) = -rotation_matrix * center;
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform);

    // 找到底部区域点云

    // 提取底部区域点云
    typename pcl::PointCloud<PointT>::Ptr target_partial_cloud(new pcl::PointCloud<PointT>);
    for (const auto &point : transformed_cloud->points)
    {
        if (point.z < bottom_height)
        {
            target_partial_cloud->push_back(point);
        }
    }

    // 将底部区域点云转换回原坐标系
    typename pcl::PointCloud<PointT>::Ptr original_target_partial_cloud(new pcl::PointCloud<PointT>);
    Eigen::Matrix4f inverse_transform = Eigen::Matrix4f::Identity();
    inverse_transform.block<3, 3>(0, 0) = rotation.toRotationMatrix();
    inverse_transform.block<3, 1>(0, 3) = center;
    pcl::transformPointCloud(*target_partial_cloud, *original_target_partial_cloud, inverse_transform);

    return original_target_partial_cloud;
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr HookLoadPositionAcquirer<PointT>::traversePointCloudInFixedRegions(
    const typename pcl::PointCloud<PointT>::Ptr &cloud,
    const Eigen::Vector3f &center,
    const Eigen::Quaternionf &rotation,
    const Eigen::Vector3f &dimensions,
    float step_height)
{
    // 1. 将点云转换到OBB坐标系
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f rot_inv = rotation.inverse().toRotationMatrix();
    transform.block<3, 3>(0, 0) = rot_inv;
    transform.block<3, 1>(0, 3) = -rot_inv * center;

    typename pcl::PointCloud<PointT>::Ptr obb_cloud(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*cloud, *obb_cloud, transform);

    // 2. 定义OBB范围
    float half_x = dimensions.x() * 0.5f;
    float half_y = dimensions.y() * 0.5f;
    float half_z = dimensions.z() * 0.5f;
    float min_x = -half_x, max_x = half_x;
    float min_y = -half_y, max_y = half_y;
    float min_z = -half_z; // 从底部开始
    float max_z = half_z;

    // 3. 从底部开始，每次在 step_height 区域内查找点
    typename pcl::PointCloud<PointT>::Ptr region_cloud(new pcl::PointCloud<PointT>);
    for (float z_top = min_z + step_height; z_top <= max_z; z_top += step_height)
    {
        region_cloud->clear();
        for (const auto &pt : obb_cloud->points)
        {
            if (pt.x >= min_x && pt.x <= max_x &&
                pt.y >= min_y && pt.y <= max_y &&
                pt.z >= (z_top - step_height) && pt.z < z_top)
            {
                region_cloud->push_back(pt);
                if (region_cloud->size() > 5)
                    break; // 找到足够的点就提前退出
            }
        }
        if (region_cloud->size() > 5)
            break; // 找到合适区域就结束
    }

    if (region_cloud->empty())
    {
        std::cout << "没有找到底部点云" << std::endl;
        return region_cloud;
    }

    // 4. 转回原坐标系
    Eigen::Matrix4f inv_transform = Eigen::Matrix4f::Identity();
    inv_transform.block<3, 3>(0, 0) = rotation.toRotationMatrix();
    inv_transform.block<3, 1>(0, 3) = center;

    typename pcl::PointCloud<PointT>::Ptr original_cloud(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*region_cloud, *original_cloud, inv_transform);

    return original_cloud;
}

// 计算点云的法向量
template<typename PointT>
pcl::PointCloud<pcl::Normal>::Ptr HookLoadPositionAcquirer<PointT>::computeNormals(const typename pcl::PointCloud<PointT>::Ptr cloud)
{
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.03);
    ne.compute(*cloud_normals);

    return cloud_normals;
}

// 计算点云的FPFH特征
template<typename PointT>
pcl::PointCloud<pcl::FPFHSignature33>::Ptr HookLoadPositionAcquirer<PointT>::computeFPFH(const typename pcl::PointCloud<PointT>::Ptr cloud, const pcl::PointCloud<pcl::Normal>::Ptr normals)
{
    pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());

    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(0.05);
    fpfh.compute(*fpfhs);

    return fpfhs;
}

// 根据目标局部点云聚类整个点云
template<typename PointT>
std::vector<pcl::PointIndices> HookLoadPositionAcquirer<PointT>::clusterPointCloudBasedOnTarget(const typename pcl::PointCloud<PointT>::Ptr cloud,
                                                                                        const typename pcl::PointCloud<PointT>::Ptr target)
{
    // 计算整个点云和目标局部点云的法向量
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals = computeNormals(cloud);
    pcl::PointCloud<pcl::Normal>::Ptr target_normals = computeNormals(target);

    // 计算整个点云和目标局部点云的FPFH特征
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloud_fpfhs = computeFPFH(cloud, cloud_normals);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_fpfhs = computeFPFH(target, target_normals);

    // 特征匹配
    pcl::KdTreeFLANN<pcl::FPFHSignature33> kdtree;
    kdtree.setInputCloud(target_fpfhs);

    std::vector<int> pointIndices;
    for (size_t i = 0; i < cloud_fpfhs->size(); ++i)
    {
        std::vector<int> indices(1);
        std::vector<float> distances(1);
        if (kdtree.nearestKSearch(cloud_fpfhs->points[i], 1, indices, distances) > 0 && distances[0] < 0.1)
        {
            pointIndices.push_back(i);
        }
    }

    // 创建一个新的点云，包含匹配到的点
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    inliers->indices = pointIndices;
    pcl::ExtractIndices<PointT> extract;
    typename pcl::PointCloud<PointT>::Ptr cloud_matched(new pcl::PointCloud<PointT>);
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.filter(*cloud_matched);

    // 聚类
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud_matched);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.02);
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_matched);
    ec.extract(cluster_indices);

    return cluster_indices;
}
//基于点簇区域生长
template<typename PointT>
bool HookLoadPositionAcquirer<PointT>::growFromLocalCluster(
    const typename  pcl::PointCloud<PointT>::Ptr &cloud,
    const std::vector<int> &local_indices,   // 已知局部簇索引
    float search_radius,
    typename  pcl::PointCloud<PointT>::Ptr &cluster)
{
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    std::vector<bool> visited(cloud->size(), false);
    std::unordered_set<int> current(local_indices.begin(), local_indices.end());

    // 标记局部点簇为已访问
    for (int idx : local_indices) visited[idx] = true;

    bool expanded = true;
    while (expanded) {
        expanded = false;
        std::unordered_set<int> new_points;

        for (int idx : current) {
            std::vector<int> neighbors;
            std::vector<float> distances;
            if (kdtree.radiusSearch(cloud->points[idx], search_radius,
                                    neighbors, distances) > 0)
            {
                for (int n : neighbors) {
                    if (!visited[n]) {
                        visited[n] = true;
                        new_points.insert(n);
                        expanded = true;
                    }
                }
            }
        }

        // 合并新点到整体簇
        current.insert(new_points.begin(), new_points.end());
    }

    // 输出结果
    cluster->reserve(current.size());
    for (int idx : current) {
        cluster->push_back(cloud->points[idx]);
    }

    return !cluster->empty();
}

template<typename PointT>
bool HookLoadPositionAcquirer<PointT>::seedRegionGrowing(
    const typename pcl::PointCloud<PointT>::Ptr &cloud,
    const Eigen::Vector3f &seed_point,
    float search_radius,
    typename pcl::PointCloud<PointT>::Ptr &cluster)
{
    // ---------- Step 0: 基本输入检测 ----------
    if (!cloud || cloud->empty()) {
        ROS_WARN("seedRegionGrowing: input cloud is empty or null");
        return false;
    }
    if (!cluster) {
        ROS_WARN("seedRegionGrowing: output cluster is null");
        return false;
    }
    if (search_radius <= 0.0f) {
        ROS_WARN("seedRegionGrowing: invalid search_radius %.3f", search_radius);
        return false;
    }
    cluster->clear();

    // ---------- Step 1: 构造 KD-Tree ----------
    pcl::KdTreeFLANN<PointT> kdtree;
    try {
        kdtree.setInputCloud(cloud);
    } catch (const std::exception &e) {
        ROS_ERROR("seedRegionGrowing: KDTree construction failed: %s", e.what());
        return false;
    }

    // ---------- Step 2: 找种子点最近邻 ----------
    PointT seed;
    seed.x = seed_point.x();
    seed.y = seed_point.y();
    seed.z = seed_point.z();

    std::vector<int> seed_idx(1);
    std::vector<float> seed_dist(1);
    if (kdtree.nearestKSearch(seed, 1, seed_idx, seed_dist) <= 0) {
        ROS_WARN("seedRegionGrowing: seed point has no nearest neighbor");
        return false;
    }

    int seed_index = seed_idx[0];
    if (std::isnan(cloud->points[seed_index].x)) {
        ROS_WARN("seedRegionGrowing: invalid seed point (NaN)");
        return false;
    }

    // ---------- Step 3: 初始化队列 ----------
    std::queue<int> q;
    std::unordered_set<int> visited;
    q.push(seed_index);
    visited.insert(seed_index);

    // ---------- Step 4: 区域生长 ----------
    const size_t MAX_CLUSTER_SIZE = 50000; // 防止死循环
    const size_t MAX_QUEUE_SIZE = 100000;

    while (!q.empty()) {
        int current_idx = q.front();
        q.pop();

        // 添加到结果
        const auto &current_point = cloud->points[current_idx];
        if (!pcl::isFinite(current_point)) continue;
        cluster->push_back(current_point);

        // 防止无限膨胀
        if (cluster->size() > MAX_CLUSTER_SIZE) {
            ROS_WARN("seedRegionGrowing: cluster size exceeded limit (%zu)", cluster->size());
            break;
        }

        std::vector<int> neighbor_indices;
        std::vector<float> neighbor_distances;
        try {
            if (kdtree.radiusSearch(current_point, search_radius, neighbor_indices, neighbor_distances) > 0) {
                for (size_t i = 0; i < neighbor_indices.size(); ++i) {
                    int idx = neighbor_indices[i];
                    if (visited.find(idx) == visited.end()) {
                        visited.insert(idx);
                        q.push(idx);
                        if (q.size() > MAX_QUEUE_SIZE) {
                            ROS_WARN("seedRegionGrowing: queue size exceeded limit (%zu)", q.size());
                            break;
                        }
                    }
                }
            }
        } catch (const std::exception &e) {
            ROS_ERROR("seedRegionGrowing: radiusSearch failed: %s", e.what());
            return false;
        }
    }
    // ---------- Step 5: 结果检查 ----------
    if (cluster->empty()) {
        ROS_WARN("seedRegionGrowing: no points grown from seed");
        return false;
    }
    return true;
}

// 计算点到直线的距离
template<typename PointT>
float HookLoadPositionAcquirer<PointT>::pointToLineDistance(const PointT &p,
                                                    const Eigen::Vector3f &A,
                                                    const Eigen::Vector3f &B)
{
    Eigen::Vector3f AP(p.x - A.x(), p.y - A.y(), p.z - A.z());
    Eigen::Vector3f AB = B - A;
    float distance = (AP.cross(AB)).norm() / AB.norm();
    return distance;
}


template<typename PointT>
bool HookLoadPositionAcquirer<PointT>::fitLineSVD(Eigen::Vector3d &centroid, Eigen::Vector3d &direction, const typename pcl::PointCloud<PointT>::Ptr &cloud)
{
    if (cloud->empty()||cloud->size() < 3){
        ROS_WARN("[fitLineSVD] too few points (%zu)", cloud->size());
        return false;
    }
    Eigen::Vector3d weightedCentroid(0, 0, 0);
    double weightSum = 0;
    std::vector<double> weights(cloud->size());

    for (size_t i = 0; i < cloud->size(); ++i)
    {
        const auto &p = cloud->points[i];
        double dist = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        double w = dist; // 距离越远，权重越大
        weights[i] = w;
        weightedCentroid += w * Eigen::Vector3d(p.x, p.y, p.z);
        weightSum += w;
    }
    if (weightSum < 1e-6) {
        ROS_WARN("[fitLineSVD] invalid weights sum (%.6f)", weightSum);
        return false;
    }
    centroid = weightedCentroid / weightSum; // ✅ 更新为加权质心

    // 协方差矩阵
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < cloud->size(); ++i)
    {
        Eigen::Vector3d q(cloud->points[i].x,
                          cloud->points[i].y,
                          cloud->points[i].z);
        q -= centroid;
        C += weights[i] * (q * q.transpose());
    }
    if (!C.allFinite()) {
        ROS_WARN("[fitLineSVD] covariance matrix has invalid values");
        return false;
    }
    // 求主方向
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(C);
    if (eig.info() != Eigen::Success) {
        ROS_WARN("[fitLineSVD] eigen decomposition failed");
        return false;
    }

    Eigen::Vector3d eigen_values = eig.eigenvalues();
    direction = eig.eigenvectors().col(2).normalized();

    if (eigen_values(2) < 1e-6 ||
        (eigen_values(2) - eigen_values(1)) / eigen_values(2) < 0.01 ||
        !direction.allFinite() || !centroid.allFinite()) {
        ROS_WARN("[fitLineSVD] unstable or invalid line fit");
        return false;
    }
    return true;
}

template<typename PointT>
bool HookLoadPositionAcquirer<PointT>::getHookLoadCluster(  typename pcl::PointCloud<PointT>::Ptr &cloud,
                                                            const float &rope_len,
                                                            const bool &load_exist_flag,
                                                            ClusterInfo<PointT>& hookClusterInfo,
                                                            ClusterInfo<PointT>& loadClusterInfo)
{
    auto computeCentroid = [](const typename pcl::PointCloud<PointT>::Ptr &cloud)
    {
        Eigen::Vector3f c(0, 0, 0);
        for (const auto &p : cloud->points)
        {
            c += Eigen::Vector3f(p.x, p.y, p.z);
        }
        c /= cloud->size();
        return c;
    };

    // 参数阈值
    const int MIN_POINTS_FOR_SVD = 10;
    const float DIRECTION_EPS = 1e-6f;
    const float MAX_DISTANCE_TO_LINE_FOR_HOOK = 3.0f;
    const float MAX_DISTANCE_TO_LINE_FOR_LOAD  = 0.5f;
    // 输入检查
    if (!cloud || cloud->empty()) {
        ROS_WARN("[getHookLoadCluster] input cloud is null or empty.");
        return false; // 关键失败：没有输入点云
    }
    try{
//============================== 绳索直线求取 =====================================
        // 获取开始时间
        auto start = std::chrono::high_resolution_clock::now();

        //  第一次滤波 - Z轴
        typename pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
        float width_filter{10.0}, height_filter{10.0}, depth_filter{20.0};
        pcl::PassThrough<PointT> pass_z;
        pass_z.setInputCloud(cloud);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(-width_filter / 2, width_filter / 2);
        pass_z.filter(*cloud_filtered);
        if (cloud_filtered->empty()) {
            ROS_WARN("[step1] After Z pass-through, cloud_filtered is empty.");
            return false; // 关键失败：没有可用于 SVD 的点
        }

        // 第二次滤波 - X轴
        pcl::PassThrough<PointT> pass_x;
        pass_x.setInputCloud(cloud_filtered);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(0.0, depth_filter);
        pass_x.filter(*cloud_filtered);
        if (cloud_filtered->empty()) {
            ROS_WARN("[step1] After X pass-through, cloud_filtered is empty.");
            return false;
        }

        // 第二次滤波 - Y轴
        pcl::PassThrough<PointT> pass_y;
        pass_y.setInputCloud(cloud_filtered);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(-height_filter / 2, height_filter / 2);
        pass_y.filter(*cloud_filtered);
        if (cloud_filtered->empty()) {
            ROS_WARN("[step1] After Y pass-through, cloud_filtered is empty.");
            return false;
        }
        Eigen::Vector3f originalCenter(depth_filter / 2, 0, 0);
        Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity();

        // 获取开始时间
        auto end_step1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_step1 - start;
        std::cout << "step1 took " << elapsed.count() << " seconds." << std::endl;

    // step2 SVD方法拟合直线（可选）
        Eigen::Vector3d weighted_centerid;
        Eigen::Vector3d direction;
        // 检查点数足够做 SVD
        if (static_cast<int>(cloud_filtered->size()) < MIN_POINTS_FOR_SVD) {
            ROS_WARN("[step2] Not enough points for fitLineSVD: %d", (int)cloud_filtered->size());
            return false;
        }
        if(!fitLineSVD(weighted_centerid, direction, cloud_filtered)){
            ROS_ERROR("[getHookLoadCluster] input cloud is null");
            return false;
        }
        // 取两个点 (沿方向正负延伸)
        double scale = 50.0; // 控制直线长度
        PointT p1(weighted_centerid[0] - scale * direction[0],
                        weighted_centerid[1] - scale * direction[1],
                        weighted_centerid[2] - scale * direction[2]);
        PointT p2(weighted_centerid[0] + scale * direction[0],
                        weighted_centerid[1] + scale * direction[1],
                        weighted_centerid[2] + scale * direction[2]);
        Eigen::Vector3f point_online_1(p1.x, p1.y, p1.z);                  // 起点
        Eigen::Vector3f point_online_2(p2.x, p2.y, p2.z);                  // 终点

        // 获取开始时间
        auto end_step2 = std::chrono::high_resolution_clock::now();
        elapsed = end_step2 - end_step1;
        std::cout << "step2 took " << elapsed.count() << " seconds." << std::endl;

//============================== 求吊钩锚点并根据锚点过滤点云 =====================================
    // step3 求聚类核心点（吊钩位置附近点）
        if (std::fabs(direction[0]) < DIRECTION_EPS) {
            ROS_WARN("[step3] direction[0] is too small (division risk).");
            return false; // 无法计算与 x=0 平面交点
        }
        float scale2zeroplane = static_cast<float>(-(weighted_centerid[0] / direction[0]));
        PointT line_pt_on_zeroplane(0.0, weighted_centerid[1] + scale2zeroplane * direction[1], weighted_centerid[2] + scale2zeroplane * direction[2]);
        PointT seed_point(line_pt_on_zeroplane.x + rope_len * direction[0],
                                line_pt_on_zeroplane.y + rope_len * direction[1],
                                line_pt_on_zeroplane.z + rope_len * direction[2]);
        Eigen::Vector3f seed_point_eigen_(seed_point.x, seed_point.y, seed_point.z);

        // 获取开始时间
        auto end_step3 = std::chrono::high_resolution_clock::now();
        elapsed = end_step3 - end_step2;
        std::cout << "step3 took " << elapsed.count() << " seconds." << std::endl;

    // step4 提取吊钩点云
        typename pcl::PointCloud<PointT>::Ptr seedregion_cloud(new pcl::PointCloud<PointT>);
        float x_range{30.0}, y_range{30.0}, z_range{30.0};
        pass_z.setInputCloud(cloud);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(seed_point.z-z_range / 2,seed_point.z+z_range/ 2);
        pass_z.filter(*seedregion_cloud);
        if (seedregion_cloud->empty()) {
            ROS_WARN("[step4] seedregion_cloud empty after Z crop.");
            return false;
        }
        // 第二次滤波 - X轴
        pass_x.setInputCloud(seedregion_cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(seed_point.x-x_range/2,seed_point.x+x_range/2);
        pass_x.filter(*seedregion_cloud);
        if (seedregion_cloud->empty()) {
            ROS_WARN("[step4] seedregion_cloud empty after X crop.");
            return false;
        }
        // 第二次滤波 - Y轴
        pass_y.setInputCloud(seedregion_cloud);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(seed_point.y-y_range / 2,seed_point.y+ y_range / 2);
        pass_y.filter(*seedregion_cloud);
        if (seedregion_cloud->empty()) {
            ROS_WARN("[step4] seedregion_cloud empty after Y crop.");
            return false;
        }
        // 2. 构建搜索树
        typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(seedregion_cloud);

        // 3. 设置欧式聚类参数
        std::vector<pcl::PointIndices> hook_cluster_indices;
        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance(0.5); // 聚类的邻近距离 (m) => 调整吊钩的点间距
        ec.setMinClusterSize(20);    // 最小簇点数，避免噪声
        ec.setMaxClusterSize(25000); // 最大簇点数，避免太大误判
        ec.setSearchMethod(tree);
        ec.setInputCloud(seedregion_cloud);
        ec.extract(hook_cluster_indices);
        // 处理 hook cluster 结果
        if (hook_cluster_indices.empty()) 
        {
            ROS_INFO("[step4] no hook clusters found in seedregion.");
            // 这里认为未找到 hook，清空并返回 false（因为 seed region 本应包含吊钩）
            hookClusterInfo.clearAll();
            return false;
        }

//============================== 提取吊钩 Hook 点云 =====================================
        //  4. 遍历每个簇，提取点云
        typename pcl::PointCloud<PointT>::Ptr cloud_hook(new pcl::PointCloud<PointT>);
        if(hook_cluster_indices.size() == 1){
            hookClusterInfo.id= 0;
            for (int idx : hook_cluster_indices[0].indices)
            {
                cloud_hook->push_back((*seedregion_cloud)[idx]);
            }
            hookClusterInfo.cloud = cloud_hook;
            hookClusterInfo.centroid = computeCentroid(cloud_hook);
            hookClusterInfo.distance = pointToLineDistance(PointT(hookClusterInfo.centroid.x(), hookClusterInfo.centroid.y(), hookClusterInfo.centroid.z()),point_online_1, point_online_2);
        }
        else if (hook_cluster_indices.size() > 1)
        {
            //存在干扰，取离直线距离最近点云
            // 已知钢丝绳直线两个点 line_A, line_B
            std::vector<ClusterInfo<PointT>> clusters;
            int cluster_id = 0;

            for (const auto &indices : hook_cluster_indices)
            {
                typename pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
                for (int idx : indices.indices)
                {
                    cloud_cluster->push_back((*seedregion_cloud)[idx]);
                }

                ClusterInfo<PointT> info;
                info.id = cluster_id++;
                info.cloud = cloud_cluster;
                info.centroid = computeCentroid(cloud_cluster);
                info.distance = pointToLineDistance(
                    PointT(info.centroid.x(), info.centroid.y(), info.centroid.z()),
                    point_online_1, point_online_2);
                if (fabs(info.distance) < 3.0)
                { // 距离直线太远的簇舍弃
                    clusters.push_back(info);
                }
            }
            if(clusters.size()==0){
                hookClusterInfo.centroid = seed_point_eigen_;
            }
            else if(clusters.size()==1){
                hookClusterInfo = clusters[0];
            }
            else{
               // 按质心到直线距离排序
                std::sort(clusters.begin(), clusters.end(),
                        [](const ClusterInfo<PointT> &a, const ClusterInfo<PointT> &b){
                            return a.distance < b.distance;
                        });
                hookClusterInfo = clusters[0];
            }
        }
        else{
            hookClusterInfo.centroid = seed_point_eigen_;
        }
        // 获取开始时间
        auto end_step4 = std::chrono::high_resolution_clock::now();
        elapsed = end_step4 - end_step3;
        std::cout << "step4 took " << elapsed.count() << " seconds." << std::endl;




//============================== 提取吊载 Load 点云 =====================================
        loadClusterInfo.clearAll();
        if(load_exist_flag){
            // // 1. 提取种子区域点云（圆柱区域）
            float radius = 3.0f; // 圆柱半径，可调
            // -------------------- Step 5.1: 粗裁剪 (长方体) --------------------
            pcl::CropBox<PointT> crop;
            crop.setInputCloud(cloud);

            Eigen::Vector4f min_pt(hookClusterInfo.centroid.x() - 1.0f, 
                                hookClusterInfo.centroid.y() - radius, 
                                hookClusterInfo.centroid.z() - radius,  // 下方延伸
                                1.0f);
            Eigen::Vector4f max_pt(hookClusterInfo.centroid.x() + 50.0f, 
                                hookClusterInfo.centroid.y() + radius, 
                                hookClusterInfo.centroid.z() + radius,   // 上方略高于吊钩
                                1.0f);

            crop.setMin(min_pt);
            crop.setMax(max_pt);

            std::vector<int> coarse_cloud_indices;
            crop.filter(coarse_cloud_indices);
            if (coarse_cloud_indices.empty()) {
                ROS_WARN("[step5] CropBox returned no points (expected load).");
                return false;
            }

            // -------------------- Step 5.2: 精裁剪 (圆柱 + 下界限制) --------------------
            // 圆柱参数
            Eigen::Vector3f line_dir = (point_online_2 - point_online_1);
            float denom2 = line_dir.squaredNorm();
            float radius2 = radius * radius;

            std::vector<int> tmp_indices;
            tmp_indices.reserve(coarse_cloud_indices.size());

            // OpenMP 并行循环
            #pragma omp parallel
            {
                std::vector<int> local_point_indices;
                local_point_indices.reserve(1024);

                #pragma omp for nowait
                for (int i=0 ;i<coarse_cloud_indices.size();i++)
                {
                    const auto& pt = cloud->points[coarse_cloud_indices[i]];
                    Eigen::Vector3f v = pt.getVector3fMap() - point_online_1;
                    float cross_norm2 = line_dir.cross(v).squaredNorm();
                    float dist2 = cross_norm2 / denom2;

                    // 距离小于半径 && 在吊钩下方 1m 以上
                    if (dist2 < radius2 && pt.x > hookClusterInfo.centroid.x() + 1.0f)
                    {
                        local_point_indices.push_back(coarse_cloud_indices[i]);
                    }
                }

                // 每个线程将局部结果合并到全局
                #pragma omp critical
                {
                    tmp_indices.insert(
                        tmp_indices.end(),
                        local_point_indices.begin(),
                        local_point_indices.end()
                    );
                }        
            }
            if (tmp_indices.empty()) {
                ROS_WARN("[step5] after cylinder filter, no candidate points.");
                return false;
            }
            typename pcl::PointCloud<PointT>::Ptr load_roi_cloud(new pcl::PointCloud<PointT>);
            pcl::copyPointCloud(*cloud, tmp_indices, *load_roi_cloud);

            // -------------------- Step 5.3: 输出结果 --------------------
            load_roi_cloud->width  = load_roi_cloud->points.size();
            load_roi_cloud->height = 1;
            load_roi_cloud->is_dense = true;
            // 获取开始时间
            auto end_step5 = std::chrono::high_resolution_clock::now();
            elapsed = end_step5 - end_step4;
            std::cout << "step5 took " << elapsed.count() << " seconds." <<" load_roi_cloud->width"<< load_roi_cloud->width<< std::endl;
            // 2. 构建搜索树
            tree->setInputCloud(load_roi_cloud);

            // 3. 设置欧式聚类参数
            std::vector<pcl::PointIndices> load_cluster_indices;
            // ec.setClusterTolerance(0.5); // 聚类的邻近距离 (m) => 调整吊钩的点间距
            // ec.setMinClusterSize(20);    // 最小簇点数，避免噪声
            // ec.setMaxClusterSize(25000); // 最大簇点数，避免太大误判
            ec.setSearchMethod(tree);
            ec.setInputCloud(load_roi_cloud);
            ec.extract(load_cluster_indices);


            if (load_cluster_indices.empty()) {
                ROS_INFO("[step5] no load clusters extracted.");
                // 期待载荷但未聚类成功 -> 视为失败
                return false;
            }

            // 遍历并过滤
            std::vector<ClusterInfo<PointT>> clusters;
            int cid = 0;
            for (const auto &indices : load_cluster_indices) {
                typename pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
                std::vector<int> cluster_indices;
                for (int idx : indices.indices) {
                    cluster_indices.push_back(tmp_indices[idx]);
                    cloud_cluster->push_back((*cloud)[tmp_indices[idx]]);
                }
                ClusterInfo<PointT> info;
                info.id = cid++;
                info.cloud = cloud_cluster;
                info.centroid = computeCentroid(cloud_cluster);
                info.distance = pointToLineDistance(PointT(info.centroid.x(), info.centroid.y(), info.centroid.z()),
                                                point_online_1, point_online_2);
                info.indices = cluster_indices;
                if (std::fabs(info.distance) < MAX_DISTANCE_TO_LINE_FOR_LOAD) clusters.push_back(info);
            }

            if (clusters.empty()) {
                ROS_INFO("[step5] no valid load clusters after distance filtering.");
                return false;
            }
            // 选最优 cluster 并做 region grow
            std::sort(clusters.begin(), clusters.end(), [](const ClusterInfo<PointT>& a, const ClusterInfo<PointT>& b){
                return a.distance < b.distance;
            });

            // 如果多个，挑两个最近的并按 x 值选择靠近臂尖的
            if (clusters.size() >= 2) {
                loadClusterInfo = clusters[0].centroid.x() < clusters[1].centroid.x() ? clusters[0] : clusters[1];
            } else {
                loadClusterInfo = clusters[0];
            }

            // 生长操作（可能失败）
            if (!seedRegionGrowing(cloud, loadClusterInfo.centroid, 1.0f, loadClusterInfo.cloud)) {
                ROS_WARN("[step5] seedRegionGrowing failed.");
                return false;
            }  
            
        }
        else{
            loadClusterInfo.clearAll();       
        }

        // 获取开始时间
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "total took " << elapsed.count() << " seconds." << std::endl;

//============================== 可视化 =====================================
        #ifdef VIS_DEBUG_MODE
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Simple PointT Cloud Visualization"));
            viewer->setBackgroundColor(0, 0, 0);
            // 场景点云
            pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_color(cloud, 255, 255, 255);
            viewer->addPointCloud<PointT>(cloud, cloud_color, "cloud_color");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_color");
            // 圆柱区域点云可视化
            pcl::visualization::PointCloudColorHandlerCustom<PointT> seedregion_cloud_color(seedregion_cloud, 0, 0, 255);
            viewer->addPointCloud<PointT>(seedregion_cloud, seedregion_cloud_color, "seedregion_cloud_color");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "seedregion_cloud_color");
            viewer->addCube(originalCenter, rotation, depth_filter, height_filter, width_filter, "filter region");
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "filter region");
            // 绳索点云
            pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_line_color(cloud_filtered, 255, 0, 255);
            viewer->addPointCloud<PointT>(cloud_filtered, cloud_line_color, "cloud_line_color");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_line_color");

            viewer->addLine<PointT>(p1, p2, 1.0, 0.0, 0.0, "pca_line"); // 红色直线
            PointT point_online_pcl(weighted_centerid[0],
                                        weighted_centerid[1],
                                        weighted_centerid[2]);
            viewer->addSphere(point_online_pcl, 0.5, 0.1, 1.0, 0.0, "point_online_pcl"); // 红色球体

            //clust center
            viewer->addSphere(seed_point, 0.5, 0.0, 1.0, 0.0, "seed_point"); // 绿色球体


            viewer->addCube(seed_point_eigen_, rotation, x_range,y_range, z_range, "filter region2");
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "filter region2");

            // 吊钩点云可视化
            if(hookClusterInfo.exist()){
                pcl::visualization::PointCloudColorHandlerCustom<PointT> hook_cluster_color(hookClusterInfo.cloud, 0, 255, 0);
                viewer->addPointCloud<PointT>(hookClusterInfo.cloud, hook_cluster_color, "hook_cluster_color");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "hook_cluster_color");
            }
            // 吊载点云可视化
            if(loadClusterInfo.exist()){
                pcl::visualization::PointCloudColorHandlerCustom<PointT> load_cluster_color(loadClusterInfo.cloud, 255, 255, 0);
                viewer->addPointCloud<PointT>(loadClusterInfo.cloud, load_cluster_color, "load_cluster_color");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "load_cluster_color");
            }

            while (!viewer->wasStopped())
            {
                viewer->spinOnce();
            }
        #endif

        return true;
    }
    catch (const pcl::PCLException& e)
    {
        ROS_ERROR("[getHookLoadCluster] PCLException: %s", e.what());
        return false;
    }
    catch(const std::exception &e){
        ROS_ERROR("[getHookLoadCluster] exception: %s", e.what());
        return false;
    }
    catch(...){
        ROS_ERROR("[getHookLoadCluster] unknown exception occurred.");
        return false;
    }
}

// 只写一次
template class HookLoadPositionAcquirer<pcl::PointXYZ>;
