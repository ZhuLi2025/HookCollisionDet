#include "CollisionDetector.h"

template <typename PointT>
CollisionDetector<PointT>::CollisionDetector(){

}
template <typename PointT>
CollisionDetector<PointT>::~CollisionDetector(){
    
}

//获取环境障碍物包围盒
template<typename PointT> 
bool CollisionDetector<PointT>::EnvCloudGeometryConstruct(const typename pcl::PointCloud<PointT>::Ptr& env_cloud , std::vector<OBB<PointT>>& obstaclesOBB){
    //欧式聚类
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(env_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.5);  // 聚类半径50cm
    ec.setMinClusterSize(10);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(env_cloud);
    ec.extract(cluster_indices);

    std::vector<OBB<PointT>> obb_list;
    int cluster_id = 0;
    if(cluster_indices.size()==0){
        ROS_INFO("[Collision Detecte] No obstacle cluster found.");
        return false;
    }
    else{
        ROS_INFO("[Collision Detecte] Found %d obstacle clusters.",cluster_indices.size());
        for (auto& indices : cluster_indices) {
            typename pcl::PointCloud<PointT>::Ptr cluster_cloud(new pcl::PointCloud<PointT>);
            for (int idx : indices.indices) {
                cluster_cloud->push_back((*env_cloud)[idx]);
            }
            OBB<PointT> obb(cluster_cloud);
            obb_list.push_back(obb);
        }
        return true;
    }

}
//获取虚拟吊钩包围盒
template<typename PointT> 
bool CollisionDetector<PointT>::VirtualHookGeometryConstruct(){
}


// 将 OBB 局部坐标系的点转换到世界坐标系
template<typename PointT>
inline Eigen::Vector3f CollisionDetector<PointT>::localToWorld(const OBB<PointT>& obb, const Eigen::Vector3f& local_pt) {
    // world = R * local + position
    return obb.rotation * local_pt + obb.position;
}

// 在 OBB 的表面采样（包含 6 个面），返回世界坐标系下的采样点
// resolution: 采样格距，单位同点云（例如 0.1 = 每 10cm 一个点）
template<typename PointT>
std::vector<Eigen::Vector3f> CollisionDetector<PointT>::sampleOBBSurface(const OBB<PointT>& obb, float resolution)
{
    std::vector<Eigen::Vector3f> samples;
    // 局部坐标下的 min/max
    const Eigen::Vector3f& pmin = obb.min_point;
    const Eigen::Vector3f& pmax = obb.max_point;

    // 面的范围：在局部坐标系下，每个面是两个坐标维度的矩形
    // 我们对 6 个面逐一采样： (x,y) 面 at z=min/max, (x,z) 面 at y=min/max, (y,z) 面 at x=min/max
    auto sampleRect = [&](int dim_fixed, float fixed_val, int dim_a, int dim_b) {
        float a_min = (dim_a==0? pmin.x() : (dim_a==1? pmin.y() : pmin.z()));
        float a_max = (dim_a==0? pmax.x() : (dim_a==1? pmax.y() : pmax.z()));
        float b_min = (dim_b==0? pmin.x() : (dim_b==1? pmin.y() : pmin.z()));
        float b_max = (dim_b==0? pmax.x() : (dim_b==1? pmax.y() : pmax.z()));

        // 保证至少有一个采样点在每个方向（防止分母为0）
        int na = std::max(1, static_cast<int>(std::ceil((a_max - a_min) / resolution)));
        int nb = std::max(1, static_cast<int>(std::ceil((b_max - b_min) / resolution)));

        for (int ia = 0; ia <= na; ++ia) {
            float a = a_min + (a_max - a_min) * (static_cast<float>(ia) / static_cast<float>(na));
            for (int ib = 0; ib <= nb; ++ib) {
                float b = b_min + (b_max - b_min) * (static_cast<float>(ib) / static_cast<float>(nb));
                Eigen::Vector3f local_pt;
                // fill by dimension
                for (int d = 0; d < 3; ++d) {
                    if (d == dim_fixed) {
                        local_pt[d] = fixed_val;
                    } else if (d == dim_a) {
                        local_pt[d] = a;
                    } else { // dim_b
                        local_pt[d] = b;
                    }
                }
                samples.push_back(localToWorld(obb, local_pt));
            }
        }
    };

    // faces: z = min, z = max  (dim_fixed = 2, dim_a=0, dim_b=1)
    sampleRect(2, pmin.z(), 0, 1);
    sampleRect(2, pmax.z(), 0, 1);
    // faces: y = min, y = max  (dim_fixed = 1, dim_a=0, dim_b=2)
    sampleRect(1, pmin.y(), 0, 2);
    sampleRect(1, pmax.y(), 0, 2);
    // faces: x = min, x = max  (dim_fixed = 0, dim_a=1, dim_b=2)
    sampleRect(0, pmin.x(), 1, 2);
    sampleRect(0, pmax.x(), 1, 2);

    // optional: 去重（相同角/边在多个面可能重复采样）——简单去重（基于坐标）
    std::vector<Eigen::Vector3f> uniq;
    uniq.reserve(samples.size());
    const float eps = 1e-6f;
    for (const auto& p : samples) {
        bool found = false;
        for (const auto& q : uniq) {
            if ((p - q).squaredNorm() < eps) { found = true; break; }
        }
        if (!found) uniq.push_back(p);
    }
    return uniq;
}


/**
 * @brief 计算 OBB 表面到环境点云的最近距离。
 *
 * @tparam PointT         点云类型（如 pcl::PointXYZ）
 * @param env_cloud        输入环境点云（已滤波、裁剪）
 * @param obb              输入目标 OBB
 * @param[out] min_distance 输出最小距离（单位：m）
 * @param[out] direction    输出方向向量（从 OBB 表面指向最近障碍点的单位向量）
 * @param sample_resolution OBB 表面采样间距（m）
 * @param threads           并行线程数（<=0 表示自动）
 * @return bool             true 表示计算成功，false 表示失败（如点云为空或异常）
 */
template<typename PointT>
bool CollisionDetector<PointT>::computeMinDistance(
    const typename pcl::PointCloud<PointT>::ConstPtr& env_cloud,
    const OBB<PointT>& obb,
    float& min_distance,
    Eigen::Vector3f& direction,
    float sample_resolution,
    int threads)
{
    min_distance = std::numeric_limits<float>::infinity();
    direction = Eigen::Vector3f::Zero();

    try {
        // 输入检查
        if (!env_cloud || env_cloud->empty()) {
            return false;
        }

        // 1) 生成采样点（世界坐标）
        std::vector<Eigen::Vector3f> samples = sampleOBBSurface(obb, sample_resolution);
        if (samples.empty()) return false;

        // 2) 构建 KD-Tree
        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(env_cloud);

        std::mutex kd_mutex;
        float min_dist2 = std::numeric_limits<float>::infinity();
        Eigen::Vector3f min_dir = Eigen::Vector3f::Zero();

#ifdef USE_OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#endif

#pragma omp parallel
        {
            float local_min_dist2 = std::numeric_limits<float>::infinity();
            Eigen::Vector3f local_min_dir = Eigen::Vector3f::Zero();

#pragma omp for schedule(static)
            for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
                const Eigen::Vector3f& sp = samples[i];
                PointT query_pt(sp.x(), sp.y(), sp.z());

                std::vector<int> nn_idx(1);
                std::vector<float> nn_dist2(1);

                // KD-tree 查询加锁保护
                {
                    std::lock_guard<std::mutex> lock(kd_mutex);
                    if (kdtree.nearestKSearch(query_pt, 1, nn_idx, nn_dist2) <= 0)
                        continue;
                }

                float d2 = nn_dist2[0];
                if (d2 < local_min_dist2) {
                    local_min_dist2 = d2;
                    const auto& np = env_cloud->points[nn_idx[0]];
                    Eigen::Vector3f vec(np.x, np.y, np.z);
                    vec -= sp;

                    float nrm = vec.norm();
                    if (nrm > 1e-9f) vec /= nrm;
                    else vec.setZero();

                    local_min_dir = vec;
                }
            }

#pragma omp critical
            {
                if (local_min_dist2 < min_dist2) {
                    min_dist2 = local_min_dist2;
                    min_dir = local_min_dir;
                }
            }
        } // end parallel

        if (!std::isfinite(min_dist2)) return false;

        min_distance = std::sqrt(min_dist2);
        direction = min_dir;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[computeMinDistance:OBB] Exception: " << e.what() << std::endl;
        return false;
    }
    catch (...) {
        std::cerr << "[computeMinDistance:OBB] Unknown exception occurred." << std::endl;
        return false;
    }
}

/**
 * @brief 计算吊钩虚拟点到环境点云的最近距离。
 *
 * @tparam PointT          点云类型（如 pcl::PointXYZ）
 * @param env_cloud         输入环境点云（已滤波、裁剪）
 * @param hook_position     吊钩虚拟点坐标（世界坐标系）
 * @param[out] min_distance 输出最小距离（单位：m）
 * @param[out] direction    输出方向向量（从吊钩指向最近障碍点的单位向量）
 * @return bool             true 表示计算成功，false 表示失败（如点云为空或异常）
 */
template<typename PointT>
bool CollisionDetector<PointT>::computeMinDistance(
    const typename pcl::PointCloud<PointT>::ConstPtr& env_cloud,
    const Eigen::Vector3f& hook_position,
    float& min_distance,
    Eigen::Vector3f& direction)
{
    min_distance = std::numeric_limits<float>::infinity();
    direction = Eigen::Vector3f::Zero();

    try {
        if (!env_cloud || env_cloud->empty()) {
            return false;
        }

        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(env_cloud);

        PointT query_pt(hook_position.x(), hook_position.y(), hook_position.z());
        std::vector<int> nn_idx(1);
        std::vector<float> nn_dist2(1);

        if (kdtree.nearestKSearch(query_pt, 1, nn_idx, nn_dist2) <= 0) {
            return false;
        }

        const auto& np = env_cloud->points[nn_idx[0]];
        Eigen::Vector3f nearest(np.x, np.y, np.z);

        Eigen::Vector3f vec = nearest - hook_position;
        float nrm = vec.norm();
        if (nrm > 1e-9f) vec /= nrm;
        else vec.setZero();

        direction = vec;
        min_distance = std::sqrt(nn_dist2[0]);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[computeMinDistance:Hook] Exception: " << e.what() << std::endl;
        return false;
    }
    catch (...) {
        std::cerr << "[computeMinDistance:Hook] Unknown exception occurred." << std::endl;
        return false;
    }
}


//简单的AABB干涉定性判定是否发生碰撞，无距离与方向信息
template<typename PointT> 
bool CollisionDetector<PointT>::collisionDetecte(const OBB<PointT>& target ,const std::vector<OBB<PointT>>& obstacles,Eigen::Vector3f& direction,float& distance){
    bool collision = false;
    distance = std::numeric_limits<float>::max();
    direction = Eigen::Vector3f::Zero();

    for (const auto& obs : obstacles)
    {
        // --------- 1. 近似 OBB 相交判断（用 AABB 近似） ----------
        Eigen::Vector3f diff = obs.position - target.position;
        Eigen::Vector3f size_target = target.max_point - target.min_point;
        Eigen::Vector3f size_obs    = obs.max_point - obs.min_point;

        bool overlap_x = std::abs(diff.x()) <= (size_target.x()/2.0f + size_obs.x()/2.0f);
        bool overlap_y = std::abs(diff.y()) <= (size_target.y()/2.0f + size_obs.y()/2.0f);
        bool overlap_z = std::abs(diff.z()) <= (size_target.z()/2.0f + size_obs.z()/2.0f);

        if (overlap_x && overlap_y && overlap_z)
        {
            collision = true;

            // --------- 2. 计算碰撞方向 ----------
            Eigen::Vector3f dir = diff.normalized();

            // --------- 3. 计算最近距离（中心到中心距离减去尺寸） ----------
            float dist_x = std::abs(diff.x()) - (size_target.x()/2.0f + size_obs.x()/2.0f);
            float dist_y = std::abs(diff.y()) - (size_target.y()/2.0f + size_obs.y()/2.0f);
            float dist_z = std::abs(diff.z()) - (size_target.z()/2.0f + size_obs.z()/2.0f);

            float min_dist = std::max({dist_x, dist_y, dist_z, 0.0f}); // 防止负值

            // 更新最小距离和方向
            if (min_dist < distance)
            {
                distance = min_dist;
                direction = dir;
            }
        }
    }

    return collision;
}


// 只写一次
template class CollisionDetector<pcl::PointXYZ>;
