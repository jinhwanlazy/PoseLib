#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/poselib.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/lm_impl.h>
#include <PoseLib/robust/optim/relative.h>
#include <PoseLib/robust/utils.h>

#include <cmath>

using namespace poselib;

namespace test::calibrated_relpose {

CameraPose random_camera(test_rng::Rng &rng) {
    Eigen::Vector3d cc = test_rng::symmetric_vec3(rng);
    cc.normalize();
    cc *= 2.0;

    Eigen::Vector3d p = test_rng::symmetric_vec3(rng, 0.1);

    Eigen::Vector3d r3 = p - cc;
    r3.normalize();

    Eigen::Vector3d r2 = r3.cross(Eigen::Vector3d::UnitX());
    r2.normalize();
    Eigen::Vector3d r1 = r2.cross(r3);

    Eigen::Matrix3d R;
    R.row(0) = r1;
    R.row(1) = r2;
    R.row(2) = r3;

    return CameraPose(R, -R * cc);
}

struct BearingScene {
    CameraPose gt_pose;
    std::vector<Point3D> bearings1;
    std::vector<Point3D> bearings2;
};

BearingScene setup_bearing_scene(size_t N, const std::string &case_name = "bearing_scene", size_t case_index = 0) {
    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    CameraPose p1 = random_camera(rng);
    CameraPose p2 = random_camera(rng);

    BearingScene scene;
    for (size_t i = 0; i < N; ++i) {
        Eigen::Vector3d Xi = test_rng::symmetric_vec3(rng);

        Eigen::Vector3d d1 = p1.apply(Xi);
        d1.normalize();
        scene.bearings1.push_back(d1);

        Eigen::Vector3d d2 = p2.apply(Xi);
        d2.normalize();
        scene.bearings2.push_back(d2);
    }

    Eigen::Matrix3d R = p2.R() * p1.R().transpose();
    Eigen::Vector3d t = p2.t - p2.R() * p1.R().transpose() * p1.t;
    scene.gt_pose = CameraPose(R, t);
    return scene;
}

// Test that the FixCameraRelativePoseRefiner with tangent-plane Jacobians
// gives zero residual at the ground truth pose.
bool test_bearing_tangent_sampson_residual_at_gt() {
    const size_t N = 20;
    BearingScene scene = setup_bearing_scene(N, "bearing_tangent_residual");

    std::vector<Eigen::Matrix<double, 3, 2>> M1(N), M2(N);
    for (size_t i = 0; i < N; ++i) {
        compute_bearing_tangent_basis(scene.bearings1[i], M1[i]);
        compute_bearing_tangent_basis(scene.bearings2[i], M2[i]);
    }

    NormalAccumulator acc;
    FixCameraRelativePoseRefiner refiner(scene.bearings1, scene.bearings2, M1, M2);
    acc.initialize(refiner.num_params);

    acc.reset_residual();
    refiner.compute_residual(acc, scene.gt_pose);
    double residual = acc.get_residual();
    REQUIRE_SMALL(residual, 1e-6);

    acc.reset_jacobian();
    refiner.compute_jacobian(acc, scene.gt_pose);
    REQUIRE_SMALL(acc.Jtr.norm(), 1e-6);

    return true;
}

// Test the Jacobian of FixCameraRelativePoseRefiner with tangent-plane M
// against finite differences.
bool test_bearing_tangent_sampson_jacobian() {
    const size_t N = 20;
    BearingScene scene = setup_bearing_scene(N, "bearing_tangent_jacobian");

    std::vector<Eigen::Matrix<double, 3, 2>> M1(N), M2(N);
    for (size_t i = 0; i < N; ++i) {
        compute_bearing_tangent_basis(scene.bearings1[i], M1[i]);
        compute_bearing_tangent_basis(scene.bearings2[i], M2[i]);
    }

    FixCameraRelativePoseRefiner<UniformWeightVector, TestAccumulator> refiner(scene.bearings1, scene.bearings2, M1, M2);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), CameraPose>(refiner, scene.gt_pose, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, scene.gt_pose);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, scene.gt_pose);
    double r2 = 0.0;
    for (int i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}

// Test refinement convergence with noisy bearing vectors.
bool test_bearing_tangent_sampson_refinement() {
    const size_t N = 30;
    BearingScene scene = setup_bearing_scene(N, "bearing_tangent_refinement");

    test_rng::Rng noise_rng = test_rng::make_rng("bearing_tangent_refinement_noise");
    for (size_t i = 0; i < N; ++i) {
        scene.bearings1[i] += test_rng::symmetric_vec3(noise_rng, 0.001);
        scene.bearings1[i].normalize();
        scene.bearings2[i] += test_rng::symmetric_vec3(noise_rng, 0.001);
        scene.bearings2[i].normalize();
    }

    std::vector<Eigen::Matrix<double, 3, 2>> M1(N), M2(N);
    for (size_t i = 0; i < N; ++i) {
        compute_bearing_tangent_basis(scene.bearings1[i], M1[i]);
        compute_bearing_tangent_basis(scene.bearings2[i], M2[i]);
    }

    FixCameraRelativePoseRefiner refiner(scene.bearings1, scene.bearings2, M1, M2);

    CameraPose pose = scene.gt_pose;
    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);
    REQUIRE(check_bundle_cost_and_gradient(stats, 1e-8, "test_bearing_tangent_sampson_refinement"));

    return true;
}

// End-to-end RANSAC test: all inliers, should recover GT pose exactly.
bool test_estimate_calibrated_relative_pose_all_inliers() {
    const size_t N = 50;
    BearingScene scene = setup_bearing_scene(N, "calibrated_relpose_all_inliers");

    RelativePoseOptions opt;
    opt.max_error = 1.0;

    CameraPose pose;
    std::vector<char> inliers;
    RansacStats stats = estimate_calibrated_relative_pose(scene.bearings1, scene.bearings2, opt, &pose, &inliers);

    REQUIRE(stats.num_inliers == N);

    size_t inlier_count = 0;
    for (size_t i = 0; i < N; ++i) {
        if (inliers[i])
            inlier_count++;
    }
    REQUIRE_EQ(inlier_count, N);

    Eigen::Vector3d t_gt = scene.gt_pose.t.normalized();
    Eigen::Vector3d t_est = pose.t.normalized();
    double t_err = std::min((t_gt - t_est).norm(), (t_gt + t_est).norm());
    REQUIRE_SMALL(t_err, 1e-4);

    Eigen::Matrix3d R_diff = pose.R() * scene.gt_pose.R().transpose();
    double angle_err = std::acos(std::min(1.0, std::max(-1.0, (R_diff.trace() - 1.0) / 2.0)));
    REQUIRE_SMALL(angle_err, 1e-4);

    return true;
}

// RANSAC test with outliers.
bool test_estimate_calibrated_relative_pose_with_outliers() {
    const size_t N = 100;
    const size_t N_outliers = 30;
    BearingScene scene = setup_bearing_scene(N, "calibrated_relpose_outliers");

    test_rng::Rng outlier_rng = test_rng::make_rng("calibrated_relpose_outlier_gen");
    for (size_t i = N - N_outliers; i < N; ++i) {
        scene.bearings2[i] = test_rng::symmetric_vec3(outlier_rng);
        scene.bearings2[i].normalize();
    }

    RelativePoseOptions opt;
    opt.max_error = 1.0;

    CameraPose pose;
    std::vector<char> inliers;
    RansacStats stats = estimate_calibrated_relative_pose(scene.bearings1, scene.bearings2, opt, &pose, &inliers);

    REQUIRE(stats.num_inliers >= N - N_outliers - 5);

    Eigen::Vector3d t_gt = scene.gt_pose.t.normalized();
    Eigen::Vector3d t_est = pose.t.normalized();
    double t_err = std::min((t_gt - t_est).norm(), (t_gt + t_est).norm());
    REQUIRE_SMALL(t_err, 0.05);

    Eigen::Matrix3d R_diff = pose.R() * scene.gt_pose.R().transpose();
    double angle_err = std::acos(std::min(1.0, std::max(-1.0, (R_diff.trace() - 1.0) / 2.0)));
    REQUIRE_SMALL(angle_err, 0.05);

    return true;
}

// Test that results match estimate_relative_pose with a pinhole camera.
bool test_calibrated_matches_camera_relpose() {
    const size_t N = 50;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    test_rng::Rng rng = test_rng::make_rng("calibrated_matches_camera");
    CameraPose p1 = random_camera(rng);
    CameraPose p2 = random_camera(rng);

    std::vector<Point2D> x1, x2;
    std::vector<Point3D> d1, d2;
    for (size_t i = 0; i < N; ++i) {
        Eigen::Vector3d Xi = test_rng::symmetric_vec3(rng);

        Eigen::Vector3d proj1 = p1.apply(Xi);
        Eigen::Vector2d xi1;
        camera.project(proj1, &xi1);
        x1.push_back(xi1);
        d1.push_back(proj1.normalized());

        Eigen::Vector3d proj2 = p2.apply(Xi);
        Eigen::Vector2d xi2;
        camera.project(proj2, &xi2);
        x2.push_back(xi2);
        d2.push_back(proj2.normalized());
    }

    Eigen::Matrix3d R = p2.R() * p1.R().transpose();
    Eigen::Vector3d t = p2.t - p2.R() * p1.R().transpose() * p1.t;
    CameraPose gt_pose(R, t);

    RelativePoseOptions opt_cam;
    opt_cam.max_error = 1.0;
    CameraPose pose_cam;
    std::vector<char> inliers_cam;
    estimate_relative_pose(x1, x2, camera, camera, opt_cam, &pose_cam, &inliers_cam);

    RelativePoseOptions opt_bear;
    opt_bear.max_error = 1.0;
    CameraPose pose_bear;
    std::vector<char> inliers_bear;
    estimate_calibrated_relative_pose(d1, d2, opt_bear, &pose_bear, &inliers_bear);

    Eigen::Vector3d t_cam = pose_cam.t.normalized();
    Eigen::Vector3d t_bear = pose_bear.t.normalized();
    double t_diff = std::min((t_cam - t_bear).norm(), (t_cam + t_bear).norm());
    REQUIRE_SMALL(t_diff, 0.1);

    Eigen::Matrix3d R_diff = pose_cam.R() * pose_bear.R().transpose();
    double angle_diff = std::acos(std::min(1.0, std::max(-1.0, (R_diff.trace() - 1.0) / 2.0)));
    REQUIRE_SMALL(angle_diff, 0.1);

    return true;
}

// Test with wide-angle bearing vectors (some with z < 0).
bool test_estimate_calibrated_relative_pose_wide_angle() {
    const size_t N = 50;
    test_rng::Rng rng = test_rng::make_rng("calibrated_relpose_wide_angle");

    CameraPose p1 = random_camera(rng);
    CameraPose p2 = random_camera(rng);

    std::vector<Point3D> d1, d2;
    size_t behind_count = 0;
    for (size_t i = 0; i < N; ++i) {
        Eigen::Vector3d Xi = test_rng::symmetric_vec3(rng, 3.0);

        Eigen::Vector3d b1 = p1.apply(Xi);
        b1.normalize();
        d1.push_back(b1);

        Eigen::Vector3d b2 = p2.apply(Xi);
        b2.normalize();
        d2.push_back(b2);

        if (b1.z() < 0 || b2.z() < 0)
            behind_count++;
    }

    Eigen::Matrix3d R = p2.R() * p1.R().transpose();
    Eigen::Vector3d t = p2.t - p2.R() * p1.R().transpose() * p1.t;
    CameraPose gt_pose(R, t);

    RelativePoseOptions opt;
    opt.max_error = 2.0;

    CameraPose pose;
    std::vector<char> inliers;
    RansacStats stats = estimate_calibrated_relative_pose(d1, d2, opt, &pose, &inliers);

    REQUIRE(stats.num_inliers > 10);

    Eigen::Vector3d t_gt = gt_pose.t.normalized();
    Eigen::Vector3d t_est = pose.t.normalized();
    double t_err = std::min((t_gt - t_est).norm(), (t_gt + t_est).norm());
    REQUIRE_SMALL(t_err, 0.1);

    Eigen::Matrix3d R_diff = pose.R() * gt_pose.R().transpose();
    double angle_err = std::acos(std::min(1.0, std::max(-1.0, (R_diff.trace() - 1.0) / 2.0)));
    REQUIRE_SMALL(angle_err, 0.1);

    return true;
}

} // namespace test::calibrated_relpose

using namespace test::calibrated_relpose;
std::vector<Test> register_calibrated_relpose_test() {
    return {TEST(test_bearing_tangent_sampson_residual_at_gt),
            TEST(test_bearing_tangent_sampson_jacobian),
            TEST(test_bearing_tangent_sampson_refinement),
            TEST(test_estimate_calibrated_relative_pose_all_inliers),
            TEST(test_estimate_calibrated_relative_pose_with_outliers),
            TEST(test_calibrated_matches_camera_relpose),
            TEST(test_estimate_calibrated_relative_pose_wide_angle)};
}
