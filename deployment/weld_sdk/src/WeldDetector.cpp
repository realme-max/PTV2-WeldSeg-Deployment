#include "WeldDetector.h"

#include "CoordinateRecovery.h"
#include "FeatureBuilder.h"
#include "KnnGraphBuilder.h"
#include "PointCloudLoader.h"
#include "PointSampler.h"
#include "ResultWriter.h"
#include "SegmentationPostProcessor.h"
#include "TensorRTInference.h"
#include "WeldGeometryExtractor.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace ptv2::weld
{
namespace
{

constexpr char kProductionEngineSha256[]{
    "a624601c63e99689fb67a6066ce8a6e346bc42dfa2a885e0f83c74f0ca742299"};
using Clock = std::chrono::steady_clock;

float elapsedMs(Clock::time_point started)
{
    return std::chrono::duration<float, std::milli>(Clock::now() - started).count();
}

bool isRegularFile(std::string const& path)
{
    std::error_code error;
    return std::filesystem::is_regular_file(std::filesystem::path(path), error) && !error;
}

} // namespace

class WeldDetector::Impl
{
public:
    WeldStatus initialize(WeldConfig const& config)
    {
        initialized_ = false;
        lastError_.clear();
        runtime_.release();
        try
        {
            std::string validationError;
            if (!config.validate(validationError))
            {
                return fail(WeldStatus::INVALID_CONFIG, std::move(validationError));
            }
            if (!isRegularFile(config.engine_path))
            {
                return fail(
                    WeldStatus::ENGINE_LOAD_FAILED,
                    "TensorRT Engine does not exist or is not a regular file: " + config.engine_path);
            }
            if (!isRegularFile(config.plugin_path))
            {
                return fail(
                    WeldStatus::PLUGIN_LOAD_FAILED,
                    "VoxelUnique Plugin does not exist or is not a regular file: " + config.plugin_path);
            }
            if (!runtime_.initialize(config.engine_path, config.plugin_path, kProductionEngineSha256))
            {
                WeldStatus const status = runtime_.lastError().find("Plugin") != std::string::npos
                    ? WeldStatus::PLUGIN_LOAD_FAILED : WeldStatus::ENGINE_LOAD_FAILED;
                return fail(status, runtime_.lastError());
            }
            config_ = config;
            initialized_ = true;
            return WeldStatus::SUCCESS;
        }
        catch (std::exception const& error)
        {
            return fail(WeldStatus::INVALID_CONFIG, "Unexpected initialization failure: " + std::string(error.what()));
        }
    }

    WeldStatus detect(std::string const& cloudPath, WeldResult& result)
    {
        result.clear();
        lastError_.clear();
        WeldStatus stage = WeldStatus::PREPROCESS_FAILED;
        try
        {
            auto const totalStarted = Clock::now();
            if (!initialized_)
            {
                return fail(WeldStatus::INVALID_CONFIG, "WeldDetector is not initialized");
            }
            if (cloudPath.empty())
            {
                return fail(WeldStatus::POINTCLOUD_LOAD_FAILED, "cloud_path must not be empty");
            }

            stage = WeldStatus::POINTCLOUD_LOAD_FAILED;
            std::vector<ptv2::pointcloud::PointXYZL> fullCloud;
            auto started = Clock::now();
            if (!loader_.load(cloudPath, fullCloud))
            {
                return fail(stage, loader_.lastError());
            }
            float const loadCloudMs = elapsedMs(started);

            stage = WeldStatus::PREPROCESS_FAILED;
            std::vector<ptv2::pointcloud::PointXYZL> sampled;
            started = Clock::now();
            if (!sampler_.sample(fullCloud, sampled, config_.input_points))
            {
                return fail(stage, sampler_.lastError());
            }
            float const samplingMs = elapsedMs(started);
            std::vector<float> features;
            if (!featureBuilder_.buildPointsFeature(fullCloud, sampled, features))
            {
                return fail(stage, featureBuilder_.lastError());
            }
            std::vector<float> adjacency;
            started = Clock::now();
            if (!graphBuilder_.build(sampled, adjacency))
            {
                return fail(stage, graphBuilder_.lastError());
            }
            float const adjacencyBuildMs = elapsedMs(started);

            stage = WeldStatus::INFERENCE_FAILED;
            std::vector<float> logits(
                static_cast<std::size_t>(config_.input_points)
                * static_cast<std::size_t>(config_.num_classes));
            started = Clock::now();
            if (!runtime_.infer(
                    features.data(), features.size(),
                    adjacency.data(), adjacency.size(),
                    logits.data(), logits.size()))
            {
                return fail(stage, runtime_.lastError());
            }
            float const inferenceWallMs = elapsedMs(started);

            stage = WeldStatus::POSTPROCESS_FAILED;
            started = Clock::now();
            std::vector<ptv2::pointcloud::PointXYZL> recovered;
            if (!coordinateRecovery_.recover(sampled, recovered))
            {
                return fail(stage, coordinateRecovery_.lastError());
            }
            std::vector<ptv2::postprocess::SegmentationPoint> segmentation;
            if (!postProcessor_.process(recovered, logits, segmentation))
            {
                return fail(stage, postProcessor_.lastError());
            }
            ptv2::postprocess::WeldGeometryResult geometry;
            if (!geometryExtractor_.extract(segmentation, geometry))
            {
                return fail(stage, geometryExtractor_.lastError());
            }
            float const postprocessMs = elapsedMs(started);

            result.task_id = std::filesystem::path(cloudPath).stem().string();
            result.total_points = static_cast<int>(segmentation.size());
            result.original_points = static_cast<int>(fullCloud.size());
            result.sampled_points = static_cast<int>(sampled.size());
            result.weld_points = static_cast<int>(geometry.weldPoints);
            result.weld_ratio = geometry.weldRatio;
            if (config_.enable_geometry)
            {
                std::copy(geometry.center.begin(), geometry.center.end(), result.center);
                std::copy(geometry.bboxMin.begin(), geometry.bboxMin.end(), result.bbox_min);
                std::copy(geometry.bboxMax.begin(), geometry.bboxMax.end(), result.bbox_max);
                result.length_mm = geometry.lengthMm;
            }
            result.inference_ms = runtime_.lastInferenceDeviceMs();
            result.load_cloud_ms = loadCloudMs;
            result.sampling_ms = samplingMs;
            result.adjacency_build_ms = adjacencyBuildMs;
            result.inference_wall_ms = inferenceWallMs;
            result.postprocess_ms = postprocessMs;
            result.error_recorder_errors = runtime_.errorRecorderErrors();
            result.labels.reserve(segmentation.size());
            for (auto const& point : segmentation) result.labels.push_back(point.label);

            if (!config_.output_path.empty())
            {
                std::filesystem::path const requested(config_.output_path);
                bool const legacyPrediction = requested.extension() == ".txt";
                std::filesystem::path outputDirectory = legacyPrediction
                    ? requested.parent_path() : requested;
                if (outputDirectory.empty()) outputDirectory = std::filesystem::current_path();
                std::filesystem::path const predictionPath = legacyPrediction
                    ? requested : outputDirectory / "prediction.txt";

                ptv2::postprocess::WeldGeometryResult outputGeometry = geometry;
                if (!config_.enable_geometry)
                {
                    outputGeometry.center = {};
                    outputGeometry.bboxMin = {};
                    outputGeometry.bboxMax = {};
                    outputGeometry.lengthMm = 0.0F;
                }
                if (!resultWriter_.write(
                        outputDirectory, result.task_id, segmentation, outputGeometry,
                        result.inference_ms, predictionPath))
                {
                    result.clear();
                    return fail(stage, resultWriter_.lastError());
                }
            }

            result.total_ms = elapsedMs(totalStarted);
            result.success = true;
            return WeldStatus::SUCCESS;
        }
        catch (std::exception const& error)
        {
            result.clear();
            return fail(stage, "Unexpected detection failure: " + std::string(error.what()));
        }
    }

    std::string lastError() const
    {
        return lastError_;
    }

private:
    WeldStatus fail(WeldStatus status, std::string message)
    {
        lastError_ = std::move(message);
        return status;
    }

    bool initialized_{false};
    WeldConfig config_{};
    std::string lastError_;
    ptv2::runtime::TensorRTInference runtime_;
    ptv2::pointcloud::PointCloudLoader loader_;
    ptv2::pointcloud::PointSampler sampler_{42U};
    ptv2::pointcloud::FeatureBuilder featureBuilder_;
    ptv2::pointcloud::KnnGraphBuilder graphBuilder_{6U};
    ptv2::postprocess::CoordinateRecovery coordinateRecovery_;
    ptv2::postprocess::SegmentationPostProcessor postProcessor_;
    ptv2::postprocess::WeldGeometryExtractor geometryExtractor_;
    ptv2::postprocess::ResultWriter resultWriter_;
};

WeldDetector::WeldDetector() : impl_(std::make_unique<Impl>()) {}
WeldDetector::~WeldDetector() = default;

WeldStatus WeldDetector::initialize(WeldConfig const& config)
{
    return impl_->initialize(config);
}

WeldStatus WeldDetector::detect(std::string const& cloudPath, WeldResult& result)
{
    return impl_->detect(cloudPath, result);
}

std::string WeldDetector::lastError() const
{
    return impl_->lastError();
}

} // namespace ptv2::weld
