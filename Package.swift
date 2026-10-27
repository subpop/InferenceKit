// swift-tools-version: 6.4
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "InferenceKit",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "InferenceKit",
            targets: ["InferenceKit"]
        ),
        .library(
            name: "InferenceKitFoundationModels",
            targets: ["InferenceKitFoundationModels"]
        ),
        .library(
            name: "InferenceKitMLX",
            targets: ["InferenceKitMLX"]
        ),
    ],
    dependencies: [
        // MLX local inference. The iOS 17 / macOS 14 floors above are driven
        // by mlx-swift-lm's platform requirements.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.32.3")),
        // Declared directly (mlx-swift-lm pins the same version) so the test
        // target can configure MLX's Metal library location.
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.32.3")),
        // HuggingFace download/tokenizer integration used by MLXHuggingFace.
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // DocC site generation for the docs workflow (command plugin only;
        // links nothing into the products).
        .package(url: "https://github.com/apple/swift-docc-plugin.git", from: "1.4.0"),
    ],
    targets: [
        .target(
            name: "InferenceKit",
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
        .target(
            name: "InferenceKitFoundationModels",
            dependencies: ["InferenceKit"],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
        .target(
            name: "InferenceKitMLX",
            dependencies: [
                "InferenceKit",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
        .testTarget(
            name: "InferenceKitTests",
            dependencies: ["InferenceKit"],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
        .testTarget(
            name: "InferenceKitFoundationModelsTests",
            dependencies: ["InferenceKit", "InferenceKitFoundationModels"],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
        .testTarget(
            name: "InferenceKitMLXTests",
            dependencies: [
                "InferenceKit",
                "InferenceKitMLX",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("ApproachableConcurrency"),
            ],
        ),
    ]
)
