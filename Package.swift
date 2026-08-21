// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MLXDA3",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1)
    ],
    products: [
        .library(name: "MLXDA3", targets: ["MLXDA3"]),
        .library(name: "MLXDA3Streaming", targets: ["MLXDA3Streaming"]),
        .executable(name: "da3-tool", targets: ["da3-tool"]),
        .executable(name: "da3-bench", targets: ["da3-bench"]),
        .executable(name: "da3-streaming-tool", targets: ["da3-streaming-tool"]),
        .executable(name: "da3-streaming-bench", targets: ["da3-streaming-bench"]),
        .executable(name: "da3-loop-tool", targets: ["da3-loop-tool"]),
        .executable(name: "da3-video", targets: ["da3-video"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.6"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0"),
        // GPL-3.0 (port of serizba/salad). Only `da3-loop-tool`, the DA3Demo
        // example app, and the test target link it — `MLXDA3` and
        // `MLXDA3Streaming` do not. See THIRD-PARTY-NOTICES.md.
        .package(url: "https://github.com/mnmly/mlx-swift-salad.git", from: "0.1.0")
    ],
    targets: [
        .target(
            name: "MLXDA3",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "da3-tool",
            dependencies: [
                "MLXDA3",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Tools/da3-tool"
        ),
        .executableTarget(
            name: "da3-bench",
            dependencies: [
                "MLXDA3",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Tools/da3-bench"
        ),
        .target(
            name: "MLXDA3Streaming",
            dependencies: [
                "MLXDA3",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ]
        ),
        .executableTarget(
            name: "da3-streaming-tool",
            dependencies: [
                "MLXDA3",
                "MLXDA3Streaming",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Tools/da3-streaming-tool"
        ),
        .executableTarget(
            name: "da3-streaming-bench",
            dependencies: [
                "MLXDA3",
                "MLXDA3Streaming",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Tools/da3-streaming-bench"
        ),
        .executableTarget(
            name: "da3-video",
            dependencies: [
                "MLXDA3",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Tools/da3-video"
        ),
        .executableTarget(
            name: "da3-loop-tool",
            dependencies: [
                "MLXDA3Streaming",
                .product(name: "MLXSALAD", package: "mlx-swift-salad"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Tools/da3-loop-tool"
        ),
        .testTarget(
            name: "MLXDA3Tests",
            dependencies: [
                "MLXDA3",
                "MLXDA3Streaming",
                .product(name: "MLXSALAD", package: "mlx-swift-salad")
            ]
        )
    ]
)
