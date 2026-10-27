// Copyright 2026 Link Dupont
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Testing
import Foundation
import MLX

import InferenceKit
@testable import InferenceKitMLX

@Suite("MLXEngine")
struct MLXEngineTests {
    /// A live smoke test. Opt in by setting:
    ///   INFERENCEKIT_MLX_SMOKE=1
    /// and optionally:
    ///   INFERENCEKIT_MLX_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit
    /// Downloads several hundred MB on first run; weights are cached by the
    /// Hub downloader afterwards.
    @Test(
        "Streams a response from a local MLX model",
        .disabled(
            if: ProcessInfo.processInfo.environment["INFERENCEKIT_MLX_SMOKE"] != "1",
            "Set INFERENCEKIT_MLX_SMOKE=1 to run"
        )
    )
    func smokeStream() async throws {
        Self.locateMetallib()
        let modelID = ProcessInfo.processInfo.environment["INFERENCEKIT_MLX_MODEL"]
            ?? "mlx-community/Llama-3.2-1B-Instruct-4bit"

        let engine = try await MLXEngine.load(modelID)
        let client = InferenceClient(engine: engine)

        var text = ""
        var completion: Completion?
        for try await event in client.stream(
            InferenceRequest(messages: [.user("Say hello in exactly three words.")])
        ) {
            switch event {
            case .textDelta(let delta):
                text += delta
            case .completed(let value):
                completion = value
            default:
                break
            }
        }

        #expect(!text.isEmpty)
        #expect(completion?.finishReason == .stop)
        #expect(completion?.assistantMessage?.content.first?.text == text)
    }

    /// MLX loads its Metal shaders from `mlx-swift_Cmlx.bundle` via NSBundle
    /// search paths. That works in apps and under `xctest`, but SwiftPM's
    /// `swift test` hosts tests in `swiftpm-testing-helper`, which dlopens the
    /// test bundle without registering it with NSBundle — so the search finds
    /// nothing and device creation dies with "Failed to load the default
    /// metallib". Point MLX straight at the metallib instead.
    private static func locateMetallib() {
        guard MLX.GPU.metallib == nil else { return }
        let metallibName = "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
        var candidates: [URL] = []

        // `swift test`: the helper passes the test executable's path, which
        // sits at <bundle>.xctest/Contents/MacOS/<name>.
        let arguments = CommandLine.arguments
        if let index = arguments.firstIndex(of: "--test-bundle-path"), index + 1 < arguments.count {
            let executable = URL(fileURLWithPath: arguments[index + 1])
            let contents = executable
                .deletingLastPathComponent()  // …/Contents/MacOS
                .deletingLastPathComponent()  // …/Contents
            let bundle = contents.deletingLastPathComponent()  // ….xctest
            candidates.append(contents.appending(path: "Resources/\(metallibName)"))
            candidates.append(bundle.appending(path: metallibName))
        }

        // Bundle-hosted contexts (xctest, Xcode): the Cmlx bundle is copied
        // into the host's resources.
        for bundle in Bundle.allBundles + [Bundle.main] {
            if let resources = bundle.resourceURL {
                candidates.append(resources.appending(path: metallibName))
            }
        }

        let fileManager = FileManager.default
        if let url = candidates.first(where: { fileManager.fileExists(atPath: $0.path(percentEncoded: false)) }) {
            MLX.GPU.metallib = url
        }
        // Otherwise, leave MLX's own search in charge.
    }
}
