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
import FoundationModels

import InferenceKit
@testable import InferenceKitFoundationModels

@Suite("FoundationModelsEngine")
struct FoundationModelsEngineTests {
    @Test("Engine model defaults to the Foundation Models configuration")
    func modelDefault() throws {
        guard #available(macOS 26.0, *) else { return }
        let engine = FoundationModelsEngine()
        #expect(engine.model.id == "apple-foundation-model")
        #expect(engine.model.displayName == "Foundation Models (Apple Intelligence)")
    }

    @Test("Engine streams a response when Apple Intelligence is available")
    func streamResponse() async throws {
        guard #available(macOS 26.0, *) else { return }
        let engine = FoundationModelsEngine()

        let availability = await MainActor.run {
            SystemLanguageModel.default.availability
        }
        guard case .available = availability else { return }

        var text = ""
        var completed = false
        for try await event in engine.stream(InferenceRequest(messages: [.user("Say hi in exactly two words.")])) {
            switch event {
            case .textDelta(let delta):
                text += delta
            case .completed(let completion):
                completed = true
                #expect(completion.finishReason == .stop)
                #expect(completion.assistantMessage?.content.first?.text == text)
            default:
                break
            }
        }
        #expect(completed)
        #expect(!text.isEmpty)
    }
}
