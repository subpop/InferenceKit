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

import Foundation

extension ModelCatalog {
    /// The catalog of known models, compiled into the library. Defined as
    /// code rather than a bundle resource so it is available in every
    /// distribution shape — apps, embedded frameworks, XCFrameworks, static
    /// libraries — with no runtime bundle lookup that could fail.
    public static let bundled = ModelCatalog(providers: [
        "anthropic": [
            "claude-opus-5-5": ModelConfiguration(
                id: "claude-opus-5-5",
                displayName: "Claude Opus 5.5",
                contextWindowTokens: 1_000_000,
                maxOutputTokens: 128_000,
                thinking: ThinkingCapability(
                    supportedEfforts: [.low, .medium, .high, .xhigh, .max],
                    alwaysOn: true
                ),
                inputModalities: [.text, .image, .pdf]
            ),
            "claude-sonnet-5-5": ModelConfiguration(
                id: "claude-sonnet-5-5",
                displayName: "Claude Sonnet 5.5",
                contextWindowTokens: 1_000_000,
                maxOutputTokens: 128_000,
                thinking: ThinkingCapability(
                    supportedEfforts: [.low, .medium, .high, .xhigh, .max],
                    alwaysOn: true
                ),
                inputModalities: [.text, .image, .pdf]
            ),
            "claude-haiku-5-5": ModelConfiguration(
                id: "claude-haiku-5-5",
                displayName: "Claude Haiku 5.5",
                contextWindowTokens: 1_000_000,
                maxOutputTokens: 128_000,
                thinking: ThinkingCapability(
                    supportedEfforts: [.low, .medium, .high, .xhigh, .max]
                ),
                inputModalities: [.text, .image, .pdf]
            ),
            "claude-opus-4-5": ModelConfiguration(
                id: "claude-opus-4-5",
                displayName: "Claude Opus 4.5",
                providerModelID: "claude-opus-4-5-20251101",
                contextWindowTokens: 200_000,
                maxOutputTokens: 32_000,
                thinking: ThinkingCapability(budgetRange: 1024...32_000),
                inputModalities: [.text, .image, .pdf]
            ),
            "claude-sonnet-4-5": ModelConfiguration(
                id: "claude-sonnet-4-5",
                displayName: "Claude Sonnet 4.5",
                providerModelID: "claude-sonnet-4-5-20250929",
                contextWindowTokens: 200_000,
                maxOutputTokens: 64_000,
                thinking: ThinkingCapability(budgetRange: 1024...64_000),
                inputModalities: [.text, .image, .pdf]
            ),
            "claude-haiku-4-5": ModelConfiguration(
                id: "claude-haiku-4-5",
                displayName: "Claude Haiku 4.5",
                providerModelID: "claude-haiku-4-5-20251001",
                contextWindowTokens: 200_000,
                maxOutputTokens: 32_000,
                thinking: ThinkingCapability(budgetRange: 1024...32_000),
                inputModalities: [.text, .image, .pdf]
            ),
            "claude-sonnet-4": ModelConfiguration(
                id: "claude-sonnet-4",
                displayName: "Claude Sonnet 4",
                providerModelID: "claude-sonnet-4-20250514",
                contextWindowTokens: 200_000,
                maxOutputTokens: 16_000,
                thinking: ThinkingCapability(budgetRange: 1024...16_000),
                inputModalities: [.text, .image, .pdf]
            ),
        ],
        "openai": [
            "gpt-6-astra": ModelConfiguration(
                id: "gpt-6-astra",
                displayName: "GPT-6 Astra",
                contextWindowTokens: 1_050_000,
                maxOutputTokens: 128_000,
                thinking: ThinkingCapability(
                    supportedEfforts: [.low, .medium, .high, .xhigh, .max]
                ),
                inputModalities: [.text, .image]
            ),
            "gpt-6.1-sol": ModelConfiguration(
                id: "gpt-6.1-sol",
                displayName: "GPT-6.1 Sol",
                contextWindowTokens: 1_050_000,
                maxOutputTokens: 128_000,
                thinking: ThinkingCapability(
                    supportedEfforts: [.low, .medium, .high, .xhigh, .max]
                ),
                inputModalities: [.text, .image]
            ),
            "gpt-6-sol": ModelConfiguration(
                id: "gpt-6-sol",
                displayName: "GPT-6 Sol",
                contextWindowTokens: 1_050_000,
                maxOutputTokens: 128_000,
                thinking: ThinkingCapability(
                    supportedEfforts: [.none, .low, .medium, .high, .xhigh, .max]
                ),
                inputModalities: [.text, .image]
            ),
            "gpt-6-luna": ModelConfiguration(
                id: "gpt-6-luna",
                displayName: "GPT-6 Luna",
                contextWindowTokens: 1_050_000,
                maxOutputTokens: 128_000,
                thinking: ThinkingCapability(
                    supportedEfforts: [.none, .low, .medium, .high, .xhigh, .max]
                ),
                inputModalities: [.text, .image]
            ),
            "gpt-5.1": ModelConfiguration(
                id: "gpt-5.1",
                displayName: "GPT-5.1",
                contextWindowTokens: 400_000,
                maxOutputTokens: 128_000,
                thinking: ThinkingCapability(
                    supportedEfforts: [.minimal, .low, .medium, .high]
                ),
                inputModalities: [.text, .image, .pdf]
            ),
            "gpt-5": ModelConfiguration(
                id: "gpt-5",
                displayName: "GPT-5",
                contextWindowTokens: 400_000,
                maxOutputTokens: 128_000,
                thinking: ThinkingCapability(supportedEfforts: [.low, .medium, .high]),
                inputModalities: [.text, .image, .pdf]
            ),
            "gpt-5-mini": ModelConfiguration(
                id: "gpt-5-mini",
                displayName: "GPT-5 mini",
                contextWindowTokens: 400_000,
                maxOutputTokens: 128_000,
                thinking: ThinkingCapability(supportedEfforts: [.low, .medium, .high]),
                inputModalities: [.text, .image, .pdf]
            ),
            "gpt-5-nano": ModelConfiguration(
                id: "gpt-5-nano",
                displayName: "GPT-5 nano",
                contextWindowTokens: 400_000,
                maxOutputTokens: 128_000,
                thinking: ThinkingCapability(supportedEfforts: [.low, .medium, .high]),
                inputModalities: [.text, .image, .pdf]
            ),
            "gpt-4.1": ModelConfiguration(
                id: "gpt-4.1",
                displayName: "GPT-4.1",
                contextWindowTokens: 1_000_000,
                maxOutputTokens: 32_768,
                inputModalities: [.text, .image]
            ),
            "gpt-4o": ModelConfiguration(
                id: "gpt-4o",
                displayName: "GPT-4o",
                contextWindowTokens: 128_000,
                maxOutputTokens: 16_384,
                inputModalities: [.text, .image]
            ),
            "o3": ModelConfiguration(
                id: "o3",
                displayName: "o3",
                contextWindowTokens: 200_000,
                maxOutputTokens: 100_000,
                thinking: ThinkingCapability(supportedEfforts: [.low, .medium, .high]),
                inputModalities: [.text, .image]
            ),
            "o4-mini": ModelConfiguration(
                id: "o4-mini",
                displayName: "o4-mini",
                contextWindowTokens: 200_000,
                maxOutputTokens: 100_000,
                thinking: ThinkingCapability(supportedEfforts: [.low, .medium, .high]),
                inputModalities: [.text, .image]
            ),
        ],
        "google": [
            "gemini-3-pro-preview": ModelConfiguration(
                id: "gemini-3-pro-preview",
                displayName: "Gemini 3 Pro Preview",
                contextWindowTokens: 1_000_000,
                maxOutputTokens: 65_536,
                thinking: ThinkingCapability(budgetRange: 128...32_768),
                inputModalities: [.text, .image, .pdf]
            ),
            "gemini-2.5-pro": ModelConfiguration(
                id: "gemini-2.5-pro",
                displayName: "Gemini 2.5 Pro",
                contextWindowTokens: 1_000_000,
                maxOutputTokens: 65_536,
                thinking: ThinkingCapability(budgetRange: 128...32_768),
                inputModalities: [.text, .image, .pdf]
            ),
            "gemini-2.5-flash": ModelConfiguration(
                id: "gemini-2.5-flash",
                displayName: "Gemini 2.5 Flash",
                contextWindowTokens: 1_000_000,
                maxOutputTokens: 65_536,
                thinking: ThinkingCapability(budgetRange: 0...24_576),
                inputModalities: [.text, .image, .pdf]
            ),
            "gemini-2.5-flash-lite": ModelConfiguration(
                id: "gemini-2.5-flash-lite",
                displayName: "Gemini 2.5 Flash-Lite",
                contextWindowTokens: 1_000_000,
                maxOutputTokens: 65_536,
                thinking: ThinkingCapability(budgetRange: 0...24_576),
                inputModalities: [.text, .image, .pdf]
            ),
        ],
    ])
}
