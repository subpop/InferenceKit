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

/// The Gemini generateContent API:
/// `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`.
public struct GeminiGenerateContentFormat: WireFormat {
    public let id = WireFormatID.geminiGenerateContent
    public let transport = WireTransport.sse

    public init() {}

    public func makeChatRequest(
        _ request: InferenceRequest,
        provider: ProviderConfiguration,
        model: ModelConfiguration
    ) throws -> URLRequest {
        if case .effort = request.options.thinking {
            throw InferenceError.unsupportedFeature("Reasoning effort", context: id.rawValue)
        }

        var thinkingConfig: GeminiRequestBody.GenerationConfig.ThinkingConfig?
        if let thinking = request.options.thinking {
            switch thinking {
            case .disabled:
                thinkingConfig = .init(includeThoughts: false, thinkingBudget: 0)
            case .automatic:
                thinkingConfig = .init(includeThoughts: true, thinkingBudget: nil)
            case .budgetedTokens(let budget):
                thinkingConfig = .init(includeThoughts: true, thinkingBudget: budget)
            case .effort:
                throw InferenceError.unsupportedFeature("Reasoning effort", context: id.rawValue)
            }
        }

        let body = GeminiRequestBody(
            contents: try encodeContents(request.messages),
            systemInstruction: request.system.isEmpty
                ? nil
                : GeminiWireContent(
                    role: nil,
                    parts: [.object(["text": .string(request.system.joinedPrompt)])]
                ),
            tools: request.tools.isEmpty
                ? nil
                : [GeminiTools(functionDeclarations: try request.tools.map(encodeTool))],
            generationConfig: .init(
                maxOutputTokens: request.options.maxOutputTokens ?? model.maxOutputTokens,
                temperature: request.options.temperature,
                topP: request.options.topP,
                stopSequences: request.options.stopSequences,
                thinkingConfig: thinkingConfig
            )
        )

        // The method suffix must attach directly to the model id
        // (`models/{model}:streamGenerateContent`), so it is built as a
        // single path string.
        let url = provider.baseURL
            .appending(path: "models/\(model.wireID):streamGenerateContent")
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            throw InferenceError.invalidConfiguration("Invalid stream URL for provider \"\(provider.id)\".")
        }
        components.queryItems = [URLQueryItem(name: "alt", value: "sse")]

        var urlRequest = URLRequest(url: components.url ?? url)
        urlRequest.httpMethod = "POST"
        urlRequest.httpBody = try JSONEncoder().encode(body)
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return urlRequest
    }

    public func makeStreamDecoder() -> any WireStreamDecoder {
        GeminiStreamDecoder()
    }

    public func listModels(
        provider: ProviderConfiguration,
        perform: RequestPerformer
    ) async throws -> [DiscoveredModel] {
        var models: [DiscoveredModel] = []
        var pageToken: String?
        while true {
            let url = provider.baseURL.appending(path: "models")
            guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
                throw InferenceError.invalidConfiguration("Invalid models URL for provider \"\(provider.id)\".")
            }
            var queryItems = [URLQueryItem(name: "pageSize", value: "100")]
            if let pageToken {
                queryItems.append(URLQueryItem(name: "pageToken", value: pageToken))
            }
            components.queryItems = queryItems

            let urlRequest = URLRequest(url: components.url ?? url)
            let (data, http) = try await perform(Transport.finalized(urlRequest, provider: provider))
            try Transport.check(http, data: data, provider: provider, format: self)

            let page = try decodeDTO(GeminiModelsPage.self, from: data)
            models += (page.models ?? []).compactMap { item in
                guard let name = item.name,
                      item.supportedGenerationMethods?.contains("generateContent") ?? true
                else { return nil }
                return DiscoveredModel(
                    id: name.replacing("models/", with: ""),
                    displayName: item.displayName
                )
            }

            if let next = page.nextPageToken, !next.isEmpty {
                pageToken = next
            } else {
                return models
            }
        }
    }

    // MARK: - Encoding

    private func encodeTool(_ tool: ToolDefinition) throws -> GeminiFunctionDeclaration {
        guard let schema = tool.parametersSchema.objectValue else {
            throw InferenceError.invalidConfiguration(
                "Tool \"\(tool.name)\" schema must be a JSON object."
            )
        }
        return GeminiFunctionDeclaration(
            name: tool.name,
            description: tool.description,
            parameters: Self.geminiSchema(.object(schema))
        )
    }

    /// Converts a JSON Schema to Gemini's Schema format: uppercase type
    /// values, strip keys Gemini rejects (`additionalProperties`).
    static func geminiSchema(_ schema: JSONValue) -> JSONValue {
        guard var object = schema.objectValue else { return schema }
        object.removeValue(forKey: "additionalProperties")

        if case .string(let type)? = object["type"] {
            object["type"] = .string(type.uppercased())
        }
        if case .object(let properties)? = object["properties"] {
            var converted: [String: JSONValue] = [:]
            for (key, value) in properties {
                converted[key] = geminiSchema(value)
            }
            object["properties"] = .object(converted)
        }
        if let items = object["items"] {
            object["items"] = geminiSchema(items)
        }
        return .object(object)
    }

    private func encodeContents(_ messages: [Message]) throws -> [GeminiWireContent] {
        var contents: [GeminiWireContent] = []
        var pendingFunctionResponses: [JSONValue] = []

        func flushResponses() {
            guard !pendingFunctionResponses.isEmpty else { return }
            contents.append(GeminiWireContent(role: "user", parts: pendingFunctionResponses))
            pendingFunctionResponses = []
        }

        for message in messages {
            switch message.role {
            case .user:
                flushResponses()
                contents.append(GeminiWireContent(
                    role: "user",
                    parts: try message.content.map(encodeUserPart)
                ))

            case .assistant:
                flushResponses()
                var parts: [JSONValue] = []
                for block in message.content {
                    if let part = try encodeAssistantPart(block) {
                        parts.append(part)
                    }
                }
                if !parts.isEmpty {
                    contents.append(GeminiWireContent(role: "model", parts: parts))
                }

            case .tool:
                for block in message.content {
                    guard case .toolResult(let result) = block else {
                        throw InferenceError.invalidConfiguration(
                            "Tool messages may only contain tool results."
                        )
                    }
                    guard let name = result.name else {
                        throw InferenceError.invalidConfiguration(
                            "Gemini tool results require the tool name."
                        )
                    }
                    var functionResponse: [String: JSONValue] = [
                        "name": .string(name),
                        "response": .object(["output": .string(result.output)]),
                    ]
                    if !result.callID.isEmpty {
                        functionResponse["id"] = .string(result.callID)
                    }
                    pendingFunctionResponses.append(.object(["functionResponse": .object(functionResponse)]))
                }
            }
        }
        flushResponses()
        return contents
    }

    private func encodeUserPart(_ block: ContentBlock) throws -> JSONValue {
        switch block {
        case .text(let text):
            return .object(["text": .string(text)])

        case .image(let image):
            if let data = image.data {
                return .object(["inlineData": .object([
                    "mimeType": .string(image.mediaType),
                    "data": .string(data.base64EncodedString()),
                ])])
            }
            if let url = image.url, let data = try? Data(contentsOf: url) {
                return .object(["inlineData": .object([
                    "mimeType": .string(image.mediaType),
                    "data": .string(data.base64EncodedString()),
                ])])
            }
            throw InferenceError.invalidConfiguration("Image block has neither data nor a loadable url.")

        case .file(let file):
            if file.isPDF || file.mediaType.hasPrefix("image/") {
                return .object(["inlineData": .object([
                    "mimeType": .string(file.mediaType),
                    "data": .string(file.data.base64EncodedString()),
                ])])
            }
            if file.isText, let text = file.textContent {
                return .object(["text": .string("[\(file.filename)]:\n\(text)")])
            }
            throw InferenceError.unsupportedFeature("Binary file attachments", context: id.rawValue)

        case .toolResult:
            throw InferenceError.invalidConfiguration(
                "Tool results belong in tool messages."
            )

        case .thinking, .toolCall:
            throw InferenceError.invalidConfiguration(
                "Block is not valid in a user message."
            )
        }
    }

    private func encodeAssistantPart(_ block: ContentBlock) throws -> JSONValue? {
        switch block {
        case .thinking(let thinking):
            var part: [String: JSONValue] = [
                "text": .string(thinking.text),
                "thought": .bool(true),
            ]
            if let signature = thinking.signature {
                part["thoughtSignature"] = .string(signature)
            }
            return .object(part)

        case .text(let text):
            return .object(["text": .string(text)])

        case .toolCall(let call):
            var functionCall: [String: JSONValue] = [
                "name": .string(call.name),
            ]
            if !call.id.isEmpty {
                functionCall["id"] = .string(call.id)
            }
            if let arguments = try? JSONValue(parsing: call.arguments), arguments.objectValue != nil {
                functionCall["args"] = arguments
            } else {
                functionCall["args"] = .object([:])
            }
            return .object(["functionCall": .object(functionCall)])

        case .toolResult:
            throw InferenceError.invalidConfiguration(
                "Tool results belong in tool messages."
            )

        case .image, .file:
            throw InferenceError.unsupportedFeature("Attachments in assistant messages", context: id.rawValue)
        }
    }
}
