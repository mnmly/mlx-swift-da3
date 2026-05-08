import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var runner = PipelineRunner()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("DA3 Demo")
                .font(.title2.bold())

            Picker("Mode", selection: $runner.mode) {
                ForEach(PipelineRunner.Mode.allCases) { m in
                    Text(m.rawValue).tag(m)
                }
            }
            .pickerStyle(.segmented)

            GroupBox("Inputs") {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                    if runner.mode == .singleImage {
                        pickerRow(label: "Image", url: runner.imageURL) {
                            if let url = pickFile(types: [.image, .png, .jpeg, .heic]) {
                                runner.imageURL = url
                            }
                        }
                    } else {
                        pickerRow(label: "Video", url: runner.videoURL) {
                            if let url = pickFile(types: [.movie, .mpeg4Movie, .quickTimeMovie]) {
                                runner.videoURL = url
                            }
                        }
                    }

                    pickerRow(label: "DA3 weights", url: runner.da3WeightsURL) {
                        if let url = pickWeightsFile() { runner.da3WeightsURL = url }
                    }

                    if runner.mode == .singleImage {
                        GridRow {
                            Text("Model config").frame(width: 130, alignment: .leading)
                            Picker("", selection: $runner.singleImageConfig) {
                                Text("da3-large").tag("da3-large")
                                Text("da3-giant").tag("da3-giant")
                                Text("da3mono-large").tag("da3mono-large")
                                Text("da3-base").tag("da3-base")
                                Text("da3-small").tag("da3-small")
                            }
                            .labelsHidden()
                        }

                        GridRow {
                            Text("Process resolution").frame(width: 130, alignment: .leading)
                            Stepper(value: $runner.processRes, in: 280...1036, step: 14) {
                                Text("\(runner.processRes)")
                            }
                        }
                    } else {
                        pickerRow(label: "SALAD weights", url: runner.saladWeightsURL) {
                            if let url = pickWeightsFile() { runner.saladWeightsURL = url }
                        }

                        GridRow {
                            Text("FPS").frame(width: 130, alignment: .leading)
                            Stepper(value: $runner.fps, in: 1...24, step: 1) {
                                Text("\(Int(runner.fps))")
                            }
                        }

                        GridRow {
                            Text("Max frames").frame(width: 130, alignment: .leading)
                            Stepper(value: $runner.maxFrames, in: 8...256, step: 8) {
                                Text("\(runner.maxFrames)")
                            }
                        }

                        GridRow {
                            Text("Chunk size").frame(width: 130, alignment: .leading)
                            Stepper(value: $runner.chunkSize, in: 4...32, step: 2) {
                                Text("\(runner.chunkSize)")
                            }
                        }

                        GridRow {
                            Text("Overlap").frame(width: 130, alignment: .leading)
                            Stepper(value: $runner.overlap, in: 1...16, step: 1) {
                                Text("\(runner.overlap)")
                            }
                        }
                    }
                }

                if runner.mode == .streaming {
                    Toggle("Enable loop closure (requires SALAD weights)", isOn: $runner.enableLoopClosure)
                        .padding(.top, 4)
                }
            }

            HStack {
                Button {
                    Task { await runner.run() }
                } label: {
                    Label("Run", systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(isRunning || !canRun)

                if runner.mode == .streaming {
                    Button {
                        if let dir = pickDirectory() {
                            Task { await runner.savePLY(to: dir) }
                        }
                    } label: {
                        Label("Save PLY + poses…", systemImage: "tray.and.arrow.down")
                    }
                    .disabled(runner.finalPredictionPoses == nil || isRunning)
                }
            }

            GroupBox("Status") {
                statusView
            }

            if let path = runner.savedPLYPath, runner.mode == .streaming {
                Text("Saved PLY: \(path)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            if let err = runner.lastError {
                Text("Error: \(err)")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }

            // Output viewer
            if case .done = runner.state {
                if runner.mode == .singleImage, let depth = runner.singleDepth {
                    GroupBox("Depth heatmap") {
                        HStack(alignment: .top, spacing: 12) {
                            if let src = runner.singleSourceImage {
                                VStack(alignment: .leading) {
                                    Text("Input")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    Image(nsImage: src)
                                        .resizable()
                                        .aspectRatio(contentMode: .fit)
                                }
                            }
                            VStack(alignment: .leading) {
                                Text("Depth (\(runner.singleProcessedHeight)×\(runner.singleProcessedWidth))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                DepthMapView(depth: depth)
                            }
                        }
                        .frame(minHeight: 240)
                    }
                }
                if runner.mode == .streaming, let poses = runner.finalPredictionPoses {
                    GroupBox("Pose viewer (drag to orbit, scroll to zoom)") {
                        PoseSceneView(
                            cameraPosesC2W: poses,
                            intrinsicsK: runner.finalPredictionIntrinsics
                        )
                        .frame(minHeight: 320)
                    }
                }
            }

            Spacer()
        }
        .padding()
        .frame(minWidth: 620, minHeight: 600)
    }

    private var canRun: Bool {
        guard runner.da3WeightsURL != nil else { return false }
        switch runner.mode {
        case .singleImage: return runner.imageURL != nil
        case .streaming:   return runner.videoURL != nil
        }
    }

    private var isRunning: Bool {
        switch runner.state {
        case .idle, .done, .failed: return false
        default: return true
        }
    }

    private func pickerRow(label: String, url: URL?, onPick: @escaping () -> Void) -> some View {
        GridRow {
            Text(label).frame(width: 130, alignment: .leading)
            HStack {
                Text(url?.lastPathComponent ?? "—")
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .foregroundStyle(url == nil ? .secondary : .primary)
                Spacer()
                Button("Choose…", action: onPick)
            }
        }
    }

    @ViewBuilder
    private var statusView: some View {
        switch runner.state {
        case .idle:
            Text("Idle. Pick inputs and run.")
                .foregroundStyle(.secondary)
        case .extractingFrames(let n):
            HStack { ProgressView(); Text("Extracting frames (\(n) so far)…") }
        case .loadingWeights:
            HStack { ProgressView(); Text("Loading DA3 weights (~1s)…") }
        case .runningSingleImage:
            HStack { ProgressView(); Text("Running single-image inference…") }
        case .runningStreaming:
            HStack { ProgressView(); Text("Running streaming inference (\(runner.frameCount) frames)…") }
        case .detectingLoops:
            HStack { ProgressView(); Text("Running SALAD loop detection…") }
        case .measuringLoops(let i, let n):
            HStack { ProgressView(); Text("Computing loop measurement \(i)/\(n)…") }
        case .refining:
            HStack { ProgressView(); Text("Refining poses with loop constraints…") }
        case .writingPLY:
            HStack { ProgressView(); Text("Writing PLY + camera poses…") }
        case .done:
            VStack(alignment: .leading, spacing: 4) {
                Label("Done", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                if runner.mode == .streaming {
                    Text("frames=\(runner.frameCount), poses=\(runner.poseCount), loops=\(runner.detectedLoopCount)")
                        .font(.caption.monospaced())
                } else {
                    Text("processed=\(runner.singleProcessedHeight)×\(runner.singleProcessedWidth)")
                        .font(.caption.monospaced())
                }
            }
        case .failed(let msg):
            Label(msg, systemImage: "xmark.octagon.fill").foregroundStyle(.red)
        }
    }

    // MARK: - NSOpenPanel helpers

    private func pickFile(types: [UTType]) -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = types
        return panel.runModal() == .OK ? panel.url : nil
    }

    /// SwiftUI's UTType for `.safetensors` is unreliable across Xcode versions;
    /// fall back to "all files" and rely on filename extension.
    private func pickWeightsFile() -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.title = "Choose .safetensors weights file"
        if let st = UTType(filenameExtension: "safetensors") {
            panel.allowedContentTypes = [st, .data]
        }
        return panel.runModal() == .OK ? panel.url : nil
    }

    private func pickDirectory() -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.title = "Choose output directory"
        return panel.runModal() == .OK ? panel.url : nil
    }
}

#Preview {
    ContentView()
}
