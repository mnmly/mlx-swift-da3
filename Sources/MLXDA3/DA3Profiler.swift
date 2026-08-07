import Foundation
import MLX

/// Opt-in phase timing for the model and the streaming pipeline.
///
/// Disabled by default and costs nothing when off. When enabled, each phase is
/// forced to materialize (`eval`) at its boundary so lazy MLX work is charged to
/// the phase that created it rather than to whoever happens to sync first.
public enum DA3Profiler {
    public nonisolated(unsafe) static var isEnabled = false

    private static let lock = NSLock()
    private nonisolated(unsafe) static var order: [String] = []
    private nonisolated(unsafe) static var totals: [String: (seconds: Double, calls: Int)] = [:]

    public static func reset() {
        lock.lock(); defer { lock.unlock() }
        order.removeAll()
        totals.removeAll()
    }

    /// Time `body`. `sync` is called with the result to force evaluation before the
    /// clock stops (no-op when profiling is off).
    public static func measure<T>(_ label: String, sync: (T) -> Void, _ body: () -> T) -> T {
        guard isEnabled else { return body() }
        let start = CFAbsoluteTimeGetCurrent()
        let value = body()
        sync(value)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        lock.lock()
        if totals[label] == nil { order.append(label) }
        let previous = totals[label] ?? (0, 0)
        totals[label] = (previous.seconds + elapsed, previous.calls + 1)
        lock.unlock()
        return value
    }

    public static func measure<T>(_ label: String, _ body: () -> T) -> T {
        measure(label, sync: { _ in }, body)
    }

    public static func report() -> String {
        lock.lock(); defer { lock.unlock() }
        let total = totals.values.reduce(0) { $0 + $1.seconds }
        func pad(_ s: String) -> String {
            s.count >= 24 ? s : s + String(repeating: " ", count: 24 - s.count)
        }
        var lines = [pad("phase") + "  seconds    calls    share"]
        for label in order {
            guard let entry = totals[label] else { continue }
            lines.append(pad(label) + String(
                format: " %8.3f %8d %7.1f%%",
                entry.seconds, entry.calls,
                total > 0 ? entry.seconds / total * 100 : 0
            ))
        }
        lines.append(pad("total") + String(format: " %8.3f", total))
        return lines.joined(separator: "\n")
    }
}
