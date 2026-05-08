import Foundation

// Sim(3) pose-graph Levenberg-Marquardt optimizer for loop closure.
//
// Mirrors `da3_streaming/loop_utils/sim3loop.py` Sim3LoopOptimizer with two
// implementation differences:
//
// 1. **Numerical Jacobian** instead of pypose's autograd-based analytic
//    Jacobian. Central differences on `Log(C @ Exp(Gi) @ Exp(Gj)^{-1})` with
//    a fixed perturbation of `1e-6`. Slower per iteration but order(N) more
//    code than analytic, and equivalent for our problem size (≤ tens of
//    poses, ≤ tens of loops).
//
// 2. **Dense linear solve** instead of scipy's sparse CSC. We're well below
//    the dimensions where sparsity pays off (`7n × 7n` ≤ 140 × 140 for
//    n=20).
//
// Algorithm matches python's `optimize()`:
//   - Convert sequential transforms to absolute poses
//   - Build loop edges
//   - Initialize Ginv = Log(absolute_poses^{-1})
//   - LM loop: residual = Log(C @ Exp(Gi) @ Exp(Gj)^{-1})
//     - Compute J via finite differences
//     - Solve (JᵀJ + λ·diag) δ = -Jᵀ r
//     - Accept/reject by cost; halve/double λ
//   - Return optimized sequential transforms.

/// A single loop closure edge: from frame index `i` to frame `j`, with the
/// observed Sim(3) measurement `(s, R, t)` (relative pose i → j).
public struct LoopConstraint {
    public let i: Int
    public let j: Int
    public let measurement: Sim3

    public init(i: Int, j: Int, measurement: Sim3) {
        self.i = i
        self.j = j
        self.measurement = measurement
    }
}

public struct Sim3LoopOptimizer {
    public struct Config {
        public var maxIterations: Int = 30
        public var lambdaInit: Double = 1e-6
        public var jacobianEpsilon: Double = 1e-6
        /// LM convergence: stop if 5-window cost ratio < this.
        public var convergenceRatio: Double = 1.5
        /// Don't bother converging if cost is already tiny.
        public var convergenceCostThreshold: Double = 1e-5
        public var verbose: Bool = false

        public init() {}
    }

    public let config: Config
    public init(config: Config = Config()) { self.config = config }

    /// Run pose-graph LM. Returns the optimized sequential transforms with
    /// the same length as the input.
    public func optimize(
        sequentialTransforms: [Sim3],
        loopConstraints: [LoopConstraint]
    ) -> [Sim3] {
        if loopConstraints.isEmpty {
            if config.verbose { print("No loop constraints; returning input as-is.") }
            return sequentialTransforms
        }

        // Build absolute poses: T_0 = identity, T_{k+1} = T_k @ relative_k.
        var absolutes: [Sim3] = [.identity]
        for rel in sequentialTransforms { absolutes.append(absolutes.last!.compose(rel)) }

        // Initialize tangent state: Ginv[k] = Log(T_k^{-1}).
        var Ginv: [Sim3Tangent] = absolutes.map { sim3Log($0.inverse) }
        let n = Ginv.count

        // Sequential constraints (k → k-1): dSij[k] = T_{k-1}^{-1} @ T_k = relative_{k-1}.
        // Loop constraints from caller.
        // Pre-compute "constants" = list of (C, i, j) edges feeding into the residual.
        var edges: [(C: Sim3, i: Int, j: Int)] = []
        // python's residual builds dSij from inverted poses Ti^{-1} @ Tj — but Tj^{-1} composed
        // with (Ti^{-1})^{-1} so let's recompute against the input's `relative` directly.
        // python:
        //   pred_inv_poses = pp.Sim3(input_poses).Inv()  → Ti^{-1}
        //   Ti = pred_inv_poses[kk]  (kk = 1..n-1)
        //   Tj = pred_inv_poses[ll]  (ll = 0..n-2)
        //   dSij = Tj @ Ti.Inv() = T_{k-1}^{-1} @ T_k = sequential[k-1]
        // So sequential transforms ARE dSij. Use them directly.
        for k in 0..<sequentialTransforms.count {
            edges.append((C: sequentialTransforms[k], i: k + 1, j: k))
        }
        for lc in loopConstraints {
            edges.append((C: lc.measurement, i: lc.i, j: lc.j))
        }

        var lambda = config.lambdaInit
        var costHistory: [Double] = []

        if config.verbose {
            print("Sim3 LM: n=\(n) poses, edges=\(edges.count) (sequential=\(sequentialTransforms.count), loop=\(loopConstraints.count))")
        }

        for itr in 0..<config.maxIterations {
            let resid = computeResiduals(Ginv: Ginv, edges: edges)
            let currentCost = resid.reduce(0.0) { $0 + $1 * $1 } / Double(max(resid.count, 1))
            costHistory.append(currentCost)

            // Jacobian via finite differences. Shape: [r, n*7].
            let r = resid.count
            let rowSize = n * 7
            let J = computeJacobian(Ginv: Ginv, edges: edges, residRef: resid)

            // Build JᵀJ (rowSize × rowSize) and Jᵀr (rowSize).
            var JtJ = [Double](repeating: 0, count: rowSize * rowSize)
            var JtR = [Double](repeating: 0, count: rowSize)
            for col in 0..<rowSize {
                for row in 0..<r {
                    let v = J[row * rowSize + col]
                    JtR[col] += v * resid[row]
                    for col2 in col..<rowSize {
                        JtJ[col * rowSize + col2] += v * J[row * rowSize + col2]
                    }
                }
            }
            // Symmetrize.
            for c1 in 0..<rowSize {
                for c2 in (c1 + 1)..<rowSize {
                    JtJ[c2 * rowSize + c1] = JtJ[c1 * rowSize + c2]
                }
            }

            // LM damping: A = JᵀJ * (1 + λ) on diagonal.
            var A = JtJ
            for d in 0..<rowSize {
                A[d * rowSize + d] *= (1.0 + lambda)
            }
            // RHS: -Jᵀr
            let b = JtR.map { -$0 }

            // Gauge fixing: pin pose 0 (frame 0) to identity by clamping its
            // 7 columns/rows. Mirrors python's freen=-1 path which solves
            // the full system — the system is rank-deficient by 7 if no
            // gauge is fixed. We zero out the first 7 dims of delta and
            // remove their rows from the linear solve.
            // Simpler approach: solve (A + ε I) δ = b with strong damping
            // on the first 7 rows only (effectively a soft prior).
            let gaugePrior = 1e6
            for d in 0..<7 {
                A[d * rowSize + d] += gaugePrior
            }

            // Solve A δ = b.
            guard let delta = solveDense(A: A, b: b, n: rowSize) else {
                if config.verbose { print("LM iter \(itr): linear solve failed; breaking") }
                break
            }

            // Tentative update: Ginv_tmp = Ginv + delta (per pose, 7-vec).
            var GinvTmp = Ginv
            for k in 0..<n {
                for d in 0..<7 {
                    GinvTmp[k][d] += delta[k * 7 + d]
                }
            }

            let newResid = computeResiduals(Ginv: GinvTmp, edges: edges)
            let newCost = newResid.reduce(0.0) { $0 + $1 * $1 } / Double(max(newResid.count, 1))

            if newCost < currentCost {
                Ginv = GinvTmp
                lambda /= 2
                if config.verbose {
                    print(String(format: "  iter %d: cost %.10f -> %.10f (accept) λ=%.4g",
                                 itr, currentCost, newCost, lambda))
                }
            } else {
                lambda *= 2
                if config.verbose {
                    print(String(format: "  iter %d: cost %.10f -> %.10f (reject) λ=%.4g",
                                 itr, currentCost, newCost, lambda))
                }
            }

            // Convergence check
            if currentCost < config.convergenceCostThreshold && itr >= 4 && costHistory.count >= 5 {
                let ratio = costHistory[costHistory.count - 5] / max(costHistory.last!, 1e-20)
                if ratio < config.convergenceRatio {
                    if config.verbose { print("  converged at iter \(itr)") }
                    break
                }
            }
        }

        // Reconstruct optimized absolute poses: T_k = Exp(Ginv[k])^{-1}
        let optAbsolutes: [Sim3] = Ginv.map { sim3Exp($0).inverse }

        // Convert back to sequential: rel_k = T_k^{-1} @ T_{k+1}
        var optSequential: [Sim3] = []
        for k in 0..<(optAbsolutes.count - 1) {
            optSequential.append(optAbsolutes[k].inverse.compose(optAbsolutes[k + 1]))
        }
        return optSequential
    }

    // MARK: - Internal helpers

    /// residual(C, Gi, Gj) = Log(C ∘ Exp(Gi) ∘ Exp(Gj)^{-1}).
    /// Returns flat array of length 7 * edges.count.
    private func computeResiduals(Ginv: [Sim3Tangent], edges: [(C: Sim3, i: Int, j: Int)]) -> [Double] {
        var out = [Double](repeating: 0, count: 7 * edges.count)
        for (idx, edge) in edges.enumerated() {
            let gi = sim3Exp(Ginv[edge.i])
            let gj = sim3Exp(Ginv[edge.j])
            let prod = edge.C.compose(gi).compose(gj.inverse)
            let r = sim3Log(prod)
            for d in 0..<7 {
                out[idx * 7 + d] = r[d]
            }
        }
        return out
    }

    /// Numerical Jacobian via central differences. Result: row-major
    /// `[r, rowSize]` flat array where `r = 7 * edges.count` and
    /// `rowSize = 7 * n_poses`.
    private func computeJacobian(
        Ginv: [Sim3Tangent], edges: [(C: Sim3, i: Int, j: Int)], residRef: [Double]
    ) -> [Double] {
        let n = Ginv.count
        let rowSize = n * 7
        let r = 7 * edges.count
        var J = [Double](repeating: 0, count: r * rowSize)
        let eps = config.jacobianEpsilon

        // For each pose k, perturb each of its 7 dimensions and recompute
        // residuals; store the column.
        var GinvP = Ginv
        var GinvM = Ginv
        for k in 0..<n {
            for d in 0..<7 {
                GinvP[k][d] += eps
                GinvM[k][d] -= eps
                let rPlus = computeResiduals(Ginv: GinvP, edges: edges)
                let rMinus = computeResiduals(Ginv: GinvM, edges: edges)
                let col = k * 7 + d
                for row in 0..<r {
                    J[row * rowSize + col] = (rPlus[row] - rMinus[row]) / (2.0 * eps)
                }
                GinvP[k][d] -= eps
                GinvM[k][d] += eps
            }
        }
        return J
    }

    /// Dense Cholesky-ish solve of `A x = b` for a symmetric matrix.
    /// Falls back to LU if Cholesky fails (e.g. not PD due to gauge).
    private func solveDense(A: [Double], b: [Double], n: Int) -> [Double]? {
        // We use Gaussian elimination with partial pivoting (LU). Since
        // this is dense and `n` is small (≤ 140), a hand-rolled solver is
        // fine and avoids pulling in MLXLinalg for a 7n × 7n matmul.
        var M = A
        var x = b
        // In-place row reduction.
        for k in 0..<n {
            // Pivot: find max |M[r,k]| for r ≥ k.
            var pivot = k
            var maxAbs = abs(M[k * n + k])
            for r in (k + 1)..<n {
                let v = abs(M[r * n + k])
                if v > maxAbs { maxAbs = v; pivot = r }
            }
            if maxAbs < 1e-14 { return nil }
            if pivot != k {
                for c in 0..<n {
                    let tmp = M[k * n + c]
                    M[k * n + c] = M[pivot * n + c]
                    M[pivot * n + c] = tmp
                }
                let tmpB = x[k]; x[k] = x[pivot]; x[pivot] = tmpB
            }
            // Eliminate
            let inv = 1.0 / M[k * n + k]
            for r in (k + 1)..<n {
                let factor = M[r * n + k] * inv
                if factor == 0 { continue }
                for c in k..<n {
                    M[r * n + c] -= factor * M[k * n + c]
                }
                x[r] -= factor * x[k]
            }
        }
        // Back-substitution
        var sol = [Double](repeating: 0, count: n)
        for k in stride(from: n - 1, through: 0, by: -1) {
            var s = x[k]
            for c in (k + 1)..<n {
                s -= M[k * n + c] * sol[c]
            }
            sol[k] = s / M[k * n + k]
        }
        return sol
    }
}
