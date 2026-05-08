import Foundation

// Sim(3) Lie group operations used by the loop closure pose-graph optimizer.
//
// Reference: Strasdat, "Scale Drift-Aware Large Scale Monocular SLAM" (2010)
// + Sophus C++ Sim3 closed-form implementation.
//
// All math is in `Double` to avoid catastrophic cancellation in the
// `(τ ↔ small angle, σ → 0)` series — the LM solver can drive ω, σ very
// close to zero on convergence, where naive `Float` arithmetic loses too
// many digits.

/// Element of Sim(3) = scaled rotation + translation.
public struct Sim3: Equatable {
    /// 3×3 rotation matrix, row-major.
    public var R: [Double]
    /// 3-vector translation.
    public var t: [Double]
    /// Positive scalar scale.
    public var s: Double

    public init(R: [Double], t: [Double], s: Double) {
        precondition(R.count == 9, "R must be 3×3 (9 elements)")
        precondition(t.count == 3, "t must be 3-vector")
        precondition(s > 0, "scale must be positive")
        self.R = R
        self.t = t
        self.s = s
    }

    public static var identity: Sim3 {
        Sim3(R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0], s: 1)
    }

    /// Group product: this ∘ other.
    /// On 4×4 form: M = [[s_a R_a | t_a]; [0 1]] @ [[s_b R_b | t_b]; [0 1]]
    /// = [[s_a R_a · s_b R_b | s_a R_a t_b + t_a]; [0 1]]
    public func compose(_ other: Sim3) -> Sim3 {
        let aR = R, bR = other.R
        var newR = [Double](repeating: 0, count: 9)
        for i in 0..<3 {
            for j in 0..<3 {
                var sum = 0.0
                for k in 0..<3 {
                    sum += aR[i * 3 + k] * bR[k * 3 + j]
                }
                newR[i * 3 + j] = sum
            }
        }
        // Scale of product: s_a * s_b
        let newS = s * other.s
        // Translation: s_a * R_a · t_b + t_a
        var newT = [Double](repeating: 0, count: 3)
        for i in 0..<3 {
            var sum = 0.0
            for k in 0..<3 {
                sum += aR[i * 3 + k] * other.t[k]
            }
            newT[i] = s * sum + t[i]
        }
        return Sim3(R: newR, t: newT, s: newS)
    }

    /// Group inverse.
    /// M^{-1} = [[R^T / s | -R^T t / s]; [0 1]]
    public var inverse: Sim3 {
        var Rinv = [Double](repeating: 0, count: 9)
        for i in 0..<3 {
            for j in 0..<3 {
                Rinv[i * 3 + j] = R[j * 3 + i]
            }
        }
        var tinv = [Double](repeating: 0, count: 3)
        for i in 0..<3 {
            var sum = 0.0
            for k in 0..<3 {
                sum += Rinv[i * 3 + k] * t[k]
            }
            tinv[i] = -sum / s
        }
        return Sim3(R: Rinv, t: tinv, s: 1.0 / s)
    }
}

/// Tangent space element ξ ∈ sim(3) = R⁷.
/// Layout: `[ν₀ ν₁ ν₂ ω₀ ω₁ ω₂ σ]` where
/// - ν: linear velocity (translation tangent)
/// - ω: angular velocity (rotation tangent, axis-angle)
/// - σ: log scale
public typealias Sim3Tangent = [Double]

// MARK: - SO(3) helpers

/// SO(3) hat: ω → ω̂ (skew-symmetric 3×3, row-major).
private func so3Hat(_ omega: [Double]) -> [Double] {
    let x = omega[0], y = omega[1], z = omega[2]
    return [
        0, -z, y,
        z, 0, -x,
        -y, x, 0
    ]
}

private func mat3MulMat3(_ a: [Double], _ b: [Double]) -> [Double] {
    var out = [Double](repeating: 0, count: 9)
    for i in 0..<3 {
        for j in 0..<3 {
            var s = 0.0
            for k in 0..<3 { s += a[i * 3 + k] * b[k * 3 + j] }
            out[i * 3 + j] = s
        }
    }
    return out
}

private func mat3MulVec3(_ a: [Double], _ v: [Double]) -> [Double] {
    [
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    ]
}

private func mat3Add(_ a: [Double], _ b: [Double]) -> [Double] {
    zip(a, b).map { $0 + $1 }
}

private func mat3Scale(_ a: [Double], _ s: Double) -> [Double] {
    a.map { $0 * s }
}

private let identity3: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]

// MARK: - Sim(3) Exp / Log

// `epsilon` is the boundary where the closed-form Sim3 formulas suffer
// catastrophic cancellation in the (sigma² + theta²) denominator. Values
// of theta or sigma below ≈1e-6 risk losing 6+ digits in fp64. The Taylor
// branches we hand off to are accurate to many more digits in that region.
private let epsilon: Double = 1e-6

/// Sim(3) exponential map. Mirrors Sophus's closed-form Sim3::exp.
public func sim3Exp(_ xi: Sim3Tangent) -> Sim3 {
    precondition(xi.count == 7)
    let nu = Array(xi[0..<3])
    let omega = Array(xi[3..<6])
    let sigma = xi[6]

    let theta2 = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]
    let theta = sqrt(theta2)
    let omegaHat = so3Hat(omega)
    let omegaHat2 = mat3MulMat3(omegaHat, omegaHat)
    let expSigma = exp(sigma)

    // Compute scalar coefficients A, B, C s.t. W = A·ω̂ + B·ω̂² + C·I
    let A: Double
    let B: Double
    let C: Double

    if theta < epsilon && abs(sigma) < epsilon {
        // Both small: use leading Taylor terms
        A = 0.5 + sigma * (1.0 / 6.0)
        B = (1.0 / 6.0)
        C = 1.0 + 0.5 * sigma
    } else if theta < epsilon {
        // ω ≈ 0, σ general
        C = (expSigma - 1.0) / sigma
        A = (sigma * expSigma - expSigma + 1.0) / (sigma * sigma)
        B = (expSigma * (sigma * sigma - 2.0 * sigma + 2.0) - 2.0) / (2.0 * sigma * sigma * sigma)
    } else if abs(sigma) < epsilon {
        // σ ≈ 0, ω general (this collapses to SO(3) left-Jacobian J_L)
        let sinT = sin(theta)
        let cosT = cos(theta)
        C = 1.0
        A = (1.0 - cosT) / theta2
        B = (theta - sinT) / (theta2 * theta)
    } else {
        // Full case: both σ and θ non-trivial.
        let sigma2 = sigma * sigma
        let sinT = sin(theta)
        let cosT = cos(theta)
        let a = expSigma * sinT
        let b = expSigma * cosT
        C = (expSigma - 1.0) / sigma
        A = (a * sigma + (1.0 - b) * theta) / (theta * (sigma2 + theta2))
        B = (C - ((b - 1.0) * sigma + a * theta) / (sigma2 + theta2)) / theta2
    }

    // W = A·ω̂ + B·ω̂² + C·I
    var W = mat3Scale(omegaHat, A)
    W = mat3Add(W, mat3Scale(omegaHat2, B))
    W = mat3Add(W, mat3Scale(identity3, C))

    // Translation = W · ν
    let t = mat3MulVec3(W, nu)

    // Rotation R = exp(ω̂) (Rodrigues)
    let R: [Double]
    if theta < epsilon {
        // R ≈ I + ω̂ + ω̂²/2
        R = mat3Add(identity3, mat3Add(omegaHat, mat3Scale(omegaHat2, 0.5)))
    } else {
        let sinT = sin(theta)
        let cosT = cos(theta)
        let aR = sinT / theta
        let bR = (1.0 - cosT) / theta2
        R = mat3Add(identity3, mat3Add(mat3Scale(omegaHat, aR), mat3Scale(omegaHat2, bR)))
    }

    return Sim3(R: R, t: t, s: expSigma)
}

/// Sim(3) logarithm: M → ξ ∈ R⁷.
public func sim3Log(_ M: Sim3) -> Sim3Tangent {
    let s = M.s
    let sigma = log(s)
    // ω from R via Rodrigues inverse
    let trace = M.R[0] + M.R[4] + M.R[8]
    let cosT = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    let theta = acos(cosT)
    let theta2 = theta * theta

    var omega = [Double](repeating: 0, count: 3)
    let omegaHat: [Double]
    if theta < epsilon {
        // Small angle: ω ≈ skew(R)/2 · (1 + θ²/12)
        omega[0] = (M.R[7] - M.R[5]) * 0.5
        omega[1] = (M.R[2] - M.R[6]) * 0.5
        omega[2] = (M.R[3] - M.R[1]) * 0.5
        omegaHat = so3Hat(omega)
    } else {
        let scale = theta / (2.0 * sin(theta))
        omega[0] = (M.R[7] - M.R[5]) * scale
        omega[1] = (M.R[2] - M.R[6]) * scale
        omega[2] = (M.R[3] - M.R[1]) * scale
        omegaHat = so3Hat(omega)
    }
    let omegaHat2 = mat3MulMat3(omegaHat, omegaHat)

    // Recompute W and invert to get ν.
    let A: Double, B: Double, C: Double
    if theta < epsilon && abs(sigma) < epsilon {
        A = 0.5 + sigma * (1.0 / 6.0)
        B = (1.0 / 6.0)
        C = 1.0 + 0.5 * sigma
    } else if theta < epsilon {
        let expSigma = exp(sigma)
        C = (expSigma - 1.0) / sigma
        A = (sigma * expSigma - expSigma + 1.0) / (sigma * sigma)
        B = (expSigma * (sigma * sigma - 2.0 * sigma + 2.0) - 2.0) / (2.0 * sigma * sigma * sigma)
    } else if abs(sigma) < epsilon {
        let sinT = sin(theta)
        let cosT = cos(theta)
        C = 1.0
        A = (1.0 - cosT) / theta2
        B = (theta - sinT) / (theta2 * theta)
    } else {
        let expSigma = exp(sigma)
        let sigma2 = sigma * sigma
        let sinT = sin(theta)
        let cosT = cos(theta)
        let a = expSigma * sinT
        let b = expSigma * cosT
        C = (expSigma - 1.0) / sigma
        A = (a * sigma + (1.0 - b) * theta) / (theta * (sigma2 + theta2))
        B = (C - ((b - 1.0) * sigma + a * theta) / (sigma2 + theta2)) / theta2
    }

    var W = mat3Scale(omegaHat, A)
    W = mat3Add(W, mat3Scale(omegaHat2, B))
    W = mat3Add(W, mat3Scale(identity3, C))

    // ν = W^{-1} · t. 3×3 inverse via cofactor / determinant.
    let nu = mat3SolveLinear(W, b: M.t)

    return [nu[0], nu[1], nu[2], omega[0], omega[1], omega[2], sigma]
}

/// Solve `A x = b` for a 3×3 matrix (cofactor expansion).
private func mat3SolveLinear(_ A: [Double], b: [Double]) -> [Double] {
    let m = A
    let det =
        m[0] * (m[4] * m[8] - m[5] * m[7])
      - m[1] * (m[3] * m[8] - m[5] * m[6])
      + m[2] * (m[3] * m[7] - m[4] * m[6])
    if abs(det) < 1e-15 {
        return [0, 0, 0]
    }
    let invDet = 1.0 / det
    let inv: [Double] = [
         (m[4] * m[8] - m[5] * m[7]) * invDet,
        -(m[1] * m[8] - m[2] * m[7]) * invDet,
         (m[1] * m[5] - m[2] * m[4]) * invDet,
        -(m[3] * m[8] - m[5] * m[6]) * invDet,
         (m[0] * m[8] - m[2] * m[6]) * invDet,
        -(m[0] * m[5] - m[2] * m[3]) * invDet,
         (m[3] * m[7] - m[4] * m[6]) * invDet,
        -(m[0] * m[7] - m[1] * m[6]) * invDet,
         (m[0] * m[4] - m[1] * m[3]) * invDet,
    ]
    return mat3MulVec3(inv, b)
}
