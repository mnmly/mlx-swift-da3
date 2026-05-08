import Foundation
import XCTest

@testable import MLXDA3Streaming

final class Sim3LieGroupTests: XCTestCase {
    func testIdentityCompose() {
        let id = Sim3.identity
        let g = Sim3(R: [0.5, -0.5 * 1.732, 0, 0.5 * 1.732, 0.5, 0, 0, 0, 1], t: [1, 2, 3], s: 1.5)
        // Compose with identity is no-op (within fp noise)
        let lhs = id.compose(g)
        let rhs = g.compose(id)
        for i in 0..<9 {
            XCTAssertEqual(lhs.R[i], g.R[i], accuracy: 1e-10)
            XCTAssertEqual(rhs.R[i], g.R[i], accuracy: 1e-10)
        }
        for i in 0..<3 {
            XCTAssertEqual(lhs.t[i], g.t[i], accuracy: 1e-10)
            XCTAssertEqual(rhs.t[i], g.t[i], accuracy: 1e-10)
        }
        XCTAssertEqual(lhs.s, g.s, accuracy: 1e-10)
    }

    func testComposeInverseIsIdentity() {
        // 30° rotation about z, scale 1.7, translation (1, -2, 3)
        let theta = Double.pi / 6
        let c = cos(theta), s = sin(theta)
        let g = Sim3(R: [c, -s, 0, s, c, 0, 0, 0, 1], t: [1, -2, 3], s: 1.7)
        let prod = g.compose(g.inverse)
        let id = Sim3.identity
        for i in 0..<9 {
            XCTAssertEqual(prod.R[i], id.R[i], accuracy: 1e-10)
        }
        for i in 0..<3 {
            XCTAssertEqual(prod.t[i], id.t[i], accuracy: 1e-10)
        }
        XCTAssertEqual(prod.s, 1.0, accuracy: 1e-10)
    }

    func testExpLogRoundTrip() {
        // Random-ish tangent with all components active
        let cases: [Sim3Tangent] = [
            [0.1, 0.2, -0.3, 0.5, -0.1, 0.2, 0.3],
            [0, 0, 0, 0, 0, 0, 0],                        // identity
            [1, -2, 3, 0, 0, 0, 0],                       // pure translation
            [0, 0, 0, 0.5, 0.7, -0.2, 0],                 // pure rotation
            [0, 0, 0, 0, 0, 0, 0.5],                      // pure scale
            [0.1, 0.1, 0.1, 1e-9, 1e-9, 1e-9, 1e-9],      // near zero (Taylor branch)
        ]
        for xi in cases {
            let g = sim3Exp(xi)
            let xi2 = sim3Log(g)
            for k in 0..<7 {
                XCTAssertEqual(
                    xi[k], xi2[k], accuracy: 1e-9,
                    "Exp/Log round-trip failed at index \(k); xi=\(xi) xi2=\(xi2)"
                )
            }
        }
    }

    func testAssociativity() {
        let g1 = sim3Exp([0.1, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        let g2 = sim3Exp([0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0])
        let g3 = sim3Exp([0.0, 0.0, 0.3, 0.0, 0.0, 0.4, 0.1])
        let lhs = g1.compose(g2).compose(g3)
        let rhs = g1.compose(g2.compose(g3))
        for i in 0..<9 { XCTAssertEqual(lhs.R[i], rhs.R[i], accuracy: 1e-10) }
        for i in 0..<3 { XCTAssertEqual(lhs.t[i], rhs.t[i], accuracy: 1e-10) }
        XCTAssertEqual(lhs.s, rhs.s, accuracy: 1e-10)
    }
}
