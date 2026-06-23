//! Low-level vector mathematics.
//!
//! All public functions operate on `&[f32]` slices — they are independent of
//! any storage layer and can be tested in complete isolation.
//!
//! # Design notes
//! * `f32` gives adequate precision for embedding similarity while halving
//!   memory vs `f64` for large corpora.
//! * No SIMD / BLAS — clarity over micro-optimisation.
//! * Every function is `#[inline]` so the compiler can eliminate call overhead
//!   when used inside the hot search loop.

// ─────────────────────────────────────────────────────────────────────────────
// Core operations
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the **dot product** (inner product) of two equal-length vectors.
///
/// `dot(a, b) = a₀·b₀ + a₁·b₁ + … + aₙ·bₙ`
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot_product: dimension mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the **Euclidean magnitude** (L2 norm) of a vector.
///
/// `‖v‖ = √(v₀² + v₁² + … + vₙ²)`
#[inline]
pub fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute the **cosine similarity** between two vectors.
///
/// ```text
/// cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
/// ```
///
/// Returns a value in `[−1.0, 1.0]`:
/// - `1.0`  → same direction (maximum similarity)
/// - `0.0`  → orthogonal
/// - `−1.0` → opposite direction
///
/// Returns `0.0` when either vector is the **zero vector** (undefined
/// mathematically, treated as "no similarity").
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mag_a = magnitude(a);
    let mag_b = magnitude(b);

    // Guard: a zero vector has no direction → similarity is undefined.
    // We return 0.0 (no similarity) rather than NaN to keep scores comparable.
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    // Clamp to [−1, 1] to neutralise floating-point drift (e.g. 1.0000001).
    let raw = dot_product(a, b) / (mag_a * mag_b);
    raw.clamp(-1.0, 1.0)
}

/// Return a **unit-length copy** of `v` (L2-normalised).
///
/// A normalised vector has `‖v‖ = 1.0`.
/// If `v` is the zero vector it is returned unchanged (cannot be normalised).
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let mag = magnitude(v);
    if mag == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / mag).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Floating-point tolerance for equality checks.
    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ── dot_product ───────────────────────────────────────────────────────────

    #[test]
    fn dot_product_simple() {
        // [1,2,3] · [4,5,6] = 4 + 10 + 18 = 32
        assert!(approx_eq(dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0));
    }

    #[test]
    fn dot_product_orthogonal_is_zero() {
        // Canonical basis vectors are orthogonal
        assert!(approx_eq(dot_product(&[1.0, 0.0], &[0.0, 1.0]), 0.0));
    }

    #[test]
    fn dot_product_with_zero_vector() {
        assert!(approx_eq(dot_product(&[3.0, 4.0], &[0.0, 0.0]), 0.0));
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn dot_product_mismatched_dims_panics() {
        dot_product(&[1.0, 2.0], &[1.0]);
    }

    // ── magnitude ─────────────────────────────────────────────────────────────

    #[test]
    fn magnitude_3_4_is_5() {
        // Pythagorean triple: ‖[3, 4]‖ = 5
        assert!(approx_eq(magnitude(&[3.0, 4.0]), 5.0));
    }

    #[test]
    fn magnitude_unit_vector() {
        assert!(approx_eq(magnitude(&[1.0, 0.0, 0.0]), 1.0));
    }

    #[test]
    fn magnitude_zero_vector() {
        assert!(approx_eq(magnitude(&[0.0, 0.0, 0.0]), 0.0));
    }

    // ── cosine_similarity ─────────────────────────────────────────────────────

    #[test]
    fn cosine_identical_vectors_is_one() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!(approx_eq(sim, 1.0), "got {sim}");
    }

    #[test]
    fn cosine_opposite_vectors_is_minus_one() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(approx_eq(sim, -1.0), "got {sim}");
    }

    #[test]
    fn cosine_orthogonal_vectors_is_zero() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(approx_eq(sim, 0.0), "got {sim}");
    }

    #[test]
    fn cosine_scaled_vector_same_direction() {
        // Scaling a vector should not change its direction → similarity = 1
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![2.0_f32, 4.0, 6.0]; // a * 2
        assert!(approx_eq(cosine_similarity(&a, &b), 1.0));
    }

    #[test]
    fn cosine_zero_vector_returns_zero() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 0.0));
    }

    #[test]
    fn cosine_result_clamped_to_minus_one_plus_one() {
        // Even with floating-point drift the result must stay in [-1, 1]
        let v = vec![1.0_f32 / 3.0_f32.sqrt(); 3];
        let sim = cosine_similarity(&v, &v);
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    #[test]
    fn cosine_45_degree_angle() {
        // cos(45°) ≈ 0.7071
        let a = vec![1.0_f32, 0.0];
        let b = vec![1.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        let expected = std::f32::consts::FRAC_1_SQRT_2; // ≈ 0.70711
        assert!(approx_eq(sim, expected), "got {sim}, expected {expected}");
    }

    // ── normalize ─────────────────────────────────────────────────────────────

    #[test]
    fn normalize_produces_unit_vector() {
        let v = normalize(&[3.0, 4.0]);
        assert!(approx_eq(magnitude(&v), 1.0));
    }

    #[test]
    fn normalize_unit_vector_unchanged() {
        let v = normalize(&[1.0, 0.0, 0.0]);
        assert!(approx_eq(v[0], 1.0));
        assert!(approx_eq(v[1], 0.0));
    }

    #[test]
    fn normalize_zero_vector_unchanged() {
        let z = vec![0.0_f32, 0.0, 0.0];
        let n = normalize(&z);
        assert_eq!(n, z);
    }

    #[test]
    fn normalize_preserves_direction() {
        // After normalising, cosine similarity with the original must be 1
        let v = vec![3.0_f32, 1.0, 4.0, 1.0, 5.0];
        let n = normalize(&v);
        assert!(approx_eq(cosine_similarity(&v, &n), 1.0));
    }
}
