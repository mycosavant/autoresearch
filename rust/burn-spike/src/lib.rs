//! Track B2 spike: does Burn support the convolution a NAM WaveNet is made of?
//!
//! The research plan flagged one risk as the thing that decides the estimate:
//! **grouped *and* dilated causal `conv1d` with a working backward pass**. If it is
//! missing, a hand-written CubeCL kernel is required and B2 becomes a much larger
//! piece of work than a port.
//!
//! This crate answers that question by running it, not by reading release notes.
//! Every claim below is a test in this file.
//!
//! ## What the answer turns out to be
//!
//! Burn 0.21 handles all three, and the padding half of the question is better than
//! the plan assumed. `PaddingConfig1d::Explicit(left, right)` takes the two sides
//! separately, so a causal convolution is `Explicit((kernel - 1) * dilation, 0)` —
//! no manual pre-padding and no slicing the tail off afterwards.
//!
//! So no custom CubeCL kernel is needed for capability reasons, and the risk the
//! plan was carrying moves from "can it express this" to "how fast is it".
//!
//! One upstream landmine found along the way: grouped conv1d **backward** panics
//! when `stride > 1` (Burn 0.21, `conv1d_weight_grad_groups`, missing a guard the
//! 2-D and 3-D paths already have). WaveNet never strides, so it does not bite here
//! — it is pinned by a `should_panic` test at the bottom of this file so that a
//! future fix is noticed rather than the constraint becoming folklore.
//!
//! ## What this spike does NOT establish
//!
//! **Nothing here has run on a GPU.** This container has no CUDA device, so the
//! tests below use the `ndarray` backend. `cargo check --features cuda` passes, but
//! compiling is not running, and a conv kernel present on one backend and broken on
//! another is an ordinary occurrence — the strided-groups bug above is exactly that
//! shape, and it is a CPU-path bug.
//!
//! The `cuda_tests` module at the bottom closes this, and running it is one command
//! on a machine with a GPU:
//!
//! ```text
//! cargo test --features cuda
//! ```
//!
//! It re-checks forward, causality and the weight gradient at A2's real dilations
//! against the `ndarray` backend, which matters more than it sounds: Burn's own
//! suite tests dilation and groups separately and **never together**, so the
//! intersection a NAM stack depends on is uncovered upstream.
//!
//! Also not established: throughput. The plan's swap criterion is parity *and* at
//! least 0.9x PyTorch throughput, and this spike measures neither speed nor
//! agreement with PyTorch's numbers. It only shows the ops exist and differentiate
//! correctly.

use burn::nn::PaddingConfig1d;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::prelude::*;

/// Build a **causal** dilated (optionally grouped) 1-D convolution.
///
/// Causality is the property an amp model cannot be wrong about: output sample `t`
/// must depend on input samples `<= t` and nothing later. A model that peeks one
/// sample into the future trains to a beautiful loss and is unshippable, because at
/// runtime that sample does not exist yet.
///
/// The trick is asymmetric padding — all of it on the left:
///
/// ```text
/// left = (kernel_size - 1) * dilation      right = 0
/// ```
///
/// which also makes the output exactly as long as the input, so a stack of these
/// composes without any per-layer bookkeeping.
pub fn causal_conv1d_config(
    channels_in: usize,
    channels_out: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
) -> Conv1dConfig {
    Conv1dConfig::new(channels_in, channels_out, kernel_size)
        .with_dilation(dilation)
        .with_groups(groups)
        .with_padding(PaddingConfig1d::Explicit((kernel_size - 1) * dilation, 0))
}

/// A2's dilation schedule, mirroring `harness/reference.py`.
///
/// Three blocks of the 1,3,7,17,41,101,239 cycle with a 1,13 fine-detail interlude
/// between the second and third. The interlude's position is load-bearing: it lines
/// up with the two kernel-15 layers in [`A2_KERNEL_SIZES`], and moving it changes
/// the receptive field.
pub const A2_DILATIONS: [usize; 23] = [
    1, 3, 7, 17, 41, 101, 239, // block A
    1, 3, 7, 17, 41, 101, 239, // block B
    1, 13, // fine-detail interlude (the k=15 pair)
    1, 3, 7, 17, 41, 101, 239, // block C
];

/// A2's per-layer kernel sizes: 6 throughout except the interlude pair.
pub const A2_KERNEL_SIZES: [usize; 23] = [
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6,
];

/// Kernel size of A2's layer-array head *rechannel* convolution.
///
/// Worth stating separately because it is easy to lose: A2's documented 6,347-sample
/// receptive field is **not** the stack's. The stack spans 6,332; the rechannel
/// convolution adds the remaining 15. Anyone reconstructing A2 from the dilation
/// list alone lands 15 samples short and has no idea why.
pub const A2_HEAD_KERNEL_SIZE: usize = 16;

/// Reference `conv1d`, written the obvious way, for tests to disagree with.
///
/// Deliberately naive: explicit loops over batch, output channel, time and kernel
/// tap, with the group mapping spelled out. If this and Burn agree, the agreement
/// means something, because the two implementations share nothing.
///
/// Layout matches Burn's: input `[batch, channels_in, length]`, weight
/// `[channels_out, channels_in / groups, kernel_size]`.
// Indexing `weight[co][ci_local][k]` is the point: this is the definition of a
// grouped dilated convolution transcribed, and it is only useful as a cross-check if
// a reader can see the index arithmetic. Iterator-zipping it would make it resemble
// the implementation it exists to disagree with.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn naive_conv1d(
    input: &[Vec<Vec<f64>>],
    weight: &[Vec<Vec<f64>>],
    bias: Option<&[f64]>,
    dilation: usize,
    groups: usize,
    pad_left: usize,
    pad_right: usize,
) -> Vec<Vec<Vec<f64>>> {
    let batch = input.len();
    let channels_in = input[0].len();
    let length = input[0][0].len();
    let channels_out = weight.len();
    let kernel_size = weight[0][0].len();

    let in_per_group = channels_in / groups;
    let out_per_group = channels_out / groups;
    let span = dilation * (kernel_size - 1);
    let out_len = length + pad_left + pad_right - span;

    let mut out = vec![vec![vec![0.0_f64; out_len]; channels_out]; batch];

    for (b, out_b) in out.iter_mut().enumerate() {
        for (co, out_c) in out_b.iter_mut().enumerate() {
            let group = co / out_per_group;
            for (t, slot) in out_c.iter_mut().enumerate() {
                let mut acc = bias.map_or(0.0, |v| v[co]);
                for ci_local in 0..in_per_group {
                    let ci = group * in_per_group + ci_local;
                    for k in 0..kernel_size {
                        // Position in the *padded* signal, mapped back to the real one.
                        let pos = t + k * dilation;
                        if pos < pad_left {
                            continue; // left zero-pad
                        }
                        let idx = pos - pad_left;
                        if idx >= length {
                            continue; // right zero-pad
                        }
                        acc += weight[co][ci_local][k] * input[b][ci][idx];
                    }
                }
                *slot = acc;
            }
        }
    }
    out
}

/// Read a rank-3 tensor into nested `Vec`s so tests can index it readably.
pub fn to_nested<B: Backend>(t: &Tensor<B, 3>) -> Vec<Vec<Vec<f64>>> {
    let [batch, channels, length] = t.dims();
    let flat: Vec<f64> = t
        .to_data()
        .convert::<f64>()
        .into_vec()
        .expect("tensor -> vec");
    let mut out = vec![vec![vec![0.0; length]; channels]; batch];
    for b in 0..batch {
        for c in 0..channels {
            for l in 0..length {
                out[b][c][l] = flat[(b * channels + c) * length + l];
            }
        }
    }
    out
}

/// A minimal WaveNet-ish stack: causal dilated convs with a residual path.
///
/// Not NAM — no gating, no head, no conditioning. It exists to check that the ops
/// compose and differentiate at A2's depth and dilation range, which is where a
/// receptive field of 6,347 samples comes from and where a shape bug would surface.
#[derive(Module, Debug)]
pub struct DilatedStack<B: Backend> {
    layers: Vec<Conv1d<B>>,
}

impl<B: Backend> DilatedStack<B> {
    /// Build a stack over the given per-layer kernel sizes and dilations.
    ///
    /// Both are per-layer rather than uniform because A2 is not uniform — two of its
    /// 23 layers use kernel 15 while the rest use 6, and a spike that only ever
    /// exercises a uniform stack would not catch a shape bug that depends on them
    /// differing.
    ///
    /// # Panics
    ///
    /// If `kernel_sizes` and `dilations` have different lengths.
    pub fn new(
        device: &Device<B>,
        channels: usize,
        kernel_sizes: &[usize],
        groups: usize,
        dilations: &[usize],
    ) -> Self {
        assert_eq!(
            kernel_sizes.len(),
            dilations.len(),
            "one kernel size per dilation"
        );
        let layers = kernel_sizes
            .iter()
            .zip(dilations)
            .map(|(&k, &d)| causal_conv1d_config(channels, channels, k, d, groups).init(device))
            .collect();
        Self { layers }
    }

    /// Receptive field of the stack in samples: sum of per-layer spans, plus one.
    ///
    /// Note this is the *stack's* field. A2's documented 6,347 additionally includes
    /// its head rechannel convolution — see [`A2_HEAD_KERNEL_SIZE`].
    pub fn receptive_field(kernel_sizes: &[usize], dilations: &[usize]) -> usize {
        kernel_sizes
            .iter()
            .zip(dilations)
            .map(|(k, d)| d * (k - 1))
            .sum::<usize>()
            + 1
    }

    /// Forward with a residual connection per layer.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = x;
        for layer in &self.layers {
            // Every layer preserves length, so the residual add needs no alignment.
            h = layer.forward(h.clone()).tanh() + h;
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::{Distribution, TensorData};

    /// f64 throughout: the finite-difference test below needs the headroom, and a
    /// spike answering "is the gradient correct" should not be fighting f32 noise
    /// while it does so.
    type B = NdArray<f64>;
    type AD = Autodiff<B>;

    fn device() -> Device<B> {
        Default::default()
    }

    fn seeded<Bk: Backend>(shape: [usize; 3], seed: u64, device: &Device<Bk>) -> Tensor<Bk, 3> {
        Bk::seed(device, seed);
        Tensor::random(shape, Distribution::Normal(0.0, 1.0), device)
    }

    // ---------------------------------------------------------------------------
    // 1. The forward op exists and computes the right thing.
    // ---------------------------------------------------------------------------

    /// The headline question, forward half: groups > 1 and dilation > 1 together.
    ///
    /// Checked against `naive_conv1d` rather than against "it did not panic", which
    /// is the failure mode that matters — a silently wrong group mapping produces a
    /// plausible tensor of the right shape.
    #[test]
    fn grouped_dilated_conv1d_matches_the_naive_reference() {
        let device = device();
        let (channels_in, channels_out, kernel_size, dilation, groups) = (4, 6, 3, 4, 2);

        let conv = causal_conv1d_config(channels_in, channels_out, kernel_size, dilation, groups)
            .init::<B>(&device);

        let input = seeded::<B>([2, channels_in, 32], 7, &device);
        let output = conv.forward(input.clone());

        let expected = naive_conv1d(
            &to_nested(&input),
            &to_nested(&conv.weight.val()),
            Some(
                &conv
                    .bias
                    .as_ref()
                    .unwrap()
                    .val()
                    .to_data()
                    .convert::<f64>()
                    .into_vec::<f64>()
                    .unwrap(),
            ),
            dilation,
            groups,
            (kernel_size - 1) * dilation,
            0,
        );

        let got = to_nested(&output);
        assert_eq!(got.len(), expected.len());
        assert_eq!(got[0].len(), channels_out);
        assert_eq!(got[0][0].len(), 32, "causal padding must preserve length");

        for b in 0..got.len() {
            for c in 0..channels_out {
                for t in 0..got[b][c].len() {
                    let (a, e) = (got[b][c][t], expected[b][c][t]);
                    assert!(
                        (a - e).abs() < 1e-10,
                        "mismatch at [{b}][{c}][{t}]: burn={a} naive={e}"
                    );
                }
            }
        }
    }

    /// A grouped convolution must not mix channels across groups.
    ///
    /// Distinct from the test above: that one would still pass if *both*
    /// implementations shared a wrong-but-consistent idea of grouping. This asserts
    /// the structural property directly, by perturbing one group's input and
    /// requiring the other group's output to be bit-identical.
    #[test]
    fn groups_do_not_leak_across_the_channel_boundary() {
        let device = device();
        let (channels, groups) = (4, 2);
        let conv = causal_conv1d_config(channels, channels, 3, 4, groups).init::<B>(&device);

        let base = seeded::<B>([1, channels, 16], 11, &device);
        let before = to_nested(&conv.forward(base.clone()));

        // Perturb channel 0, which belongs to group 0 only.
        let mut perturbed = to_nested(&base);
        for v in perturbed[0][0].iter_mut() {
            *v += 5.0;
        }
        let flat: Vec<f64> = perturbed
            .iter()
            .flat_map(|b| b.iter().flat_map(|c| c.iter().copied()))
            .collect();
        let perturbed =
            Tensor::<B, 3>::from_data(TensorData::new(flat, [1, channels, 16]), &device);
        let after = to_nested(&conv.forward(perturbed));

        let out_per_group = channels / groups;
        for c in 0..channels {
            let same = (0..16).all(|t| (before[0][c][t] - after[0][c][t]).abs() < 1e-12);
            if c < out_per_group {
                assert!(!same, "channel {c} is in the perturbed group and should move");
            } else {
                assert!(same, "channel {c} is in another group and must not move");
            }
        }
    }

    /// The property an amp model cannot be wrong about.
    ///
    /// Changing the input strictly *after* time `t` must leave output `t` alone.
    /// A symmetric-padding mistake breaks this while leaving shapes and loss curves
    /// looking entirely healthy.
    #[test]
    fn causal_padding_does_not_leak_future_samples() {
        let device = device();
        let (length, split) = (64, 40);
        let conv = causal_conv1d_config(1, 1, 6, 17, 1).init::<B>(&device);

        let base = seeded::<B>([1, 1, length], 3, &device);
        let before = to_nested(&conv.forward(base.clone()));

        let mut tampered = to_nested(&base);
        for v in tampered[0][0].iter_mut().skip(split) {
            *v = 100.0; // wildly different, so any leak is obvious
        }
        let flat: Vec<f64> = tampered[0][0].clone();
        let tampered = Tensor::<B, 3>::from_data(TensorData::new(flat, [1, 1, length]), &device);
        let after = to_nested(&conv.forward(tampered));

        for t in 0..split {
            assert!(
                (before[0][0][t] - after[0][0][t]).abs() < 1e-12,
                "output {t} changed after editing input {split}.., so the conv sees the future"
            );
        }
        // And the change must actually reach the later outputs, or this test would
        // pass on a conv that ignores its input entirely.
        assert!(
            (before[0][0][split] - after[0][0][split]).abs() > 1e-6,
            "output at the edit point did not move; the test is not exercising anything"
        );
    }

    // ---------------------------------------------------------------------------
    // 2. The backward pass exists and is correct.
    // ---------------------------------------------------------------------------

    /// The headline question, backward half.
    ///
    /// Grouped convolution backward is a classic gap — present for `groups == 1`,
    /// `todo!()` otherwise. This asserts gradients are produced for both the input
    /// and the weight, with groups and dilation both engaged.
    #[test]
    fn gradients_exist_for_grouped_dilated_conv() {
        let device = device();
        let conv = causal_conv1d_config(4, 4, 3, 7, 2).init::<AD>(&device);

        let input = seeded::<AD>([2, 4, 24], 5, &device).require_grad();
        let loss = conv.forward(input.clone()).powi_scalar(2).sum();
        let grads = loss.backward();

        let grad_in = input.grad(&grads).expect("no gradient wrt input");
        let grad_w = conv.weight.val().grad(&grads).expect("no gradient wrt weight");

        assert_eq!(grad_in.dims(), [2, 4, 24]);
        assert_eq!(grad_w.dims(), conv.weight.val().dims());

        let finite_and_nonzero = |t: Tensor<B, 3>| {
            let v: Vec<f64> = t.to_data().convert::<f64>().into_vec().unwrap();
            v.iter().all(|x| x.is_finite()) && v.iter().any(|x| x.abs() > 1e-12)
        };
        assert!(finite_and_nonzero(grad_in), "input gradient is zero or non-finite");
        assert!(finite_and_nonzero(grad_w), "weight gradient is zero or non-finite");
    }

    /// The load-bearing test: is the gradient *right*, not merely present?
    ///
    /// Central differences against the analytic weight gradient. An autodiff rule
    /// that mishandles the group mapping or the dilation stride still returns a
    /// full, finite, plausible tensor — this is what distinguishes that from a
    /// correct one.
    #[test]
    fn weight_gradient_matches_central_differences() {
        let device = device();
        let (kernel_size, dilation, groups) = (3, 5, 2);
        let conv = causal_conv1d_config(4, 4, kernel_size, dilation, groups).init::<AD>(&device);
        let input = seeded::<AD>([1, 4, 20], 13, &device);

        // Scalar objective, so each weight has a well-defined scalar derivative.
        let loss = conv.forward(input.clone()).powi_scalar(2).sum();
        let grads = loss.backward();
        let analytic = to_nested(&conv.weight.val().grad(&grads).expect("weight gradient"));

        let w0 = to_nested(&conv.weight.val().inner());
        let input_inner = input.inner();
        let bias: Vec<f64> = conv
            .bias
            .as_ref()
            .unwrap()
            .val()
            .inner()
            .to_data()
            .convert::<f64>()
            .into_vec()
            .unwrap();
        let nested_input = to_nested(&input_inner);

        let objective = |w: &[Vec<Vec<f64>>]| -> f64 {
            naive_conv1d(
                &nested_input,
                w,
                Some(&bias),
                dilation,
                groups,
                (kernel_size - 1) * dilation,
                0,
            )
            .iter()
            .flat_map(|b| b.iter().flat_map(|c| c.iter()))
            .map(|v| v * v)
            .sum()
        };

        let eps = 1e-6;
        let mut checked = 0;
        for co in 0..w0.len() {
            for ci in 0..w0[co].len() {
                for k in 0..w0[co][ci].len() {
                    let mut plus = w0.clone();
                    let mut minus = w0.clone();
                    plus[co][ci][k] += eps;
                    minus[co][ci][k] -= eps;
                    let numeric = (objective(&plus) - objective(&minus)) / (2.0 * eps);
                    let exact = analytic[co][ci][k];
                    let scale = numeric.abs().max(exact.abs()).max(1.0);
                    assert!(
                        (numeric - exact).abs() / scale < 1e-6,
                        "grad mismatch at w[{co}][{ci}][{k}]: autodiff={exact} numeric={numeric}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no weights were checked");
    }

    // ---------------------------------------------------------------------------
    // 3. It composes at A2's real depth, and it trains.
    // ---------------------------------------------------------------------------

    /// A2's receptive field decomposes into a stack span plus a head span.
    ///
    /// The composition rule this asserts is the one the causal padding in
    /// [`causal_conv1d_config`] implements, so getting it wrong here would mean the
    /// stack silently sees a different amount of history than A2 does.
    #[test]
    fn the_a2_schedule_reaches_its_documented_receptive_field() {
        let stack_span: usize = A2_KERNEL_SIZES
            .iter()
            .zip(A2_DILATIONS.iter())
            .map(|(k, d)| d * (k - 1))
            .sum();

        assert_eq!(stack_span + 1, 6_332, "the stack alone spans 6,332 samples");

        // The remaining 15 come from the layer-array head rechannel convolution,
        // which is not part of the dilation list at all.
        assert_eq!(
            stack_span + 1 + (A2_HEAD_KERNEL_SIZE - 1),
            6_347,
            "A2's documented receptive field is stack + head rechannel span"
        );
    }

    /// The interlude's position matters, and this is what proves it.
    ///
    /// Guard on the constant above: if someone "tidies" the dilation list by moving
    /// the 1,13 pair to the front, it no longer aligns with the kernel-15 layers and
    /// the receptive field changes. Without this the reordering is invisible.
    #[test]
    fn moving_the_interlude_changes_the_receptive_field() {
        let mut reordered = A2_DILATIONS;
        reordered.rotate_right(9); // 1,13 to the front

        let span = |d: &[usize]| -> usize {
            A2_KERNEL_SIZES
                .iter()
                .zip(d.iter())
                .map(|(k, dd)| dd * (k - 1))
                .sum::<usize>()
                + 1
        };

        assert_ne!(
            span(&reordered),
            span(&A2_DILATIONS),
            "if this passes, the interlude's position is not load-bearing after all"
        );
    }

    /// Full A2 depth and width, with its real mixed kernel sizes.
    ///
    /// Length preservation across all 23 layers is the property that lets the
    /// residual adds in `forward` work without alignment bookkeeping; at dilation
    /// 239 with kernel 15, an off-by-one in the padding shows up here and nowhere
    /// cheaper.
    #[test]
    fn an_a2_shaped_stack_runs_and_preserves_length() {
        let device = device();
        let stack = DilatedStack::<B>::new(&device, 8, &A2_KERNEL_SIZES, 1, &A2_DILATIONS);

        // Comfortably longer than the 6,332-sample stack span, so the deepest layers
        // are doing real work rather than reading padding.
        let length = 8_192;
        assert!(length > DilatedStack::<B>::receptive_field(&A2_KERNEL_SIZES, &A2_DILATIONS));

        let out = stack.forward(seeded::<B>([1, 8, length], 17, &device));
        assert_eq!(out.dims(), [1, 8, length]);
    }

    /// End to end: gradients reach every layer of a deep grouped+dilated stack, and
    /// one optimizer step reduces the loss. This is the claim "Burn can train this
    /// architecture", reduced to something that either happens or does not.
    #[test]
    fn a_grouped_dilated_stack_takes_a_training_step() {
        use burn::optim::{AdamConfig, GradientsParams, Optimizer};

        let device = device();
        let mut stack = DilatedStack::<AD>::new(&device, 4, &[3, 3, 3, 3], 2, &[1, 3, 7, 17]);
        let mut optimizer = AdamConfig::new().init();

        let input = seeded::<AD>([2, 4, 256], 23, &device);
        let target = seeded::<AD>([2, 4, 256], 29, &device);

        let initial = stack
            .forward(input.clone())
            .sub(target.clone())
            .powi_scalar(2)
            .mean();
        let initial_value: f64 = initial.clone().into_scalar();

        for _ in 0..5 {
            let loss = stack
                .forward(input.clone())
                .sub(target.clone())
                .powi_scalar(2)
                .mean();
            let grads = GradientsParams::from_grads(loss.backward(), &stack);
            stack = optimizer.step(1e-2, stack, grads);
        }

        let final_value: f64 = stack
            .forward(input)
            .sub(target)
            .powi_scalar(2)
            .mean()
            .into_scalar();

        assert!(
            final_value < initial_value,
            "loss did not decrease: {initial_value} -> {final_value}"
        );
    }

    // ---------------------------------------------------------------------------
    // 4. A landmine in Burn 0.21, pinned so we notice when it is defused.
    // ---------------------------------------------------------------------------

    /// **Upstream bug in Burn 0.21**: grouped conv1d backward panics when strided.
    ///
    /// `conv1d_weight_grad_groups` in `burn-backend` is missing the kernel-size
    /// truncation guard that the 2-D and 3-D versions received in tracel-ai/burn
    /// PR #3521 (issue #3511). The grouped weight-gradient convolution returns a
    /// kernel longer than the real one, and `slice_assign` gets a shape it cannot
    /// broadcast:
    ///
    /// ```text
    /// ndarray: could not broadcast array from shape: [2, 2, 4] to: [2, 2, 3]
    ///   at burn-backend-0.21.0/src/backend/ops/modules/conv.rs:951
    /// ```
    ///
    /// The trigger, mapped by running the grid rather than reasoning about it:
    ///
    /// | groups | stride | padding | backward |
    /// |---|---|---|---|
    /// | 2 | 1 | any of (0,0) (1,1) (2,0) | ok |
    /// | 2 | 2 | any of (0,0) (1,1) (2,0) | **panic** |
    /// | 1 | 2 | any | ok |
    ///
    /// So it is `groups > 1` **and** `stride > 1`. Padding symmetry is irrelevant —
    /// worth stating because the obvious hypothesis, that the asymmetric path's
    /// outer `pad` op dodges it by handing the backend `padding = [0]`, is wrong.
    /// It was written into this file as fact until the companion test below failed.
    ///
    /// **Why it does not affect this project**: WaveNet does not stride. Every
    /// convolution in a NAM-shaped stack is stride 1, which is the column that
    /// works, and the grouped+dilated backward tests above all pass.
    ///
    /// Pinned as `should_panic` so that a future Burn fixing this fails the test and
    /// tells us, rather than the constraint quietly becoming folklore. Worth an
    /// upstream report — the 1-D case looks simply to have been missed.
    #[test]
    #[should_panic(expected = "could not broadcast")]
    fn grouped_strided_conv1d_backward_hits_an_upstream_slice_bug() {
        let device: Device<AD> = Default::default();

        let conv = Conv1dConfig::new(4, 4, 3)
            .with_groups(2)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .init::<AD>(&device);

        let input = seeded::<AD>([1, 4, 16], 31, &device).require_grad();
        let loss = conv.forward(input).powi_scalar(2).sum();
        let _ = loss.backward();
    }

    /// The controls for the test above: same grouping, stride 1, both padding
    /// symmetries. Without these, `should_panic` could be read as "grouped conv
    /// backward is broken in Burn", which is the wrong conclusion and would sink a
    /// viable plan.
    #[test]
    fn grouped_unstrided_conv1d_backward_is_fine_at_either_padding() {
        let device: Device<AD> = Default::default();

        for padding in [PaddingConfig1d::Explicit(1, 1), PaddingConfig1d::Explicit(2, 0)] {
            let conv = Conv1dConfig::new(4, 4, 3)
                .with_groups(2)
                .with_stride(1)
                .with_padding(padding.clone())
                .init::<AD>(&device);

            let input = seeded::<AD>([1, 4, 16], 31, &device).require_grad();
            let loss = conv.forward(input.clone()).powi_scalar(2).sum();
            let grads = loss.backward();

            assert!(
                conv.weight.val().grad(&grads).is_some(),
                "weight gradient missing for padding {padding:?}"
            );
            assert!(
                input.grad(&grads).is_some(),
                "input gradient missing for padding {padding:?}"
            );
        }
    }
}


/// CUDA verification, compiled only with `--features cuda`.
///
/// The reason this exists as its own module rather than a backend swap on the tests
/// above: Burn's own test suite covers `dilation > 1, groups = 1` and
/// `dilation = 1, groups > 1` on CUDA, but **nothing anywhere exercises both at
/// once** — not for conv1d, not for conv2d, forward or backward. The kernels treat
/// the two features orthogonally and reading them suggests it is fine, but a NAM
/// stack leans on exactly that untested intersection at dilations up to 239.
///
/// So this checks it by running it, on the machine that will do the training:
///
/// ```text
/// cargo test --features cuda -- --nocapture
/// ```
///
/// The gradient check is cross-backend — same weights, same input, CUDA against
/// `ndarray` — which is stronger than a self-consistency check and does not require
/// reimplementing the backward pass.
#[cfg(feature = "cuda")]
#[cfg(test)]
mod cuda_tests {
    use super::*;
    use burn::backend::{Autodiff, Cuda, NdArray};
    use burn::module::Param;
    use burn::tensor::TensorData;

    type Cpu = NdArray<f32>;
    type Gpu = Cuda<f32>;

    /// f32 on both sides, in different association orders, through a cascade of
    /// convolutions. This is engineering agreement, not bit reproduction.
    const TOLERANCE: f32 = 1e-4;

    /// Deterministic weights, so both backends get byte-identical inputs without
    /// depending on the two RNGs agreeing.
    fn ramp(n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.37).sin() * scale)
            .collect()
    }

    struct Case {
        channels: usize,
        kernel: usize,
        dilation: usize,
        groups: usize,
        length: usize,
    }

    fn build<B: Backend>(case: &Case, device: &Device<B>) -> (Conv1d<B>, Tensor<B, 3>) {
        let mut conv = causal_conv1d_config(
            case.channels,
            case.channels,
            case.kernel,
            case.dilation,
            case.groups,
        )
        .init::<B>(device);

        let w_shape = [case.channels, case.channels / case.groups, case.kernel];
        let w_len = w_shape.iter().product::<usize>();
        conv.weight = Param::from_tensor(Tensor::<B, 3>::from_data(
            TensorData::new(ramp(w_len, 0.5), w_shape),
            device,
        ));
        conv.bias = conv.bias.map(|_| {
            Param::from_tensor(Tensor::<B, 1>::from_data(
                TensorData::new(ramp(case.channels, 0.1), [case.channels]),
                device,
            ))
        });

        let x_shape = [1, case.channels, case.length];
        let input = Tensor::<B, 3>::from_data(
            TensorData::new(ramp(case.channels * case.length, 1.0), x_shape),
            device,
        );
        (conv, input)
    }

    /// A2's real dilations, paired with group counts a NAM variant might use.
    fn cases() -> Vec<Case> {
        let mut out = Vec::new();
        for &dilation in &[1usize, 3, 17, 101, 239] {
            for &groups in &[1usize, 2, 4] {
                out.push(Case {
                    channels: 8,
                    kernel: 6,
                    dilation,
                    groups,
                    length: 1024,
                });
            }
        }
        out
    }

    #[test]
    fn forward_agrees_with_the_cpu_backend() {
        let (gpu, cpu): (Device<Gpu>, Device<Cpu>) = (Default::default(), Default::default());

        for case in cases() {
            let (conv_gpu, x_gpu) = build::<Gpu>(&case, &gpu);
            let (conv_cpu, x_cpu) = build::<Cpu>(&case, &cpu);

            let got = to_nested(&conv_gpu.forward(x_gpu));
            let want = to_nested(&conv_cpu.forward(x_cpu));

            for c in 0..case.channels {
                for t in 0..case.length {
                    let (a, e) = (got[0][c][t], want[0][c][t]);
                    assert!(
                        (a - e).abs() <= f64::from(TOLERANCE) * e.abs().max(1.0),
                        "dilation={} groups={}: forward mismatch at [{c}][{t}]: cuda={a} cpu={e}",
                        case.dilation,
                        case.groups
                    );
                }
            }
        }
    }

    /// The one the plan actually rides on: grouped **and** dilated backward, on the
    /// hardware, checked against a backend that is not it.
    #[test]
    fn weight_gradients_agree_with_the_cpu_backend() {
        let (gpu, cpu): (Device<Autodiff<Gpu>>, Device<Autodiff<Cpu>>) =
            (Default::default(), Default::default());

        for case in cases() {
            let (conv_gpu, x_gpu) = build::<Autodiff<Gpu>>(&case, &gpu);
            let (conv_cpu, x_cpu) = build::<Autodiff<Cpu>>(&case, &cpu);

            let g_gpu = conv_gpu.forward(x_gpu).powi_scalar(2).sum().backward();
            let g_cpu = conv_cpu.forward(x_cpu).powi_scalar(2).sum().backward();

            let got = to_nested(&conv_gpu.weight.val().grad(&g_gpu).expect("cuda weight grad"));
            let want = to_nested(&conv_cpu.weight.val().grad(&g_cpu).expect("cpu weight grad"));

            for co in 0..got.len() {
                for ci in 0..got[co].len() {
                    for k in 0..got[co][ci].len() {
                        let (a, e) = (got[co][ci][k], want[co][ci][k]);
                        let scale = a.abs().max(e.abs()).max(1.0);
                        assert!(
                            (a - e).abs() / scale <= f64::from(TOLERANCE),
                            "dilation={} groups={}: grad mismatch at w[{co}][{ci}][{k}]: \
                             cuda={a} cpu={e}",
                            case.dilation,
                            case.groups
                        );
                    }
                }
            }
        }
    }

    /// Causality on the GPU. Same property as the CPU test, restated because a
    /// padding bug in a different kernel is a different bug.
    #[test]
    fn causal_padding_holds_on_the_gpu() {
        let device: Device<Gpu> = Default::default();
        let case = Case {
            channels: 4,
            kernel: 6,
            dilation: 17,
            groups: 2,
            length: 512,
        };
        let (conv, input) = build::<Gpu>(&case, &device);

        let before = to_nested(&conv.forward(input.clone()));

        let split = 300;
        let mut tampered = to_nested(&input);
        for channel in tampered[0].iter_mut() {
            for v in channel.iter_mut().skip(split) {
                *v = 100.0;
            }
        }
        let flat: Vec<f32> = tampered[0]
            .iter()
            .flat_map(|c| c.iter().map(|v| *v as f32))
            .collect();
        let tampered = Tensor::<Gpu, 3>::from_data(
            TensorData::new(flat, [1, case.channels, case.length]),
            &device,
        );
        let after = to_nested(&conv.forward(tampered));

        for c in 0..case.channels {
            for t in 0..split {
                assert!(
                    (before[0][c][t] - after[0][c][t]).abs() <= f64::from(TOLERANCE),
                    "gpu conv sees the future at [{c}][{t}]"
                );
            }
        }
    }
}
