use fast_audio::{BinaryOperand as Side, GradientMode as G, Layout, Tensor};
use std::collections::BTreeSet;

pub const SIDES: [Side; 2] = [Side::Left, Side::Right];
pub const SENTINEL: u32 = 0x7fc12345;
#[derive(Clone)]
pub struct Case {
    pub dims: [usize; 3],
    pub a: Layout,
    pub b: Layout,
    pub y: Layout,
    pub dy: Layout,
    pub da: Layout,
    pub db: Layout,
    pub direct: bool,
}
impl Case {
    pub fn gradient(&self, side: Side) -> &Layout {
        if side == Side::Left {
            &self.da
        } else {
            &self.db
        }
    }
    pub fn dependencies(&self, side: Side) -> [bool; 2] {
        let active = self.dims.iter().all(|&n| n > 0);
        [active && side == Side::Right, active && side == Side::Left]
    }
}
pub fn layout(rows: usize, cols: usize, style: usize) -> Layout {
    if rows == 0 || cols == 0 {
        return Layout::from_strides([rows, cols], [isize::MIN; 2], 3).unwrap();
    }
    let (mut steps, origin) = match style % 9 {
        0 => ([cols as isize, 1], 3),
        1 => ([(cols + 3) as isize, 1], 2),
        2 => ([1, (rows + 3) as isize], 1),
        3 => ([-((cols + 3) as isize), 1], 2 + (rows - 1) * (cols + 3)),
        4 => ([(cols * 2 + 3) as isize, -2], 2 + (cols - 1) * 2),
        5 => ([-2, (rows * 2 + 3) as isize], 3 + (rows - 1) * 2),
        6 => ([0, 1], 3),
        7 => ([1, 1], 1),
        _ => ([0, 0], 1),
    };
    if rows == 1 {
        steps[0] = isize::MIN;
    }
    if cols == 1 {
        steps[1] = isize::MIN;
    }
    Layout::from_strides([rows, cols], steps, origin).unwrap()
}
pub fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for dims in [
        [1, 1, 1],
        [1, 7, 3],
        [5, 1, 9],
        [3, 4, 1],
        [2, 3, 4],
        [3, 5, 7],
        [5, 17, 16],
        [17, 3, 33],
        [9, 5, 65],
        [3, 7, 127],
        [2, 5, 128],
        [3, 2, 129],
        [7, 3, 257],
        [2, 1, 257],
        [0, 3, 4],
        [3, 0, 4],
        [3, 4, 0],
        [0, 3, 0],
        [3, 0, 0],
        [0, 0, 3],
        [3, 5, 15],
        [3, 5, 17],
        [3, 5, 31],
        [3, 5, 32],
        [3, 5, 63],
        [3, 5, 64],
        [3, 5, 255],
        [3, 5, 256],
        [33, 65, 17],
        [129, 2, 3],
        [2, 129, 3],
    ] {
        let [m, n, k] = dims;
        for s in 0..12 {
            out.push(Case {
                dims,
                a: layout(m, k, s),
                b: layout(k, n, s * 5 + 1),
                y: layout(m, n, (s * 7 + 1) % 6),
                dy: layout(m, n, s * 7 + 3),
                da: layout(m, k, (s * 5 + 2) % 6),
                db: layout(k, n, (s * 3 + 4) % 6),
                direct: false,
            });
        }
    }
    for dims in [[3, 5, 7], [5, 17, 33]] {
        let [m, n, k] = dims;
        for flags in 0..8 {
            out.push(Case {
                dims,
                a: layout(m, k, if flags & 1 == 0 { 0 } else { 2 }),
                b: layout(k, n, if flags & 2 == 0 { 0 } else { 2 }),
                y: layout(m, n, if flags & 4 == 0 { 0 } else { 2 }),
                dy: layout(m, n, if flags & 1 == 0 { 0 } else { 2 }),
                da: layout(m, k, if flags & 2 == 0 { 0 } else { 2 }),
                db: layout(k, n, if flags & 4 == 0 { 0 } else { 2 }),
                direct: true,
            });
        }
    }
    out.push(Case {
        dims: [33, 65, 129],
        a: layout(33, 129, 5),
        b: layout(129, 65, 3),
        y: layout(33, 65, 4),
        dy: layout(33, 65, 7),
        da: layout(33, 129, 3),
        db: layout(129, 65, 5),
        direct: false,
    });
    assert_eq!(out.len(), 389);
    out
}
pub fn address(l: &Layout, i: usize, j: usize) -> usize {
    (l.offset() as i128 + i as i128 * l.strides()[0] as i128 + j as i128 * l.strides()[1] as i128)
        as usize
}
pub fn addresses(l: &Layout) -> Vec<usize> {
    let dims = l.shape().dims();
    if l.shape().is_empty() {
        return Vec::new();
    }
    (0..dims[0])
        .flat_map(|i| (0..dims[1]).map(move |j| address(l, i, j)))
        .collect()
}
pub fn buffer(l: &Layout, salt: usize) -> Tensor {
    let len = l.span().map_or(l.offset(), |s| s.end) + 2;
    Tensor::from_vec(
        [len],
        (0..len)
            .map(|i| (((i * 13 + salt * 17) % 97) as f32 - 48.) / 97.)
            .collect(),
    )
    .unwrap()
}
pub fn seed(l: &Layout, finite: bool) -> Tensor {
    let mut t = buffer(l, 0);
    t.data_mut().fill(f32::from_bits(SENTINEL));
    if finite {
        for (i, a) in addresses(l).into_iter().enumerate() {
            t.data_mut()[a] = 0.2 + (i % 7) as f32 / 13.;
        }
    }
    t
}
pub fn bits(t: &Tensor) -> Vec<u32> {
    t.data().iter().map(|x| x.to_bits()).collect()
}
pub fn logical(l: &Layout, t: &Tensor) -> Vec<f64> {
    addresses(l)
        .into_iter()
        .map(|a| t.data()[a] as f64)
        .collect()
}
pub fn forward(dims: [usize; 3], a: &[f64], b: &[f64]) -> Vec<f64> {
    let [m, n, k] = dims;
    let mut out = vec![0.; m * n];
    if m == 0 || n == 0 {
        return out;
    }
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                out[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    out
}
pub fn derivative(dims: [usize; 3], side: Side, a: &[f64], b: &[f64], dy: &[f64]) -> Vec<f64> {
    let [m, n, k] = dims;
    let mut out = vec![0.; if side == Side::Left { m * k } else { k * n }];
    if out.is_empty() {
        return out;
    }
    if side == Side::Left {
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    out[i * k + p] += dy[i * n + j] * b[p * n + j];
                }
            }
        }
    } else {
        for p in 0..k {
            for j in 0..n {
                for i in 0..m {
                    out[p * n + j] += a[i * k + p] * dy[i * n + j];
                }
            }
        }
    }
    out
}
pub fn close(actual: f32, want: f64) {
    assert!(
        (actual as f64 - want).abs() <= 2e-5 + 2e-5 * want.abs(),
        "{actual} != {want}"
    );
}
pub fn holes(l: &Layout, initial: &Tensor, actual: &Tensor) {
    let used: BTreeSet<_> = addresses(l).into_iter().collect();
    for (i, (&a, &b)) in initial.data().iter().zip(actual.data()).enumerate() {
        if !used.contains(&i) {
            assert_eq!(a.to_bits(), b.to_bits(), "hole {i}");
        }
    }
}
pub fn verify(
    l: &Layout,
    expected: &[f64],
    initial: &Tensor,
    actual: &Tensor,
    mode: G,
    zero: bool,
) {
    assert_eq!(expected.len(), l.shape().len());
    for (&index, &want) in addresses(l).iter().zip(expected) {
        close(
            actual.data()[index],
            want + if mode == G::Accumulate {
                initial.data()[index] as f64
            } else {
                0.
            },
        );
        if zero && mode == G::Overwrite {
            assert_eq!(actual.data()[index].to_bits(), 0);
        }
    }
    holes(l, initial, actual);
}
pub fn identities(
    c: &Case,
    side: Side,
    a: &Tensor,
    b: &Tensor,
    y: &Tensor,
    dy: &Tensor,
    gradient: &Tensor,
) {
    let aa = logical(&c.a, a);
    let bb = logical(&c.b, b);
    let dd = logical(&c.dy, dy);
    let gg = logical(c.gradient(side), gradient);
    let lhs: f64 = logical(&c.y, y).iter().zip(&dd).map(|(y, d)| y * d).sum();
    let input = if side == Side::Left { &aa } else { &bb };
    let rhs: f64 = input.iter().zip(&gg).map(|(x, g)| x * g).sum();
    assert!(
        (lhs - rhs).abs() <= 2e-5 + 2e-5 * lhs.abs(),
        "adjoint {lhs} != {rhs}"
    );
    if input.is_empty() {
        return;
    }
    for i in [0, input.len() / 2, input.len() - 1] {
        let mut x = input.clone();
        x[i] += 1e-4;
        let objective = |x: &[f64]| {
            let value = if side == Side::Left {
                forward(c.dims, x, &bb)
            } else {
                forward(c.dims, &aa, x)
            };
            value.iter().zip(&dd).map(|(y, d)| y * d).sum::<f64>()
        };
        let plus = objective(&x);
        x[i] -= 2e-4;
        close(gg[i] as f32, (plus - objective(&x)) / 2e-4);
    }
}

/// Optional raw-bit export for a separate pinned Torch process; outside hot guards.
pub struct Exporter {
    writer: Option<std::io::BufWriter<std::fs::File>>,
}
pub struct Sample<'a> {
    pub index: usize,
    pub binding: usize,
    pub backend: &'a str,
    pub case: &'a Case,
    pub inputs: [&'a Tensor; 3],
    pub output: &'a Tensor,
    /// Overwrite then accumulate; each contains left then right gradients.
    pub gradients: [[&'a Tensor; 2]; 2],
}
fn logical_bits(layout: &Layout, tensor: &Tensor) -> Vec<u32> {
    addresses(layout)
        .into_iter()
        .map(|a| tensor.data()[a].to_bits())
        .collect()
}

pub fn training_fixture() -> [Tensor; 4] {
    [
        Tensor::from_vec([3, 2], vec![0.2, -0.3, 0.4, 0.1, -0.2, 0.5]).unwrap(),
        Tensor::from_vec([3, 2], vec![0.05, -0.1, 0.2, 0.15, -0.05, 0.25]).unwrap(),
        Tensor::from_vec([2, 2], vec![0.1, -0.2, 0.3, 0.4]).unwrap(),
        Tensor::from_vec([], vec![0.05]).unwrap(),
    ]
}
/// Independent analytic MSE/SGD control; every parameter step stays in FP64.
pub fn training_control(
    weights: &mut [f64],
    x: &Tensor,
    target: &Tensor,
    rate: f64,
) -> (f64, Vec<f64>) {
    let xx: Vec<_> = x.data().iter().map(|&v| v as f64).collect();
    let output = forward([3, 2, 2], &xx, weights);
    let residual: Vec<_> = output
        .iter()
        .zip(target.data())
        .map(|(&y, &t)| y - t as f64)
        .collect();
    let loss = residual.iter().map(|r| r * r).sum::<f64>() / 6.;
    let dy: Vec<_> = residual.iter().map(|r| 2. * r / 6.).collect();
    let grad = derivative([3, 2, 2], Side::Right, &xx, weights, &dy);
    for (w, &g) in weights.iter_mut().zip(&grad) {
        *w -= rate * g;
    }
    (loss, grad)
}
pub fn export_training(
    export: &mut Exporter,
    backend: &str,
    step: usize,
    inputs: [&Tensor; 4],
    loss: &Tensor,
    gradient: &Tensor,
    updated: &Tensor,
) {
    if !export.enabled() {
        return;
    }
    export.record(serde_json::json!({"schema_version":1,"kind":"sgd","backend":backend,"step":step,"steps":8,"dims":[3,2,2],
        "x":bits(inputs[0]),"target":bits(inputs[1]),"weights":bits(inputs[2]),"rate":bits(inputs[3]),
        "loss":bits(loss),"gradient":bits(gradient),"updated_weights":bits(updated)}));
}
impl Exporter {
    pub fn new(stem: Option<&str>) -> Self {
        let directory = stem.and_then(|s| {
            std::env::var_os("FAST_AUDIO_MATMUL_EXPORT_DIR")
                .map(|p| (std::path::PathBuf::from(p), s))
        });
        let writer = directory.map(|(dir, stem)| {
            std::fs::create_dir_all(&dir).unwrap();
            std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(dir.join(format!("{stem}.jsonl")))
                    .unwrap(),
            )
        });
        Self { writer }
    }
    pub fn enabled(&self) -> bool {
        self.writer.is_some()
    }
    pub fn record(&mut self, value: serde_json::Value) {
        use std::io::Write;
        if let Some(writer) = self.writer.as_mut() {
            serde_json::to_writer(&mut *writer, &value).unwrap();
            writer.write_all(b"\n").unwrap();
        }
    }
    pub fn matrix(&mut self, s: Sample<'_>) {
        if !self.enabled() {
            return;
        }
        let c = s.case;
        self.record(serde_json::json!({"schema_version":1,"kind":"matrix","backend":s.backend,
            "case_index":s.index,"binding":s.binding,"dims":c.dims,
            "a":logical_bits(&c.a,s.inputs[0]),"b":logical_bits(&c.b,s.inputs[1]),"upstream":logical_bits(&c.dy,s.inputs[2]),
            "output":logical_bits(&c.y,s.output),"left_gradient":logical_bits(&c.da,s.gradients[0][0]),"right_gradient":logical_bits(&c.db,s.gradients[0][1]),
            "left_accumulate":logical_bits(&c.da,s.gradients[1][0]),"right_accumulate":logical_bits(&c.db,s.gradients[1][1]),
            "left_seed":logical_bits(&c.da,&seed(&c.da,true)),"right_seed":logical_bits(&c.db,&seed(&c.db,true))}));
    }
    pub fn finish(mut self) {
        use std::io::Write;
        if let Some(writer) = self.writer.as_mut() {
            writer.flush().unwrap();
        }
    }
}
