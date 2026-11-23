use fast_audio::{Tensor, cpu::Cpu};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

struct CountingAllocator;
thread_local! { static ALLOCS: Cell<Option<usize>> = const { Cell::new(None) }; }
// SAFETY: delegates every allocation and deallocation to System with unchanged arguments.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.with(|counter| {
            if let Some(n) = counter.get() {
                counter.set(Some(n + 1));
            }
        });
        // SAFETY: forwards the allocator contract and layout without modification.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr and layout are those received from the caller of this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
pub fn no_allocations(run: impl FnOnce()) {
    ALLOCS.with(|counter| counter.set(Some(0)));
    run();
    let count = ALLOCS.with(|counter| counter.replace(None)).unwrap();
    assert_eq!(count, 0, "prepared execution allocated Rust heap storage");
}
pub fn backends() -> Vec<Cpu> {
    #[allow(unused_mut)]
    let mut out = vec![Cpu::Reference];
    #[cfg(feature = "mkl")]
    out.push(Cpu::mkl(1).unwrap());
    out
}
pub fn data(dims: impl AsRef<[usize]>) -> Tensor {
    let mut t = Tensor::zeros(dims).unwrap();
    for (i, value) in t.data_mut().iter_mut().enumerate() {
        *value = ((i * 37 + 11) % 101) as f32 / 51.0 - 1.0;
    }
    t
}
pub fn close(actual: &[f32], expected: &[f64], atol: f64, rtol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&x, &y)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (x as f64 - y).abs() <= atol + rtol * y.abs(),
            "element {i}: {x} != {y}"
        );
    }
}
